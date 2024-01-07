import os
import shutil
import sys
from pathlib import Path
import time
import torch

import pandas as pd
import geopandas as gpd

file = Path(__file__).resolve()
root = str(file).split('Crop2Seg')[0]
sys.path.append(root)
os.chdir(os.path.join(root, 'Crop2Seg'))
# --------------------------------------

from streamlit_option_menu import option_menu

import streamlit as st

from src.webapp.home import home
from src.webapp.postprocess import cropmap
from src.webapp.get_data import get_info, get_ts, get_LPIS
from src.webapp.prediction import generate_prediction
from src.webapp.cache_management import cache_mgmt
from src.helpers.postprocess import homogenize

from src.global_vars import ACCOUNT, PASSWORD

if os.name == 'nt':
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    except:
        pass

st.set_page_config(page_title="Crop2Seg", layout="wide")

# A dictionary of apps in the format of {"App title": "App icon"}
# More icons can be found here: https://icons.getbootstrap.com


apps = [
    {"func": home, "title": "Home", "icon": "house"},
    {"func": cropmap, "title": "Crop map", "icon": "map"},
    {"func": cache_mgmt, "title": "Cache", "icon": "gear"}
]

titles = [app["title"] for app in apps]
titles_lower = [title.lower() for title in titles]
icons = [app["icon"] for app in apps]

params = st.experimental_get_query_params()
default_index = 0
try:
    if st.session_state['locked']:
        default_index = 1
    else:
        default_index = 0
except:
    if "page" in params:
        default_index = int(titles_lower.index(params["page"][0].lower()))
    else:
        default_index = 0

with st.sidebar:
    selected = option_menu(
        "Main Menu",
        options=titles,
        icons=icons,
        menu_icon="cast",
        default_index=default_index,
    )
    st.sidebar.title("How to use it:")
    st.sidebar.info(
        """
        1. Click on Home tab.
        2. Click on map to choose patch in Czech republic
        3. Choose year and specify temporal window for which prediction should be performed
        4. Click on Predict crops button
        5. Fill in your OpenSearch API credentials if needed (used for data download)
        6. Wait until download and prediction are conducted
        7. After successful prediction go to Crop Map tab (left sidebar)
        8. Explore predicted layer 

    """
    )

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This web app is part of project for crop type prediction within Czech republic based on
        yearly time-series of Sentinel-2 images.
            [GitHub](https://github.com/Many98/Crop2Seg).
    """
    )

if 'authorized' not in st.session_state:
    st.session_state['authorized'] = True

if 'show_credentials' not in st.session_state:
    st.session_state['show_credentials'] = False

if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False

if 'show_crop_map' not in st.session_state:
    st.session_state['show_crop_map'] = False

if 'year' not in st.session_state:
    st.session_state['year'] = 2019

if 'start_dt' not in st.session_state:
    st.session_state['start_dt'] = '3-2019'

if 'end_dt' not in st.session_state:
    st.session_state['end_dt'] = '9-2019'

if 'patch' not in st.session_state:
    st.session_state['patch'] = None

if 'last_click' not in st.session_state:
    st.session_state['last_click'] = None

if 'patch_error' not in st.session_state:
    st.session_state['patch_error'] = False

if 'lpis_enabled' not in st.session_state:
    st.session_state['lpis_enabled'] = True

if 'cache_enabled' not in st.session_state:
    st.session_state['cache_enabled'] = True

if 'password' not in st.session_state:
    st.session_state['password'] = PASSWORD

if 'account' not in st.session_state:
    st.session_state['account'] = ACCOUNT

if 'run_pipeline' not in st.session_state:
    st.session_state['run_pipeline'] = False

if 'locked' not in st.session_state:
    st.session_state['locked'] = False

if selected == 'Home':
    opts = {f'9-{st.session_state["year"] - 1}': 0, f'10-{st.session_state["year"] - 1}': 1,
            f'11-{st.session_state["year"] - 1}': 2, f'12-{st.session_state["year"] - 1}': 3,
            f'1-{st.session_state["year"]}': 4, f'2-{st.session_state["year"]}': 5, f'3-{st.session_state["year"]}': 6,
            f'4-{st.session_state["year"]}': 7, f'5-{st.session_state["year"]}': 8,
            f'6-{st.session_state["year"]}': 9, f'7-{st.session_state["year"]}': 10,
            f'8-{st.session_state["year"]}': 11, f'9-{st.session_state["year"]}': 12,
            f'10-{st.session_state["year"]}': 13}
    placeholder = st.empty()

    with placeholder.container():
        if not st.session_state['run_pipeline']:
            if not st.session_state['locked']:
                out = home()
            if st.session_state['show_credentials']:
                st.warning(f'Unauthorized access to OpenSearch API. \n'
                           f'Please specify correct credentials for https://dhr1.cesnet.cz/')
                user_name = st.text_input("Enter a OpenSearch API username", placeholder='your username',
                                          on_change=time.sleep(0.1),
                                          disabled=st.session_state['locked'] and st.session_state['authorized'])
                password = st.text_input("Enter a OpenSearch API password", type="password",
                                         placeholder='your password',
                                         on_change=time.sleep(0.1),
                                         disabled=st.session_state['locked'] and st.session_state['authorized'])
                st.session_state['account'] = user_name
                st.session_state['password'] = password
                st.session_state['authorized'] = True
            col1, col2 = st.columns([0.3, 0.7])

            with col1:
                year = st.slider('Year of prediction', 2015, 2023, st.session_state["year"],
                                 help='Specify year for which crop map should be generated',
                                 disabled=st.session_state['locked'])
                change_year = int(year) - int(st.session_state['year'])
                st.session_state['start_dt'] = st.session_state['start_dt'].split('-')[
                                                   0] + f"-{int(st.session_state['start_dt'].split('-')[1]) + change_year}"
                st.session_state['end_dt'] = st.session_state['end_dt'].split('-')[
                                                 0] + f"-{int(st.session_state['end_dt'].split('-')[1]) + change_year}"
                st.session_state['year'] = year

            with col2:
                opts = {f'9-{year - 1}': 0, f'10-{year - 1}': 1, f'11-{year - 1}': 2, f'12-{year - 1}': 3,
                        f'1-{year}': 4, f'2-{year}': 5, f'3-{year}': 6, f'4-{year}': 7, f'5-{year}': 8,
                        f'6-{year}': 9, f'7-{year}': 10, f'8-{year}': 11, f'9-{year}': 12, f'10-{year}': 13}

                start_dt, end_dt = st.select_slider(
                    'Temporal window', options=list(opts.keys()),
                    value=(f"{st.session_state['start_dt']}", f"{st.session_state['end_dt']}"),
                    help='Specify range within chosen year',
                    disabled=st.session_state['locked'])
                st.session_state['start_dt'] = start_dt
                st.session_state['end_dt'] = end_dt

            st.write(f'You selected year {st.session_state["year"]} with temporal window between',
                     st.session_state['start_dt'],
                     'and', st.session_state['end_dt'])

            lpis_enabled = st.toggle('Enable LPIS', value=st.session_state['lpis_enabled'],
                                     help='Whether to use LPIS vector data and'
                                          ' perform homogenization of prediction. Note that'
                                          'LPIS data will be downloaded for selected year.',
                                     disabled=st.session_state['locked'])
            st.session_state['lpis_enabled'] = lpis_enabled

            cache_enabled = st.toggle('Enable cache', value=st.session_state['cache_enabled'],
                                      help='Whether to cache all obtained data.'
                                           'It includes all Sentinel-2 tiles and LPIS data.'
                                           'Otherwise all data are deleted.', disabled=True)#st.session_state['locked'])

            st.session_state['cache_enabled'] = cache_enabled

            make_prediction = st.button('Predict crops', help='Perform crop type prediction for selected patch within'
                                                              'selected temporal window. Note that this step involves'
                                                              'Sentinel-2 data downloading.',
                                        disabled=st.session_state['locked'])

            st.info('Demo can be very memory intensive. To save some system memory we use cache storage.\n'
                    'To explicitely manage/delete cached files use tab Cache in left sidebar', icon="ℹ️")

            if not st.session_state['authorized']:
                st.error(f'Unauthorized access to OpenSearch API. \n'
                         f'Please specify correct credentials for https://dhr1.cesnet.cz/')
            if st.session_state['predicted'] and not make_prediction:
                st.success('Prediction performed successfully. Please check Crop Map tab in left sidebar', icon="✅")
            if st.session_state['patch_error']:
                st.error(f'Please select patch on map. Patch should be within Czech Republic.')
            elif st.session_state['patch'] is None and make_prediction:
                st.session_state['patch_error'] = True
                st.error(f'Please select patch on map. Patch should be within Czech Republic.')
            elif st.session_state['patch'] != -1 and make_prediction:
                st.session_state['patch_error'] = False
                st.session_state['show_credentials'] = False
                st.session_state['run_pipeline'] = True
                st.session_state['locked'] = True
                st.session_state['show_crop_map'] = False
                # time.sleep(0.2)
                st.rerun()

        else:

            if st.session_state['run_pipeline']:
                # st.warning('Please do not change page layout otherwise it will stop whole processing pipeline. \n'
                #           'This is currently known issue. If you need to stop or reset processing use provided button.')
                st.warning('There is known issue with memory leaks when downloading S2 data. \n '
                           'Current workaround is to let process download data and change browser tab to '
                           'another one i.e. do something else in tab other than Crop2Seg and sometimes check '
                           'progress.')
                reset = st.button('Reset', help='Stop or reset processing pipeline.')

                if reset:
                    st.session_state['run_pipeline'] = False
                    st.session_state['locked'] = False
                    st.session_state['predicted'] = False
                    # time.sleep(0.8)
                    st.rerun()

                with st.status("Building time-series ...", expanded=True) as status:
                    st.session_state['authorized'] = True
                    st.session_state['show_credentials'] = False

                    # tilename, patch_id, patch_transform, patch_bounds = get_info(x=out[0], y=out[1])
                    tilename, patch_bounds = get_info(st.session_state['patch'])
                    affine_transform = [[10, 0], [0, -10], [patch_bounds[0], patch_bounds[-1]]]
                    st.write("Retrieving Sentinel-2 data...")
                    dates, temp_dir = get_ts(tilename, st.session_state['patch'], patch_bounds,
                                             year=st.session_state['year'],
                                             start_dt=opts[st.session_state['start_dt']],
                                             end_dt=opts[st.session_state['end_dt']],
                                             cache=st.session_state['cache_enabled'],
                                             password=st.session_state['password'],
                                             account=st.session_state['account'])

                    if dates != 401:
                        status.update(label=f"Sentinel-2 time series built successfully!", state="complete",
                                      expanded=False)
                        st.session_state['authorized'] = True
                        st.session_state['show_credentials'] = False
                    else:
                        status.update(label=f'Unauthorized access to OpenSearch API. \n'
                                            f'Please specify correct credentials for https://dhr1.cesnet.cz/',
                                      state="error",
                                      expanded=False)
                        st.session_state['authorized'] = False
                        st.session_state['show_credentials'] = True
                        st.session_state['run_pipeline'] = False
                        st.session_state['locked'] = False
                        # time.sleep(1)
                        st.rerun()

                with st.status("Predicting...", expanded=True) as status:

                    # TODO https://docs.streamlit.io/library/advanced-features/caching
                    #  https://docs.streamlit.io/library/api-reference/performance/st.cache_data

                    os.makedirs('src/webapp/cache/prediction', exist_ok=True)
                    os.makedirs('data/export/', exist_ok=True)
                    if torch.cuda.is_available():
                        proba = generate_prediction('cuda', st.session_state['year'], temp_dir, affine_transform)
                    else:
                        proba = generate_prediction('cpu', st.session_state['year'], temp_dir, affine_transform)

                    status.update(label=f'Prediction generated successfully!', state="complete",
                                  expanded=False)
                if st.session_state['lpis_enabled']:
                    with st.status(f"Preparing LPIS data for year {st.session_state['year']} ...",
                                   expanded=True) as status:
                        try:
                            get_LPIS('src/webapp/', st.session_state['year'])

                            st.write('Performing homogenization...')
                            gdf_homogenized = None
                            if os.path.isfile(f"src/webapp/cache/prediction/prediction_homogenized_{st.session_state['year']}.shp"):
                                gdf_homogenized = gpd.read_file(f"src/webapp/cache/prediction/prediction_homogenized_{st.session_state['year']}.shp")
                                if not gdf_homogenized[gdf_homogenized['name'] == os.path.split(temp_dir)[-1]].empty:
                                    st.write('Homogenization already generated... Skipping')
                                else:
                                    homogenized = homogenize(proba,
                                                             vector_data_path=f"src/webapp/cache/lpis/{st.session_state['year']}1231-CR-DPB-SHP.shp",
                                                             affine=affine_transform, epsg='epsg:32633', vector_epsg='epsg:5514',
                                                             type_='hard')
                                    homogenized.loc[:, 'name'] = os.path.split(temp_dir)[-1]

                                    st.write("Exporting layers...")

                                    if gdf_homogenized is not None:
                                        gdf_homogenized = pd.concat([gdf_homogenized, homogenized])
                                    else:
                                        gdf_homogenized = homogenized

                                    gdf_homogenized.to_file(
                                        f"src/webapp/cache/prediction/prediction_homogenized_{st.session_state['year']}.shp")
                            else:
                                homogenized = homogenize(proba,
                                                         vector_data_path=f"src/webapp/cache/lpis/{st.session_state['year']}1231-CR-DPB-SHP.shp",
                                                         affine=affine_transform, epsg='epsg:32633',
                                                         vector_epsg='epsg:5514',
                                                         type_='hard')
                                homogenized.loc[:, 'name'] = os.path.split(temp_dir)[-1]

                                st.write("Exporting layers...")

                                if gdf_homogenized is not None:
                                    gdf_homogenized = pd.concat([gdf_homogenized, homogenized])
                                else:
                                    gdf_homogenized = homogenized

                                gdf_homogenized.to_file(f"src/webapp/cache/prediction/prediction_homogenized_{st.session_state['year']}.shp")

                            status.update(label=f'LPIS data loaded successfully!', state="complete",
                                          expanded=False)
                        except Exception as e:
                            status.update(label=f'Error occured when loading LPIS layer... Skipping', state="error",
                                          expanded=False)

                st.session_state['predicted'] = True
                st.session_state['show_crop_map'] = True
                st.session_state['run_pipeline'] = False

                if st.session_state['predicted']:
                    st.success('Prediction performed successfully. Please check Crop Map tab in left sidebar', icon="✅")


elif selected == 'Crop map':
    cropmap(st.session_state['year'])

elif selected == 'Cache':
    cache_mgmt()
