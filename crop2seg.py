# ### small boiler plate to add src to sys path
import os
import sys
from pathlib import Path

file = Path(__file__).resolve()
root = str(file).split('Crop2Seg')[0]
sys.path.append(root)
os.chdir(os.path.join(root, 'Crop2Seg'))
# --------------------------------------

from streamlit_option_menu import option_menu

import streamlit as st

from src.webapp.home import home
from src.webapp.postprocess import cropmap
from src.webapp.get_data import get_info, get_ts, generate_grid, get_LPIS

st.set_page_config(page_title="Crop2Seg", layout="wide")

# A dictionary of apps in the format of {"App title": "App icon"}
# More icons can be found here: https://icons.getbootstrap.com

generate_grid('src/webapp')

apps = [
    {"func": home, "title": "Home", "icon": "house"},
    {"func": cropmap, "title": "Crop map", "icon": "map"},
]

titles = [app["title"] for app in apps]
titles_lower = [title.lower() for title in titles]
icons = [app["icon"] for app in apps]

params = st.experimental_get_query_params()

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

if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False

if 'year' not in st.session_state:
    st.session_state['year'] = 2019

if 'start_dt' not in st.session_state:
    st.session_state['start_dt'] = '9-2018'

if 'end_dt' not in st.session_state:
    st.session_state['end_dt'] = '10-2019'

if 'patch' not in st.session_state:
    st.session_state['patch'] = None

if 'last_click' not in st.session_state:
    st.session_state['last_click'] = None

if 'patch_error' not in st.session_state:
    st.session_state['patch_error'] = False

if 'process' not in st.session_state:
    st.session_state['process'] = False

if 'lpis_enabled' not in st.session_state:
    st.session_state['lpis_enabled'] = True

if 'cache_enabled' not in st.session_state:
    st.session_state['cache_enabled'] = True

if selected == 'Home':
    out = home()

    if not st.session_state['authorized']:
        user_name = st.text_input("Enter a OpenSearch API username")
        password = st.text_input("Enter a OpenSearch API password", type="password")

    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        year = st.slider('Year of prediction', 2015, 2023, st.session_state["year"],
                         help='Specify year for which crop map should be generated')
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
            help='Specify range within chosen year')
        st.session_state['start_dt'] = start_dt
        st.session_state['end_dt'] = end_dt

    st.write(f'You selected year {st.session_state["year"]} with temporal window between', st.session_state['start_dt'],
             'and', st.session_state['end_dt'])

    lpis_enabled = st.toggle('Enable LPIS', value=st.session_state['lpis_enabled'],
                             help='Whether to use LPIS vector data and'
                                  ' perform homogenization of prediction. Note that'
                                  'LPIS data will be downloaded for selected year.')
    st.session_state['lpis_enabled'] = lpis_enabled

    cache_enabled = st.toggle('Enable cache', value=st.session_state['cache_enabled'],
                              help='Whether to cache all obtained data.'
                                   'It includes all Sentinel-2 tiles and LPIS data.'
                                   'Otherwise all data are deleted.')
    st.session_state['cache_enabled'] = cache_enabled

    make_prediction = st.button('Predict crops', help='Perform crop type prediction for selected patch within'
                                                      'selected temporal window. Note that this step involves'
                                                      'Sentinel-2 data downloading.')

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
        with st.status("Downloading Sentinel-2 time-series data...", expanded=True) as status:
            st.write("Extracting patch...")
            # tilename, patch_id, patch_transform, patch_bounds = get_info(x=out[0], y=out[1])
            tilename, patch_bounds = get_info(st.session_state['patch'])
            st.write("Retrieving data...")
            '''
            ts, dates = get_ts(tilename, patch_bounds, year=year, start_dt=opts[st.session_state['start_dt']],
                               end_dt=opts[st.session_state['end_dt']],
                               cache=True, password=password, account=user_name)
            '''
            st.write(f"{tilename}")
            st.write(f"{patch_bounds}")
            st.write(f"{opts[st.session_state['start_dt']]}:{opts[st.session_state['end_dt']]}")
            dates = 200
            if dates != 401:
                status.update(label=f"Sentinel-2 data download completed!", state="complete",
                              expanded=False)
                st.session_state['authorized'] = True
            else:
                status.update(label=f'Unauthorized access to OpenSearch API. \n'
                                    f'Please specify correct credentials for https://dhr1.cesnet.cz/', state="error",
                              expanded=False)
                st.session_state['authorized'] = False
                st.rerun()

        with st.status("Predicting...", expanded=True) as status:
            st.write("Loading neural net...")
            st.write("Generating raw prediction...")
            st.write("Postprocessing...")
            st.write("Performing vectorization of raster layer...")

            st.session_state['predicted'] = True

            st.write("Exporting layers")

            status.update(label=f'Prediction generated successfully!', state="complete",
                          expanded=False)
        if st.session_state['lpis_enabled']:
            with st.status(f"Loading LPIS data for year {year} ...", expanded=True) as status:
                get_LPIS('src/webapp/', year)

                status.update(label=f'LPIS data loaded successfully!', state="complete",
                              expanded=False)

        if st.session_state['predicted']:
            st.success('Prediction performed successfully. Please check Crop Map tab in left sidebar', icon="✅")

elif selected == 'Crop map':
    cropmap(st.session_state['year'])
