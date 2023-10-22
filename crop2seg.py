from streamlit_option_menu import option_menu

import streamlit as st

from src.webapp.home import upload, home
from src.webapp.postprocess import cropmap
from src.webapp.get_data import get_info, get_ts
from src.webapp.postprocess import vectorize

st.set_page_config(page_title="Streamlit Geospatial", layout="wide")

# A dictionary of apps in the format of {"App title": "App icon"}
# More icons can be found here: https://icons.getbootstrap.com

apps = [
    {"func": home, "title": "Home", "icon": "house"},
    {"func": cropmap, "title": "Crop map", "icon": "map"},
    {"func": upload, "title": "Upload", "icon": "cloud-upload"},
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

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This web [app] is part of project for crop type prediction within Czech republic based on
        yearly time-series of Sentinel-2 images.
            [GitHub](https://github.com/Many98/Crop2Seg).

        This working exmaple was based on : <https://github.com/giswqs/streamlit-template>

        More menu icons: <https://icons.getbootstrap.com>
    """
    )

for app in apps:
    if app["title"] == selected:
        out = app["func"]()

        user_name = st.text_input("Enter a OpenSearch API username")
        password = st.text_input("Enter a OpenSearch API password", type="password")

        year = st.slider('Year of prediction', 2015, 2023, 2018,
                         help='Specify year for which crop map should be generated')

        make_prediction = st.button('Predict crops')

        if out is None and make_prediction:
            st.error(f'Please select point on map. Point should be within Czech Republic.')
        elif out is not None and make_prediction:

            with st.status("Downloading Sentinel-2 time-series data...", expanded=True) as status:
                st.write("Extracting patch...")
                tilename, patch_id, patch_transform, patch_bounds = get_info(x=out[0], y=out[1])
                st.write("Retrieving data...")
                # TODO user should choose year
                #ts, dates = get_ts(tilename, patch_bounds, year=year, cache=True, password=password, account=user_name)
                dates = 200
                if dates != 401:
                    status.update(label=f"Download completed!", state="complete",
                                  expanded=False)
                else:
                    status.update(label=f'Unauthorized access to OpenSearch API. \n'
                                    f'Please specify correct credentials for https://dhr1.cesnet.cz/', state="error",
                                  expanded=False)
            st.error(f'Unauthorized access to OpenSearch API. \n'
                     f'Please specify correct credentials for https://dhr1.cesnet.cz/')
            with st.status("Predicting...", expanded=True) as status:
                st.write("Loading neural net...")
                st.write("Generating raw prediction...")
                st.write("Postprocessing...")
                st.write("Performing vectorization of raster layer...")


                st.write("Exporting layers")

                status.update(label=f'Prediction generated successfully!', state="complete",
                              expanded=False)
            st.success('Please check Crop Map tab for your prediction', icon="✅")

        break


