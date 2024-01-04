
import leafmap.foliumap as leafmap
from folium import LatLngPopup
from shapely.geometry import Point
import geopandas as gpd
import streamlit as st

if 'patch' not in st.session_state:
    st.session_state['patch'] = None


def home():
    st.title("Crop2Seg demo")

    st.markdown(
        """
    Predict crop types for Czech republic using Sentinel-2 time-series

    """
    )

    grid = gpd.read_file('src/webapp/data/s2_grid/grid.shp')
    grid = grid.to_crs(4326)
    grid = grid.reset_index()
    try:
        if st.session_state['patch'] != -1:
            chosen = grid[grid['index'] == st.session_state['patch']]
            opa = 0.8
        else:
            chosen = grid
            opa = 0.0
    except:
        chosen = grid
        opa = 0.0

    m = leafmap.Map(locate_control=True, location=[49.78, 15.37], zoom=7)
    m.add_basemap("HYBRID")
    m.add_basemap("ROADMAP")
    m.add_child(LatLngPopup())

    m.add_gdf(grid, layer_name=f'grid', zoom_to_layer=False, info_mode="on_hover",
              style={
                  # "stroke": True,
                  "color": "red",
                  "opacity": .4,
                  "fill": True,
                  "fillOpacity": 0.2,
              }
              )

    m.add_gdf(chosen, layer_name=f'patch', zoom_to_layer=False, info_mode=None,
              style={
                  # "stroke": True,
                  "color": "green",
                  "opacity": .0,
                  "fill": True,
                  "fillOpacity": opa,
              }
              )

    map = m.to_streamlit(height=380, bidirectional=True)

    try:
        last_click = m.st_last_click(map)
    except:
        last_click = st.session_state['last_click']

    if last_click is not None and last_click != st.session_state['last_click']:
        st.session_state['last_click'] = last_click
        x, y = last_click[0], last_click[1]
        point = gpd.GeoDataFrame(geometry=[Point(y, x)], crs='EPSG:4326')
        joined = grid.sjoin(point, how="inner")
        try:
            i = int(joined['index'].values[0])

            if i != st.session_state['patch']:
                st.session_state['patch_error'] = False
                st.session_state['patch'] = i

        except:
            st.session_state['patch'] = None
        st.rerun()
