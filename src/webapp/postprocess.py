# ############
# SCRIPT CONTAINING ALL RELEVANT FUNCTIONS FOR POSTPROCESSING PREDICTION
# ############

import rasterio
from rasterio.features import shapes
import geopandas as gpd
import leafmap.foliumap as leafmap

import streamlit as st


def vectorize(tif_path):

    with rasterio.open(tif_path) as f:
        img = f.read()
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for j, (s, v)
            in enumerate(shapes(img[0], mask=None, transform=f.transform)))

    gdf = gpd.GeoDataFrame.from_features(list(results))
    gdf = gdf.set_crs('epsg:32633')

    return gdf


def cropmap():

    st.title("Crop Map")

    filepath = "https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_cities.csv"
    m = leafmap.Map(tiles="stamentoner")
    m.add_heatmap(
        filepath,
        latitude="latitude",
        longitude="longitude",
        value="pop_max",
        name="Heat map",
        radius=20,
    )
    m.to_streamlit(height=700)


