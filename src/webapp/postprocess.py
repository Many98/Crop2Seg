# ############
# SCRIPT CONTAINING ALL RELEVANT FUNCTIONS FOR POSTPROCESSING PREDICTION
# ############
import os
import time
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import leafmap.foliumap as leafmap
from shapely.geometry import box as shapely_box

import streamlit as st


def crop_cmap():
    """
    Auxiliary function to return dictionary with color map used for visualization
    of classes in S2TSCZCrop dataset
    """

    return {0: '#000000',  # Background
            1: '#a0db8e',  # Permanent grassland
            2: '#cc5500',  # Annual fruit and vegetable
            3: '#e9de89',  # Summer cereals
            4: '#f4ecb1',  # Winter cereals
            5: '#dec928',  # Rapeseed
            6: '#f0a274',  # Maize
            7: '#556b2f',  # Annual forage crops
            8: '#94861b',  # Sugar beat
            9: '#767ee1',  # Flax and Hemp
            10: '#7d0015',  # Permanent fruit
            11: '#9299a9',  # Hopyards
            12: '#dea7b0',  # Vineyards
            13: '#ff0093',  # Other crops
            14: '#c0d8ed',  # Not classified (removed from training and evaluation)
            15: '#ffffff'  # Border of semantic classes
            }


labels = ['Background 0', 'Permanent grassland 1', 'Annual fruit and vegetable 2', 'Summer cereals 3',
          'Winter cereals 4', 'Rapeseed 5', 'Maize 6', 'Annual forage crops 7', 'Sugar beat 8', 'Flax and Hemp 9',
          'Permanent fruit 10', 'Hopyards 11', 'Vineyards 12', 'Other crops 13', 'Not classified 14']


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


def cropmap(year):
    st.header('Crop Map', divider='rainbow')
    st.caption('Predictions will be exported in this map')

    # here how to export proj file because in lpis is not proj file and it is in krovak crs
    # with fiona.open(
    # f"src/webapp/cache/lpis/{year}1231-CR-DPB-SHP.shp"
    #    ) as src:
    # driver = src.driver
    # schema = src.schema
    # feat = src[1]

    # with fiona.open(f"src/webapp/cache/lpis/example.shp", "w", driver=driver, crs=gdf2.crs,
    #                schema=schema) as dst:
    #    print(len(dst))
    #    dst.write(feat)

    m = leafmap.Map(center=(50, 14), draw_export=True, zoom=8)
    m.add_basemap("HYBRID")
    m.add_basemap("ROADMAP")

    if st.session_state['show_crop_map']:
        with st.status("Loading layers ...", expanded=True) as status:
            try:
                gdf_pred = gpd.read_file(f'src/webapp/cache/prediction/prediction_{year}.shp')
                # remove unnecessary polygons
                gdf_pred = gdf_pred[gdf_pred.raster_val != 0]
                gdf_pred = gdf_pred[gdf_pred.area > 5000]

                cmap = crop_cmap()
                gdf_pred['crop'] = gdf_pred['raster_val'].apply(lambda x: labels[int(x)])

                if st.session_state['lpis_enabled']:
                    try:
                        gdf_homogenized = gpd.read_file(f'src/webapp/cache/prediction/prediction_homogenized_{year}.shp')

                        # workaround to get proper crs and bbox
                        gdf2 = gdf_pred.copy()
                        gdf2 = gdf2.to_crs('epsg:5514')  # krovak
                        b = gdf2.total_bounds
                        bbox = shapely_box(b[0], b[1], b[2], b[3])
                        features_19 = gpd.read_file(f'src/webapp/cache/lpis/{year}1231-CR-DPB-SHP.shp',
                                                    bbox=bbox)
                        features_19 = features_19.to_crs(32633)

                        vymera = [i for i in features_19.columns if 'VYMERA' in i]

                        features_19 = features_19[['ID_DPB', 'CTVEREC', 'PLATNYOD', 'KULTURANAZ', 'KULTURAKOD',
                                                   'KULTURA', 'KULTURAOD', 'geometry'] + vymera]

                        m.add_gdf(features_19, layer_name=f'lpis_{year}',
                                  style={
                                      "stroke": True,
                                      "color": "red",
                                      "weight": .1,
                                      "opacity": .5,
                                      "fill": True,
                                      "fillOpacity": 0.5,
                                  }
                                  )
                        gdf_homogenized['crop'] = gdf_homogenized['raster_val'].apply(lambda x: labels[int(x)])
                        m.add_data(gdf_homogenized, layer_name=f'homogenized_{year}', labels=labels,
                                   colors=list(cmap.values())[:-1],  # without boundary
                                   column="raster_val",
                                   legend_position='bottomright',
                                   scheme='UserDefined',
                                   classification_kwds={'bins': [i for i in range(0, 15)]},
                                   # style_kwds =dict(color="gray", LineWidth=10, weight=0.99)
                                   # k=len(labels)+1,
                                   style={
                                       "stroke": True,
                                       "color": "gray",
                                       "weight": .1,
                                       "opacity": .1,
                                       "fill": True,
                                       "fillOpacity": 0.1,
                                   }, fields=['crop']
                                   )
                    except Exception as e:
                        status.update(label=f"Error when loading LPIS data... Skipping", state="error",
                                      expanded=False)
                        st.error(f'Error : {e}')

                m.add_data(gdf_pred, layer_name=f'prediction_{year}', labels=labels,
                           colors=list(cmap.values())[:-1],  # without boundary
                           column="raster_val",
                           legend_position='bottomright',
                           scheme='UserDefined',
                           classification_kwds={'bins': [i for i in range(0, 15)]},
                           # style_kwds =dict(color="gray", LineWidth=10, weight=0.99)
                           # k=len(labels)+1,
                           style={
                               "stroke": True,
                               "color": "gray",
                               "weight": .1,
                               "opacity": .1,
                               "fill": True,
                               "fillOpacity": 0.1,
                           },
                           fields=['crop'])
                m.zoom_to_gdf(gdf_pred)
            except Exception as e:
                status.update(label=f"Error when loading prediction layer... Skipping", state="error",
                              expanded=False)
                st.error(f'Error : {e}')

            status.update(label=f"Layers loaded successfully!", state="complete",
                          expanded=False)
    # m.add_legend(title='Legend', labels=labels, layer_name='prediction', colors=cmap.values(), legend_dict=cmap)
    '''
    m.add_heatmap(
        filepath,
        latitude="latitude",
        longitude="longitude",
        value="pop_max",
        name="Heat map",
        radius=20,
    )
    '''
    m.to_streamlit(height=600)
    st.session_state['predicted'] = False
    if st.session_state['run_pipeline'] and st.session_state['locked']:
        # time.sleep(0.2)
        st.session_state['locked'] = False
        st.rerun()

