import numpy as np
from scipy.ndimage.measurements import label
import torch

import geopandas as gpd
import rioxarray
import rasterio
from rasterio.features import shapes
from rasterio.coords import BoundingBox
from shapely.geometry import box as shapely_box

import os
from typing import Tuple, Union

from src.helpers.sentinel2raster import export_to_tif


@export_to_tif
def prediction2raster(prediction: np.ndarray, crs: str, affine: list, export: bool = False,
                      export_dir: str = '', export_name: str = '') -> (
        Tuple)[Union[rasterio.io.DatasetReader, rasterio.io.MemoryFile], str]:
    """
    exports prediction (top 1 labels) and probabilities to raster

    prediction: array of shape [NUM CLASSES, HEIGHT, WIDTH] with class probabilities
    crs: crs of output raster
    affine: affine transform of raster
    export: whether to export to tif
    """

    assert prediction.ndim == 3, 'Prediction array is expected to be 3 dimensional vector of soft predictions'
    assert prediction.max() <= 1., 'Prediction array is expected to be 3 dimensional vector of soft predictions'

    profile = {'driver': 'GTiff', 'dtype': rasterio.float32, 'nodata': 0.0, 'width': prediction.shape[1],
               'height': prediction.shape[2], 'count': prediction.shape[0] + 1,
               'crs': rasterio.crs.CRS.from_epsg(crs),
               'transform': rasterio.Affine(affine[0][0], affine[1][0], affine[2][0],
                                            affine[0][1], affine[1][1], affine[2][1]),
               'blockxsize': 128,
               'blockysize': 128, 'tiled': True, 'compress': 'lzw'}

    hard_pred = np.argmax(prediction, axis=0).astype('float32')

    prediction = np.vstack([hard_pred[None, ...], prediction])

    # TODO MemoryFile does not empties /vsim/... -> stop using it ... use just plain np.array accessed via read method
    #  use NamedTemporaryFile instead https://rasterio.readthedocs.io/en/stable/topics/memory-files.html
    memfile = rasterio.io.MemoryFile(filename=f'{export_name}.tif')
    with memfile.open(**profile) as rdata:
        rdata.write(prediction.astype('float32'))  # write the data

    if export:
        os.makedirs(export_dir, exist_ok=True)

    return memfile.open(), os.path.join(export_dir, export_name + '.tif')


def prediction2polygon_layer(prediction: np.ndarray, affine: list, epsg: str = 'epsg:32633') -> gpd.GeoDataFrame:
    """
    exports prediction (top 1 labels) to polygon layer (GeoDataFrame)

    prediction: array of shape [NUM CLASSES, HEIGHT, WIDTH] with class probabilities
                or [HEIGHT, WIDTH] with proper top 1 labels
    epsg: epsg of polygon layer
    affine: affine transform of raster
    """
    if prediction.ndim == 2:
        hard_pred = prediction.astype(np.uint8)
    elif prediction.ndim == 3:
        hard_pred = np.argmax(prediction, axis=0).astype(np.uint8)
    else:
        raise Exception(f'Prediction array must 2 or 3 dimensional')

    transform = rasterio.Affine(affine[0][0], affine[1][0], affine[2][0],
                                affine[0][1], affine[1][1], affine[2][1])

    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for j, (s, v)
        in enumerate(shapes(hard_pred, mask=None, transform=transform)))

    gdf = gpd.GeoDataFrame.from_features(list(results), crs=epsg)

    return gdf


def raster2polygon_layer(raster_path: Union[str, rasterio.io.MemoryFile, rasterio.io.DatasetReader]):
    """
    converts (predicted) raster to polygon vector layer (geodataframe)
    only first hard labels are used
    """
    if isinstance(raster_path, rasterio.io.MemoryFile) or isinstance(raster_path, rasterio.io.DatasetReader):
        ff = raster_path.files[0]
        raster_path.close()
    elif isinstance(raster_path, str):
        ff = raster_path
    else:
        raise Exception('Unsupported type of input raster_path')
    r = rasterio.open(ff)

    affine = r.transform
    epsg = r.crs.data['init']
    r.close()

    with rasterio.open(ff) as f:
        img = f.read(0)
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for j, (s, v)
            in enumerate(shapes(img, mask=None, transform=affine)))

    gdf = gpd.GeoDataFrame.from_features(list(results))
    gdf = gdf.set_crs(epsg)

    return gdf


def raster2point_layer(raster_path: Union[str, rasterio.io.MemoryFile]):
    """
    converts (predicted) raster to point vector layer (geodataframe)
    """
    if isinstance(raster_path, rasterio.io.MemoryFile) or isinstance(raster_path, rasterio.io.DatasetReader):
        rds = rioxarray.open_rasterio(raster_path.files[0])
        raster_path.close()
    elif isinstance(raster_path, str):
        rds = rioxarray.open_rasterio(raster_path)
    else:
        raise Exception('Unsupported type of input raster_path')

    rds.name = "data"
    df = rds.squeeze().to_dataframe().reset_index()
    geometry = gpd.points_from_xy(df.x, df.y)
    return gpd.GeoDataFrame(df, crs=rds.rio.crs, geometry=geometry)


def homogenize(prediction: Union[str, rasterio.io.MemoryFile, rasterio.io.DatasetReader], vector_data_path: str,
               affine: list, epsg: str = 'epsg:32633'):
    """
    homogenize (polygonize) prediction on external data vector data (LPIS)
    """
    if isinstance(prediction, rasterio.io.MemoryFile) or isinstance(prediction, rasterio.io.DatasetReader):
        bbox = prediction.bounds
        affine = prediction.transform
        shape = prediction.shape[-2:]
        gdf = raster2polygon_layer(prediction)

    elif isinstance(prediction, str):
        r = rasterio.open(prediction)

        bbox = r.bounds
        affine = r.transform
        shape = r.shape[-2:]
        r.close()

        gdf = raster2polygon_layer(prediction)
    elif isinstance(prediction, np.ndarray):
        gdf = prediction2polygon_layer(prediction, affine, epsg)
        bb = gdf.total_bounds
        bbox = BoundingBox(*list(bb))
    else:
        raise Exception('Unsupported type of input prediction')

    filtered_features = gpd.read_file(vector_data_path, bbox=shapely_box(bbox.left, bbox.bottom, bbox.right, bbox.top))

    # features.sindex
    # indices = features.sindex.query(shapely_box(bbox.left, bbox.bottom, bbox.right, bbox.top),
    #                                predicate='intersects')
    # filtered_features = features.iloc[indices]

    filtered_features = filtered_features.reset_index()

    merged = gpd.overlay(filtered_features, gdf, how='intersection')
    merged['area'] = merged['geometry'].area

    cc = merged[merged['raster_val'] > 0.0].groupby('index', as_index=False)[['area']].max()

    gg = cc[['index', 'area']].merge(filtered_features[['index', 'geometry']], on='index', how='inner')
    uu = gg[['area', 'index', 'geometry']].merge(merged[['area', 'raster_val']], on='area', how='inner')
    tt = gpd.GeoDataFrame(uu)

    # TODO enrich it with other per polygons stats like probability etc
    '''
    shapes_ = tt[['geometry', 'raster_val']].values.tolist()

    out = rasterio.features.rasterize(shapes_,
                                      out_shape=shape,
                                      fill=0,  # fill value for background
                                      out=None,
                                      transform=affine,
                                      all_touched=False,
                                      # merge_alg=MergeAlg.replace,  # ... used is default
                                      default_value=1,
                                      dtype=rasterio.uint8
                                      )
    '''
    return tt[['geometry', 'raster_val']]


def homogenize_boundaries(prediction: np.ndarray, affine: list, epsg: str = 'epsg:32633', boundary_code: int = 15):
    """
    auxiliary function to polygonize predictions based on boundary predictions
    Note that currently function is not very robust and expects that boundary is encoded as `15`
    """
    element = np.ones((3, 3))
    element[0, 0] = 0
    element[0, 2] = 0
    element[2, 2] = 0
    element[2, 0] = 0

    prediction = torch.from_numpy(prediction)

    top_2 = prediction.topk(k=2, dim=0)

    pred_t1 = top_2[1][0][0].numpy()
    proba_t1 = top_2[0][0][0].numpy()
    pred_t2 = top_2[1][0][1].numpy()
    proba_t2 = top_2[0][0][1].numpy()

    # super = np.where((pred_1_b == boundary_code) | ((pred_2_b == boundary_code) & (proba_2_b > 0.3)), 0, 1)
    # TODO we should probably perform opening https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    #  to remove pixels blocking
    super = np.where((pred_t1 == boundary_code) | ((pred_t2 == boundary_code) & (proba_t2 > 0.3)) | (pred_t1 == 0), 0, 1)
    altered = np.where((pred_t1 == boundary_code), pred_t2, pred_t1)

    labeled, ncomp = label(super, element)

    sizes = [np.where(labeled == i)[0].shape[0] for i in range(ncomp)]
    to_small = [i for i in range(ncomp) if sizes[i] < 13]

    # remove too small components
    labeled2 = np.where(np.isin(labeled, to_small), 0, labeled)

    extracted_shapes = prediction2polygon_layer(labeled2, affine, epsg=epsg)

    extracted_shapes = extracted_shapes[extracted_shapes.raster_val != 0]

    #extracted_shapes.plot()
    '''
    plt.xlim(extracted_shapes.bounds.min().loc['minx'], extracted_shapes.bounds.max().loc['maxx'])
    plt.ylim(extracted_shapes.bounds.min().loc['miny'], extracted_shapes.bounds.max().loc['maxy'])
    plt.show()
    '''

    gdf = prediction2polygon_layer(pred_t1, affine, epsg=epsg)

    extracted_shapes = extracted_shapes.reset_index()

    merged = gpd.overlay(extracted_shapes, gdf, how='intersection', keep_geom_type=False)
    merged['area'] = merged['geometry'].area

    cc = merged[merged['raster_val'] > 0.0].groupby('index', as_index=False)[['area']].max()

    gg = cc[['index', 'area']].merge(extracted_shapes[['index', 'geometry']], on='index', how='inner')
    uu = gg[['area', 'index', 'geometry']].merge(merged[['area', 'raster_val', 'index']], on=['area', 'index'],
                                                 how='inner')
    tt = gpd.GeoDataFrame(uu)

    '''
    # shapes_ = merged2[['geometry', 'value']].values.tolist()
    shapes_ = tt[['geometry', 'raster_val']].values.tolist()

    out = rasterio.features.rasterize(shapes_,
                                      out_shape=pred_1_b.shape,
                                      fill=0,  # fill value for background
                                      out=None,
                                      transform=transform,
                                      all_touched=False,
                                      # merge_alg=MergeAlg.replace,  # ... used is default
                                      default_value=1,
                                      dtype=rasterio.uint8
                                      )
    '''

    return tt[['geometry', 'raster_val']]
