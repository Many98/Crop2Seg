import numpy as np
import pandas as pd
from scipy.ndimage.measurements import label
import torch

import geopandas as gpd
import rioxarray
import rasterio
from rasterio.features import shapes
from rasterio.coords import BoundingBox
from shapely.geometry import box as shapely_box
from shapely import Point

import os
from typing import Tuple, Union

from src.helpers.sentinel2raster import export_to_tif
from src.datasets.s2_ts_cz_crop import S2TSCZCropDataset


@export_to_tif
def prediction2raster(prediction: np.ndarray, epsg: int, affine: list, export: bool = False,
                      export_dir: str = '', export_name: str = '') -> (
        Tuple)[Union[rasterio.io.DatasetReader, rasterio.io.MemoryFile], str]:
    """
    Exports prediction (top 1 labels) and probabilities to raster

    Parameters
    ----------
    prediction: np.ndarray
         array of shape [NUM CLASSES, HEIGHT, WIDTH] with class probabilities
                or [HEIGHT, WIDTH] with proper top 1 labels
    epsg: int
        epsg code of raster
    affine: list
        affine transform of raster
    export: bool
        Whether to export raster to tif
    export_name: str
        optional name of exported raster
    export_dir: str
        optional directory of exported raster
    Returns
    -------
    gpd.GeoDataFrame
    """

    assert prediction.max() <= 1., 'Prediction array is expected to be 3 dimensional vector of soft predictions'

    if prediction.ndim == 2:
        prediction = prediction.astype(np.uint8)
        count = 1
        h = prediction.shape[1]
        w = prediction.shape[0]
        dtype = rasterio.uint8
    elif prediction.ndim == 3:
        hard_pred = np.argmax(prediction, axis=0).astype('float32')
        prediction = np.vstack([hard_pred[None, ...], prediction])
        count = prediction.shape[0]
        h = prediction.shape[2]
        w = prediction.shape[1]
        dtype = rasterio.float32
    else:
        raise Exception(f'Prediction array must 2 or 3 dimensional but is of dim {prediction.ndim}')

    profile = {'driver': 'GTiff', 'dtype': dtype, 'nodata': 0.0, 'width': w,
               'height': h, 'count': count,
               'crs': rasterio.crs.CRS.from_epsg(epsg),
               'transform': rasterio.Affine(affine[0][0], affine[1][0], affine[2][0],
                                            affine[0][1], affine[1][1], affine[2][1]),
               'blockxsize': 128,
               'blockysize': 128, 'tiled': True, 'compress': 'lzw'}

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
    Exports prediction (top 1 labels) to polygon layer (GeoDataFrame)

    Parameters
    ----------
    prediction: np.ndarray
        array of shape [NUM CLASSES, HEIGHT, WIDTH] with class probabilities
                or [HEIGHT, WIDTH] with proper top 1 labels
    epsg: str
        epsg code of polygon layer
    affine: list
        affine transform of raster
    Returns
    -------
    gpd.GeoDataFrame
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


def prediction2point_layer(prediction: np.ndarray, affine: list, epsg: str = 'epsg:32633') -> gpd.GeoDataFrame:
    """
    Exports prediction (top 1 labels) to point layer (GeoDataFrame)

    Parameters
    ----------
    prediction: np.ndarray
        array of shape [NUM CLASSES, HEIGHT, WIDTH] with class probabilities
                or [HEIGHT, WIDTH] with proper top 1 labels
    epsg: str
        epsg code of polygon layer
    affine: list
        affine transform of raster
    Returns
    -------
    gpd.GeoDataFrame
    """
    if prediction.ndim == 2:
        prediction = prediction.astype(np.uint8)
    elif prediction.ndim == 3:
        hard_pred = np.argmax(prediction, axis=0).astype('float32')
        prediction = np.vstack([hard_pred[None, ...], prediction])
    else:
        raise Exception(f'Prediction array must 2 or 3 dimensional but {prediction.ndim}-dimensional array provided')

    cols, rows = np.meshgrid(np.arange(prediction.shape[-2]), np.arange(prediction.shape[-1]))
    transform = rasterio.Affine(affine[0][0], affine[1][0], affine[2][0],
                                affine[0][1], affine[1][1], affine[2][1])

    xs, ys = rasterio.transform.xy(transform, rows, cols)

    xcoords = np.array(xs)
    ycoords = np.array(ys)

    if prediction.ndim == 3:
        results = (
            {'properties': {str(k - 1) if k > 0 else 'raster_val': v for k, v in enumerate(prediction[:, r, c])},
             'geometry': Point(xcoords[0, c], ycoords[r, 0])}
            for c in range(prediction.shape[-1]) for r in range(prediction.shape[-2])
        )
    else:
        results = (
            {'properties': {'raster_val': prediction[r, c]},
             'geometry': Point(xcoords[0, c], ycoords[r, 0])}
            for c in range(prediction.shape[-1]) for r in range(prediction.shape[-2])
        )

    gdf = gpd.GeoDataFrame.from_features(list(results), crs=epsg)

    return gdf


def raster2polygon_layer(raster: Union[str, rasterio.io.MemoryFile, rasterio.io.DatasetReader]) -> gpd.GeoDataFrame:
    """
    Converts (predicted) raster to polygon vector layer (geodataframe)
    only hard labels are used

    Parameters
    ----------
    raster: str or rasterio.io.MemoryFile or rasterio.io.DatasetReader
    Returns
    -------
    gpd.GeoDataFrame
    """
    if isinstance(raster, rasterio.io.MemoryFile) or isinstance(raster, rasterio.io.DatasetReader):
        ff = raster.files[0]
        raster.close()
    elif isinstance(raster, str):
        ff = raster
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


def raster2point_layer(raster: Union[str, rasterio.io.MemoryFile, rasterio.io.DatasetReader]) -> gpd.GeoDataFrame:
    """
    Converts (predicted) raster to point vector layer (geodataframe)

    Parameters
    ----------
    raster: str or rasterio.io.MemoryFile or rasterio.io.DatasetReader
    Returns
    -------
    gpd.GeoDataFrame
    """
    if isinstance(raster, rasterio.io.MemoryFile) or isinstance(raster, rasterio.io.DatasetReader):
        rds = rioxarray.open_rasterio(raster.files[0])
        raster.close()
    elif isinstance(raster, str):
        rds = rioxarray.open_rasterio(raster)
    else:
        raise Exception('Unsupported type of input raster_path')

    rds.name = "data"
    df = rds.squeeze().to_dataframe().reset_index()
    geometry = gpd.points_from_xy(df.x, df.y)
    return gpd.GeoDataFrame(df, crs=rds.rio.crs, geometry=geometry)


def soften(polygons: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Auxiliary function to get polygonization based on soft prediction

    Parameters
    ----------
    polygons: gpd.GeoDataFrame
        Stores polygon layer which will be enriched with soft prediction
    points: gpd.GeoDataFrame
        Stores point layer with soft prediction for every point
    Returns
    -------
    gpd.GeoDataFrame or np.ndarray
    """
    polygons = polygons[polygons.columns.difference(['index'])]
    merged = gpd.sjoin(polygons, points, how='left', predicate='covers')
    merged = merged.reset_index()
    merged = merged.rename(columns={'index': 'index_'})

    cols = []
    for c in merged.columns:
        try:
            _ = int(c)
            cols.append(c)
        except:
            continue

    cc = merged.groupby('index_', as_index=False)[cols].mean()
    cc = cc[cc.columns.difference(['index_'])]
    cc = cc[[str(i) for i in range(cc.shape[1])]]
    vals = cc.values
    top_2 = torch.from_numpy(vals).topk(k=2, dim=1)
    top1 = top_2[1][:, 0].numpy()
    top2 = top_2[1][:, 1].numpy()
    z = np.where(top1 == 0)
    top1[z] = np.where(top_2[0][z][:, 0] > 0.7, 0, top2[z])
    cc['soft_label'] = top1
    cc['soft_top2_label'] = top2

    cc = cc.reset_index()
    polygons = polygons.reset_index()
    out = cc.merge(polygons, on='index', how='left')

    return gpd.GeoDataFrame(out)


def polygonize(prediction: Union[str, rasterio.io.MemoryFile, rasterio.io.DatasetReader, np.ndarray],
               affine: list, epsg: str = 'epsg:32633', array_out: bool = False, type_: str = 'hard') -> Union[
    gpd.GeoDataFrame, np.ndarray]:
    """
    Polygonize prediction (based on top-1 prediction)
    Additionally enrich polygons with soft (predicted) distribution, soft labels, soft top 2 labels,
    and finally some score which will evaluate overall confidence of polygon

    Parameters
    ----------
    prediction: str or rasterio.io.MemoryFile or rasterio.io.DatasetReader or np.ndarray
                array of shape [NUM CLASSES, HEIGHT, WIDTH] with class probabilities
    epsg: str
        epsg code of polygon layer
    affine: list
        affine transform of raster
    array_out: bool
        Whether to export homogenized prediction as numpy array
    type_ : str
        Type of processing / labels. Can be `hard` - final label is hard label based on hard labels of prediction
                                            `soft` - fanl label is soft enriched with distribution
    Returns
    -------
    gpd.GeoDataFrame or np.ndarray
    """
    if isinstance(prediction, rasterio.io.MemoryFile) or isinstance(prediction, rasterio.io.DatasetReader):
        bbox = prediction.bounds
        transform = prediction.transform
        shape = prediction.shape[-2:]
        polygons = raster2polygon_layer(prediction)
        points = raster2point_layer(prediction) if type_ == 'soft' else None

    elif isinstance(prediction, str):
        r = rasterio.open(prediction)

        bbox = r.bounds
        transform = r.transform
        shape = r.shape[-2:]
        r.close()

        polygons = raster2polygon_layer(prediction)
        points = raster2point_layer(prediction) if type_ == 'soft' else None

    elif isinstance(prediction, np.ndarray):
        polygons = prediction2polygon_layer(prediction, affine, epsg)
        points = prediction2point_layer(prediction, affine, epsg) if type_ == 'soft' else None
        bb = polygons.total_bounds
        bbox = BoundingBox(*list(bb))
        shape = prediction.shape[-2:]
        transform = rasterio.Affine(affine[0][0], affine[1][0], affine[2][0],
                                    affine[0][1], affine[1][1], affine[2][1])
    else:
        raise Exception('Unsupported type of input prediction')

    if type_ == 'hard':
        polygons['raster_val'] = polygons['raster_val'].astype(np.uint8)
        if array_out:
            shapes_ = polygons[['geometry', 'raster_val']].values.tolist()
            out = rasterio.features.rasterize(shapes_,
                                              out_shape=shape,
                                              fill=0,  # fill value for background
                                              out=None,
                                              transform=transform,
                                              all_touched=False,
                                              # merge_alg=MergeAlg.replace,  # ... used is default
                                              default_value=1,
                                              dtype=rasterio.uint8
                                              )
            return out

        return polygons
    else:
        out = soften(polygons, points)

        if not array_out:
            return out

        out['soft_top2_label'] = out['soft_top2_label'].astype(np.uint8)

        shapes_ = out[['geometry', 'soft_top2_label']].values.tolist()
        out = rasterio.features.rasterize(shapes_,
                                          out_shape=shape,
                                          fill=0,  # fill value for background
                                          out=None,
                                          transform=transform,
                                          all_touched=False,
                                          # merge_alg=MergeAlg.replace,  # ... used is default
                                          default_value=1,
                                          dtype=rasterio.uint8
                                          )
        return out


def homogenize(prediction: Union[str, rasterio.io.MemoryFile, rasterio.io.DatasetReader, np.ndarray],
               vector_data_path: str, affine: list, epsg: str = 'epsg:32633', vector_epsg: str = 'epsg: 32633',
               array_out: bool = False, type_: str = 'hard') -> Union[gpd.GeoDataFrame, np.ndarray]:
    """
    Homogenize prediction using external vector data (LPIS)

    Parameters
    ----------
    prediction: str or rasterio.io.MemoryFile or rasterio.io.DatasetReader or np.ndarray
                array of shape [NUM CLASSES, HEIGHT, WIDTH] with class probabilities
                or [HEIGHT, WIDTH] with proper top 1 labels
    vector_data_path: str
        Absolute path to vector data (shapefile) used for homogenization
    epsg: str
        epsg code of polygon layer
    affine: list
        affine transform of raster
    array_out: bool
        Whether to export homogenized prediction as numpy array
    type_ : str
        Type of processing / labels. Can be `hard` - final label is hard label based on hard labels of prediction
                                            `soft` - return hard and soft label
    Returns
    -------
    gpd.GeoDataFrame or np.ndarray
    """
    if isinstance(prediction, rasterio.io.MemoryFile) or isinstance(prediction, rasterio.io.DatasetReader):
        bbox = prediction.bounds
        transform = prediction.transform
        shape = prediction.shape[-2:]
        gdf = raster2polygon_layer(prediction)
        points = raster2point_layer(prediction) if type_ == 'soft' else None

    elif isinstance(prediction, str):
        r = rasterio.open(prediction)

        bbox = r.bounds
        transform = r.transform
        shape = r.shape[-2:]
        r.close()

        gdf = raster2polygon_layer(prediction)
        points = raster2point_layer(prediction) if type_ == 'soft' else None
    elif isinstance(prediction, np.ndarray):
        gdf = prediction2polygon_layer(prediction, affine, epsg)
        points = prediction2point_layer(prediction, affine, epsg) if type_ == 'soft' else None
        gdf_ = gdf.to_crs(vector_epsg)
        bb = gdf_.total_bounds
        bbox = BoundingBox(*list(bb))
        shape = prediction.shape[-2:]
        transform = rasterio.Affine(affine[0][0], affine[1][0], affine[2][0],
                                    affine[0][1], affine[1][1], affine[2][1])
    else:
        raise Exception('Unsupported type of input prediction')

    filtered_features = gpd.read_file(vector_data_path, bbox=shapely_box(bbox.left, bbox.bottom, bbox.right, bbox.top))
    filtered_features = filtered_features.to_crs(epsg)

    # features.sindex
    # indices = features.sindex.query(shapely_box(bbox.left, bbox.bottom, bbox.right, bbox.top),
    #                                predicate='intersects')
    # filtered_features = features.iloc[indices]

    filtered_features = filtered_features.reset_index()
    filtered_features['area_polygon'] = filtered_features['geometry'].area
    #import hashlib
    #filtered_features.loc[:, 'hash'] = filtered_features['geometry'].apply(
    #    lambda x: hashlib.sha256(bytes(str(x), 'utf-8')).hexdigest())

    merged = gpd.overlay(filtered_features, gdf, how='intersection')
    merged['area'] = merged['geometry'].area

    kk = merged.groupby(['index', 'raster_val'], as_index=False)[['area', 'area_polygon']].agg({'area': 'sum',
                                                                                                'area_polygon': ['sum',
                                                                                                                 'count']})
    # cc = kk.groupby('index', as_index=False)[['area']].max()
    cc = kk[(kk.raster_val > 0) | ((kk.raster_val == 0) & (
                (kk.area['sum'] / (kk.area_polygon['sum'] / kk.area_polygon['count'])) > 0.75))].groupby('index',
                                                                                                         as_index=False)[
        [('area', 'sum')]].max()

    cc.columns = cc.columns.droplevel(1)
    #gg = cc[['index', 'area']].merge(filtered_features[['index', 'geometry', 'Legenda', 'hash']], on='index',
    #                                 how='inner')
    gg = cc[['index', 'area']].merge(filtered_features[['index', 'geometry']], on='index',
                                     how='inner')

    kk.columns = kk.columns.droplevel(1)
    #uu = gg[['area', 'index', 'geometry', 'hash', 'Legenda']].merge(kk[['area', 'raster_val', 'index']],
    #                                                                on=['area', 'index'], how='inner')
    uu = gg[['area', 'index', 'geometry']].merge(kk[['area', 'raster_val', 'index']],
                                                 on=['area', 'index'], how='inner')

    tt = gpd.GeoDataFrame(uu)
    #return tt['hash'].values, tt['area'].values, tt['raster_val'].values, tt['Legenda'].values
    if type_ == 'hard':
        if array_out:
            shapes_ = tt[['geometry', 'raster_val']].values.tolist()

            out = rasterio.features.rasterize(shapes_,
                                              out_shape=shape,
                                              fill=0,  # fill value for background
                                              out=None,
                                              transform=transform,
                                              all_touched=False,
                                              # merge_alg=MergeAlg.replace,  # ... used is default
                                              default_value=1,
                                              dtype=rasterio.uint8
                                              )

            return out
        else:
            return tt[['geometry', 'raster_val']]

    out = soften(tt, points)

    if array_out:
        shapes_ = out[['geometry', 'soft_label']].values.tolist()

        out = rasterio.features.rasterize(shapes_,
                                          out_shape=shape,
                                          fill=0,  # fill value for background
                                          out=None,
                                          transform=transform,
                                          all_touched=False,
                                          # merge_alg=MergeAlg.replace,  # ... used is default
                                          default_value=1,
                                          dtype=rasterio.uint8
                                          )

    return out


def homogenize_boundaries(prediction: np.ndarray, affine: list,
                          epsg: str = 'epsg:32633', boundary_code: int = 15,
                          array_out: bool = False) -> Union[gpd.GeoDataFrame, np.ndarray]:
    """
    Auxiliary function for homogenization of predictions based on boundary predictions
    Note that currently function is not very robust and expects that boundary is encoded as `15`

    Parameters
    ----------
    prediction: np.ndarray
    epsg: str
        epsg code of polygon layer
    affine: list
        affine transform of raster
    boundary_code: int
        integer label encoding boundary class (Default is 15)
    array_out: bool
        Whether to export homogenized prediction as numpy array
    Returns
    -------
    gpd.GeoDataFrame or np.ndarray
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
    super = np.where((pred_t1 == boundary_code) | ((pred_t2 == boundary_code) & (proba_t2 > 0.3)) | (pred_t1 == 0), 0,
                     1)
    altered = np.where((pred_t1 == boundary_code), pred_t2, pred_t1)

    labeled, ncomp = label(super, element)

    sizes = [np.where(labeled == i)[0].shape[0] for i in range(ncomp)]
    to_small = [i for i in range(ncomp) if sizes[i] < 13]

    # remove too small components
    labeled2 = np.where(np.isin(labeled, to_small), 0, labeled)

    extracted_shapes = prediction2polygon_layer(labeled2, affine, epsg=epsg)

    extracted_shapes = extracted_shapes[extracted_shapes.raster_val != 0]

    # extracted_shapes.plot()
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

    if array_out:
        shapes_ = tt[['geometry', 'raster_val']].values.tolist()
        transform = rasterio.Affine(affine[0][0], affine[1][0], affine[2][0],
                                    affine[0][1], affine[1][1], affine[2][1])
        out = rasterio.features.rasterize(shapes_,
                                          out_shape=pred_t1.shape,
                                          fill=0,  # fill value for background
                                          out=None,
                                          transform=transform,
                                          all_touched=False,
                                          # merge_alg=MergeAlg.replace,  # ... used is default
                                          default_value=1,
                                          dtype=rasterio.uint8
                                          )

        return out
    else:
        return tt[['geometry', 'raster_val']]
