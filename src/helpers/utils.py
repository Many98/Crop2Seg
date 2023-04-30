from pyproj import Proj, Transformer, transform
import numpy as np
import rasterio
import os

from typing import Tuple, Union

from src.helpers.dataset_creator import DatasetCreator
from src.helpers.sentinel2raster import export_to_tif


def progress_bar(j, count, size=50, prefix=""):
    x = int(size * j / count)
    if j == count:
        print(f'{prefix} [{u"█" * x}{"." * (size - x)}]', flush=True)
    elif j < count:
        print(f'{prefix} [{u"█" * x}{"." * (size - x)}]', end="\r", flush=True)


def distribute_args(iterable, num_cpus):
    s = int(len(iterable) / num_cpus)
    args = [[i * s, (i + 1) * s] for i in range(num_cpus)]
    if (len(iterable)) % num_cpus != 0:
        args[-1][1] = len(iterable)

    return args


@export_to_tif
def unpatchify(id: int, data: np.array, metadata_path: str, nodata: int = 0, dtype: str = 'uint8',
               export: bool = False) -> Tuple[Union[rasterio.io.DatasetReader, rasterio.io.MemoryFile], str]:
    """
    Function to create raster object from input `data` and using information from metadata located
    at `metadata_path`
    Parameters
    ----------
    id: int
        Id of patch within self.metadata. It is needed to find proper affine transform and crs
    data: np.ndarray
        time-series of Sentinel 2 L2A raw data of shape T x C+1 x H x W
    metadata_path: str
        Absolute path to metadata.json file
    nodata: int
        Specify how to set nodata in output raster
    dtype: str
        Data type string representation for raster.
        Default is uint8
    export: bool
        Whether to export resulting raster. Raster will be exported to directory `export` near to `metadata_path`
    """
    metadata = DatasetCreator.load_metadata(metadata_path)

    assert data.ndim == 2, '`data` array is expected to be 2d (matrix)'

    affine = metadata[metadata['ID_PATCH'] == id]['affine'].values[0]
    crs_ = metadata[metadata['ID_PATCH'] == id]['crs'].values[0]

    profile = {'driver': 'GTiff', 'dtype': dtype, 'nodata': nodata, 'width': data.shape[1],
               'height': data.shape[0], 'count': 1,
               'crs': rasterio.crs.CRS.from_epsg(crs_),
               'transform': rasterio.Affine(affine[0][0], affine[1][0], affine[2][0],
                                            affine[0][1], affine[1][1], affine[2][1]),
               'blockxsize': 128,
               'blockysize': 128, 'tiled': True, 'compress': 'lzw'}

    # TODO MemoryFile does not empties /vsim/... -> stop using it ... use just plain np.array accessed via read method
    #  use NamedTemporaryFile instead https://rasterio.readthedocs.io/en/stable/topics/memory-files.html
    memfile = rasterio.io.MemoryFile(filename=f'raster_{id}.tif')
    with memfile.open(**profile) as rdata:
        rdata.write(data[None, ...].astype('uint8'))  # write the data

    # with rasterio.open('example.tif', 'w', **profile) as dst:
    #         dst.write(data, 1)

    if export:
        os.makedirs(os.path.join(os.path.split(metadata_path)[0], 'export'), exist_ok=True)

    return memfile.open(), os.path.join(os.path.split(metadata_path)[0], 'export', f'raster_{id}.tif')


def transform_from_crs(left, bottom, right, top, src_crs="EPSG:4326", dst_crs="EPSG:32633"):
    # transformer = Transformer.from_crs(src_crs, dst_crs)
    #  insert as array of x-coordinates, array of y cordinates (longitude (x) then latitude (y))
    # transformed = transformer.transform([left, right], [bottom, top])
    # above method not working so we will use deprecated method

    p_src = Proj(init=src_crs, preserve_units=False)
    p_dst = Proj(init=dst_crs, preserve_units=False)

    x1, y1 = p_src((left, right), (bottom, top))
    x2, y2 = transform(p_src, p_dst, x1, y1)

    return {'left': x2[0], 'bottom': y2[0],
            'right': x2[1], 'top': y2[1]}


def UTMtoWGS(vyrez):
    """Converts UTM zone 33N to WGS84.
    EPSG:32633 UTM zone 33N 
    EPSG:4326 WGS84 - World Geodetic System 1984; used in GPS

    Parameters
    vyrez : np.ndarray
        [[bottom, left], [top, right]]
    Returns
        [[left, bottom], [right, top]]
    """
    transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326")
    a = transformer.transform(vyrez[:, 0], vyrez[:, 1])
    return np.transpose(np.array(a))  # TODO this joke needs to be done in another way (at least np.flip via axis=0)


def WGStoUTM(vyrez):
    """Converts WGS84 to UTM zone 33N.
    EPSG:32633 UTM zone 33N 
    EPSG:4326 WGS84 - World Geodetic System 1984; used in GPS

    Parameters
    vyrez : np.ndarray
        [[bottom, left], [top, right]]
    Returns
        [[left, bottom], [right, top]]
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633")
    a = transformer.transform(vyrez[:, 0], vyrez[:, 1])

    return np.transpose(np.array(a))  # Longitude, Latitude  TODO and this joke as well (at least np.flip via axis=1)
