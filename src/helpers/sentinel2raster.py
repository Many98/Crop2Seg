import os
import glob
import re
from datetime import datetime

import numpy as np

import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
from rasterio.windows import from_bounds
from rasterio import MemoryFile
from rasterio import plot

from utils import progress_bar


def export_to_tif(func_to_be_decorated):
    def wrapper(*args, **kwargs):
        raster_reader, path_to_export = func_to_be_decorated(*args, **kwargs)
        if kwargs.get('export', False):
            if not os.path.isfile(path_to_export):
                profile = raster_reader.profile
                profile.update(driver='GTiff', compress='lzw', blockxsize=128, blockysize=128)
                with rasterio.open(path_to_export, 'w', **profile) as dst:
                    dst.write(raster_reader.read())
            else:
                print(f'There already is file with given path `{path_to_export}`')
            raster_reader.close()
            return path_to_export
        return raster_reader

    return wrapper


def check_raster(raster_file, consider_S2R=False):
    """
    Checks if provided parameter `raster_file` is either
    str, rasterio.io.DatasetReader or rasterio.vrt.WarpedVRT, rasterio.io.MemoryFie
    Parameters
    ----------
    raster_file: str or rasterio.io.DatasetReader or rasterio.vrt.WarpedVRT or rasterio.io.MemoryFie
        Raster-like object
    consider_S2R: bool
        Whether to consider Sentinel2Raster object as raster-like object
    Returns
    -------

    """
    src = None
    if isinstance(raster_file, str):
        if os.path.isfile(raster_file):
            try:
                src = rasterio.open(raster_file)

            except rasterio.errors.RasterioError as e:
                print(e)
    elif isinstance(raster_file, rasterio.io.DatasetReader) or isinstance(raster_file, rasterio.vrt.WarpedVRT):
        if not raster_file.closed:
            src = raster_file
        else:
            raise Exception(f'{raster_file} is not opened')
    elif isinstance(raster_file, rasterio.io.MemoryFile):
        src = raster_file
    elif consider_S2R and isinstance(raster_file, Sentinel2Raster):
        if not raster_file.closed:
            src = raster_file
        else:
            raise Exception(f'{raster_file} is not opened Sentinel2Raster object')
    else:
        raise Exception(f'{raster_file} is not path to raster nor opened rasterio.io.DatasetReader or '
                        f'rasterio.vrt.WarpedVRT object')
    return src


def raster_plot(raster_file, title, cmap='RdYlGn'):
    """
    Plot raster
    Parameters
    ----------
    raster_file: str or rasterio.io.DatasetReader
    title
    cmap

    Returns
    -------

    """
    src = check_raster(raster_file)
    plot.show(src, title=title, cmap=cmap)
    src.close()


class Sentinel2Raster(object):
    """
    This class mainly serves as interface to Sentinel2 L2A tiles in .SAFE format but also is used
    for handling tile rasters generated after supperresolving with neural network (see sentinel.py module)
    It will unify raster resolution to 10m by cubic interpolation.
    Everything is done on the fly.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.split(self.file_path)[1]
        self.file_dir = os.path.split(self.file_path)[0]
        self.date = self.__extract_date()
        self.tile = self.__extract_tile()
        self.__closed = False
        self.__xml_path = None
        if self.__is_safe() and self.__get_xml() is not None:
            self.__xml_path = self.__get_xml()
            self.__dataset = rasterio.open(self.__xml_path, 'r')
            self.__property_reader = rasterio.open(self.__dataset.subdatasets[0])
        elif self.__is_raster():
            self.__dataset = rasterio.open(self.file_path, 'r')
            self.__property_reader = self.__dataset
        else:
            raise Exception(f'Invalid `file_path`: {self.file_path}'
                            'Should be raster or Sentinel-2 L2A tile in .SAFE format')
        self._data = None  # we will call method `__2raster` only if needed otherwise its just bottleneck
        self._descriptions = self.__property_reader.descriptions
        self._profile = self.__property_reader.profile
        self._count = self.__property_reader.count
        self._dtypes = self.__property_reader.dtypes

    def __enter__(self):
        if self._data is None:
            self._data = self.__2raster()
            # self.__property_reader = self.data
        return self._data  # TODO make this work probably with help of MemoryFile

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.__dataset.close()
        if self._data is not None:
            # self.data.close()
            del self._data
            self._data = None
        self.__closed = True
        # self.data = None

    def read(self, channel=None):
        if self._data is None:
            self._data = self.__2raster()
            # self.__property_reader = self.data
        print(f'Reading {self.tile_name}_{self.date} ...')
        if isinstance(channel, list):
            channel = np.array(channel)
        elif channel is None:
            return self._data[:]
        return self._data[channel - 1]

    @export_to_tif
    def to_tif(self, export=True):
        """
        Invokes `export_to_tif` decorator to export data to raster
        Specifically it can be used to export tile in .SAFE format to raster if needed
        Parameters
        ----------
        export

        Returns
        -------

        """
        if self._data is None:
            self._data = self.__2raster()
            # self.__property_reader = self.data
        return self._data, os.path.join(self.file_dir, self.file_name.split('.')[0] + '.tif')

    def to_temp_file(self):
        pass

    @property
    def closed(self):
        return self.__closed

    @property
    def day(self):
        return datetime.strptime(self.date, '%Y%m%d').day if self.date is not None else None

    @property
    def month(self):
        return datetime.strptime(self.date, '%Y%m%d').month if self.date is not None else None

    @property
    def year(self):
        return datetime.strptime(self.date, '%Y%m%d').year if self.date is not None else None

    @property
    def tile_name(self):
        return self.tile if self.tile is not None else None

    @property
    def crs(self):
        return self.__property_reader.crs

    @property
    def driver(self):
        return self.__property_reader.driver

    @property
    def profile(self):
        return self._profile

    @property
    def height(self):
        return self.__property_reader.height

    @property
    def width(self):
        return self.__property_reader.width

    @property
    def count(self):
        return self._count

    @property
    def transform(self):
        return self.__property_reader.transform

    @property
    def bounds(self):
        return self.__property_reader.bounds

    @property
    def shape(self):
        return self.__property_reader.shape

    @property
    def descriptions(self):
        return self._descriptions

    @property
    def block_shapes(self):
        return self.__property_reader.block_shapes

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def gcps(self):
        return self.__property_reader.gcps

    def __extract_date(self):
        # looking for 1st occurrence of date in YYYYMMDD format
        date = re.search("([0-9]{4}[0-9]{2}[0-9]{2})", self.file_name)
        return date.group(1) if date is not None else None

    def __extract_tile(self):
        # looking for 1st occurrence of tile name in T format
        tile_ = re.search("(T[0-9]{2}[A-Z]{3})", self.file_name)
        return tile_.group(1) if tile_ is not None else None

    def __get_xml(self):
        xml = None
        for file in glob.iglob(os.path.join(self.file_path, "MTD*.xml")):
            xml = file
        return xml

    def __is_safe(self):
        if '.SAFE' in self.file_path and 'GRANULE' in os.listdir(self.file_path):
            return True
        return False

    def __is_raster(self):
        if ('.tif' in self.file_path or '.tiff' in self.file_path or '.jp2' in self.file_path) \
                and os.path.isfile(self.file_path):
            return True
        return False

    def __get_scl(self):
        """
        Look for scene classification mask in same directory as input raster
        Returns
        -------

        """
        scl = None
        for file in glob.iglob(os.path.join(self.file_dir, f"{self.tile}_{self.date}*SCL*.jp2")):
            scl = file
        return scl

    def __2raster(self):
        progress_bar(j=0, count=12, prefix=f'Loading {self.tile_name}_{self.date} into memory')
        final_raster_array = []
        final_raster_profile = None
        final_raster_description = []

        if self.__is_safe():
            # read 10 m bands
            with rasterio.open(self.__dataset.subdatasets[0]) as raster_10_m:
                final_raster_profile = raster_10_m.profile
                final_raster_array.append(raster_10_m.read())
                final_raster_description = final_raster_description + list(raster_10_m.descriptions)

            progress_bar(j=2, count=12, prefix=f'Loading {self.tile_name}_{self.date} into memory')

            # read 20 m bands and upsample them to 10 m resolution
            with rasterio.open(self.__dataset.subdatasets[1]) as raster_20_m:
                final_raster_array.append(raster_20_m.read([i for i in range(1, 7)], out_shape=(
                    6,
                    int(raster_20_m.height * 2),
                    int(raster_20_m.width * 2)
                ),
                                                           resampling=Resampling.cubic))
                final_raster_description = final_raster_description + list(raster_20_m.descriptions[:6])

            progress_bar(j=4, count=12, prefix=f'Loading {self.tile_name}_{self.date} into memory')

            # read 60 m bands and upsample them to 10 m resolution
            with rasterio.open(self.__dataset.subdatasets[2]) as raster_60_m:
                final_raster_array.append(raster_60_m.read([1, 2], out_shape=(
                    2,
                    int(raster_60_m.height * 6),
                    int(raster_60_m.width * 6)
                ),
                                                           resampling=Resampling.cubic))
                final_raster_description = final_raster_description + list(raster_60_m.descriptions[:2])

            progress_bar(j=6, count=12, prefix=f'Loading {self.tile_name}_{self.date} into memory')

            # read scene classification mask and upsample it to 10 m resolution (with nearest neighbours method)
            with rasterio.open(self.__dataset.subdatasets[1]) as raster_20_m:
                final_raster_array.append(raster_20_m.read(9, out_shape=(
                    1,
                    int(raster_20_m.height * 2),
                    int(raster_20_m.width * 2)
                ),
                                                           resampling=Resampling.nearest)[np.newaxis, :, :])
                final_raster_description = final_raster_description + list(raster_20_m.descriptions[8:9])

            progress_bar(j=8, count=12, prefix=f'Loading {self.tile_name}_{self.date} into memory')
        else:
            # read 10 m bands
            progress_bar(j=2, count=12, prefix=f'Loading {self.tile_name}_{self.date} into memory')
            final_raster_profile = self.__dataset.profile
            final_raster_array.append(self.__dataset.read())
            final_raster_description = final_raster_description + list(self.__dataset.descriptions)

            progress_bar(j=6, count=12, prefix=f'Loading {self.tile_name}_{self.date} into memory')

            if self.__get_scl() is not None:
                # read scene classification mask and upsample it to 10 m resolution (with nearest neighbours method)
                with rasterio.open(self.__get_scl()) as scl:
                    final_raster_array.append(scl.read(1, out_shape=(
                        1,
                        int(scl.height * 2),
                        int(scl.width * 2)
                    ),
                                                       resampling=Resampling.nearest)[np.newaxis, :, :])
                    final_raster_description = final_raster_description + ['SCL Scene classification']
                progress_bar(j=8, count=12, prefix=f'Loading {self.tile_name}_{self.date} into memory')
            else:
                print(f'No scene classification raster found')

        final_raster_array = np.concatenate(final_raster_array, axis=0)
        final_raster_profile.update(driver='GTiff', count=final_raster_array.shape[0], compress='lzw',
                                    dtype=rasterio.uint16)
        progress_bar(j=9, count=12, prefix=f'Loading {self.tile_name}_{self.date} into memory')

        # TODO MemoryFile does not empties /vsim/... -> stop using it ... use just plain np.array accessed via read method
        # memfile = MemoryFile(filename='my_raster.tif')
        # with memfile.open(**final_raster_profile) as data:
        #    data.write(final_raster_array)  # write the data
        #    data._set_all_descriptions(final_raster_description)

        progress_bar(j=11, count=12, prefix=f'Loading {self.tile_name}_{self.date} into memory')

        # m = memfile.open()
        self._profile = final_raster_profile
        self._descriptions = final_raster_description
        self._count = final_raster_array.shape[0]
        self._dtypes = self._count * (self._profile.get('dtype'),)
        progress_bar(j=12, count=12, prefix=f'Loading {self.tile_name}_{self.date} into memory')

        return final_raster_array


@export_to_tif
def raster_resample(raster_file, factor, method=Resampling.nearest, export=False):
    """
    Creates new resampled raster.
    Parameters
    ----------
    raster_file: str or rasterio.io.DatasetReader or rasterio.vrt.WarpedVRT
        Absolute path to raster file or raster DatasetReader object
    factor: tuple or float
        Factor > 1 will upsample data
    method: rasterio.enums.Resampling
        Method used for resampling. Can be Resampling.nearest or Resampling.bilinear, Resampling.bilinear

    export: bool
        Whether to export result into raster file otherwise numpy array and raster profile is returned
    Returns
    -------
    Absolute path to resampled raster if `export_to_tif=True` otherwise rasterio.io.DatasetReader and
    new raster filename

    """
    if isinstance(factor, tuple):
        factor_w, factor_h = factor
    else:
        factor_w, factor_h = factor, factor
    src = check_raster(raster_file)
    # scale image transform
    transform = src.transform * src.transform.scale(
        (src.width / int(src.width * factor_w)),
        (src.height / int(src.height * factor_h))
    )
    new_path = src.files[0].split('.')[0] + f'_resampled_to_{int(transform.a)}m.tif'

    if factor != 1.0:
        # resample on-the-fly
        vrt = WarpedVRT(src, transform=transform, height=int(src.height * factor_h), width=int(src.width * factor_w),
                        resampling=method)
        # close only if it is instance of rasterio.io.DatasetReader
        # closing WarpedVRT object object causes problems with reading from wrong memory location
        if isinstance(src, rasterio.io.DatasetReader):
            src.close()
        return vrt, new_path
    else:
        return src, new_path


@export_to_tif
def raster_reproject(raster_file, dst_crs='EPSG:32633', method=Resampling.nearest, export=False,
                     resolution=None):
    """
    Creates new reprojected raster
    Parameters
    ----------
   raster_file: str or rasterio.io.DatasetReader or rasterio.vrt.WarpedVRT
        Absolute path to raster file or raster rasterio.io.DatasetReader object or rasterio.vrt.WarpedVRT object
    dst_crs: str
        Coordinate reference system of destination raster e.g. 'EPSG:32633'
    method: rasterio.enums.Resampling
        Method used for resampling. Can be Resampling.nearest or Resampling.bilinear, Resampling.bilinear
    export: bool
        Whether to export result into raster file otherwise numpy array and raster profile is returned

    Returns
    -------
    absolute path to reprojected raster if `export_to_tif=True` otherwise VRT object and new raster path
    """
    src = check_raster(raster_file)
    new_path = src.files[0].split('.')[0] + f'_reprojected_to_{dst_crs}.tif'
    if src.crs != dst_crs:
        # reproject on-the-fly
        left, bottom, right, top = src.bounds
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs=src.crs, dst_crs=dst_crs, width=src.width, height=src.height,
            left=left, bottom=bottom, right=right, top=top,
            resolution=src.transform[0] if resolution is None else resolution)  # we do not want change in resolution !!
        # this is needed only if working with
        # aproximate arcseconds resolutions
        vrt = WarpedVRT(src, crs=dst_crs, resampling=method,
                        transform=dst_transform, width=dst_width, height=dst_height)
        # close only if it is instance of rasterio.io.DatasetReader
        # closing WarpedVRT object object causes problems with reading from wrong memory location
        if isinstance(src, rasterio.io.DatasetReader):
            src.close()
        return vrt, new_path
    else:
        return src, new_path


@export_to_tif
def normalized_difference_index(raster_file, band1, band2, min_max=(0, 10000), export=False):
    """
    Calculates normalized index. Calculation is performed only on valid data pixels other pixels are set to -2
    As valid pixels are considered only pixels with values 4, 5, 6, 7 coresponding to vegetation, not_vegetated, water
    and not classified according to Sentinel-2  scene classification documentation see
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    Parameters
    ----------
    raster_file: str or Sentinel2Raster
        Absolute path to Sentinel-2 raster file
    band1: str
        Name of band in Sentinel2 convention. E.g. Can be `B1`, `B2`, ... `B12`
    band2: str
        Name of band in Sentinel2 convention. E.g. Can be `B1`, `B2`, ... `B12`
    min_max: tuple or None
        Defines range of new array after minmax normalization. If None then nothing happens.
        Supported is only nonnegative range with max value of 60000. Nodata will be set from -2.0 to 65535
    export: bool
        Whether to export result into raster file otherwise numpy array and raster profile is returned
    Returns
    -------
    TODO add status bar
    TODO stop using MemoryFile because of memory leakes .. currently used only if `export=True`
    """
    possible_bands = ['B' + str(i) for i in range(1, 13)]
    possible_bands.remove('B10')

    if band1 not in possible_bands or band2 not in possible_bands:
        raise Exception(f'band1:{band1} and band2:{band2} must be one of {possible_bands}')

    if isinstance(raster_file, Sentinel2Raster):
        if not raster_file.closed:
            raster = raster_file
    elif isinstance(raster_file, str):
        raster = Sentinel2Raster(raster_file)  # contains raster and scene classification resampled to 10m resolution

    desc = raster.descriptions

    b1_index = [i for i in range(len(desc)) if re.search(f'({band1} )', desc[i])]
    b2_index = [i for i in range(len(desc)) if re.search(f'({band2} )', desc[i])]

    if b1_index and b2_index:
        bands_array = raster.read([b1_index[0] + 1, b2_index[0] + 1]).astype('float32')
    else:
        raise Exception(f'Index of band not found in raster, possibly {band1} or {band2} is not present in raster'
                        f'description')
    mask_index = [i for i in range(len(raster.descriptions)) if re.search(f'(SCL )', raster.descriptions[i])]
    if mask_index:
        mask_array = raster.read(mask_index[0] + 1)
        bands_array = np.where((mask_array == 4) | (mask_array == 5) | (mask_array == 6) | (mask_array == 7),
                               bands_array, 0).astype('float32')
    else:
        print('Mask of raster not found. Resulting index can contain invalid values')

    np.seterr(divide='ignore', invalid='ignore')
    index = np.where(bands_array[0] + bands_array[1] == 0., -2.0,
                     (bands_array[0] - bands_array[1]) / (bands_array[0] + bands_array[1]))

    index_profile = raster.profile
    index_profile.update(dtype='float32')
    index_profile.update(count=1)
    index_profile.update(nodata=-2)

    if min_max is not None:
        if min_max[0] >= 0 and min_max[1] <= 60000:
            std = (index + 1) / 2
            index = std * (min_max[1] - min_max[0]) + min_max[0]
            index = np.where(index < 0, 65535, index).astype('uint16')
            index_profile.update(dtype='uint16')
            index_profile.update(nodata=65535)
        else:
            raise Exception(f'{min_max} should be non-negative range with maxvalue of 60000')

    new_path = raster.file_path.split('.')[0] + f'_normalized_difference_{band1}_{band2}.tif'
    if export:
        memfile = MemoryFile()
        data = memfile.open(**index_profile)  # DatasetWriter object
        data.write(np.expand_dims(index, axis=0))  # write the data
        data._set_all_descriptions((f'Normalized difference index of bands {band1}, {band2}',))

        data.close()  # close the DatasetWriter

        # raster.close()

        return memfile.open(), new_path
    return np.expand_dims(index, axis=0), new_path


def pixel2coord(raster_file):
    """
    Returns arrays of longitudes and latitudes
    Parameters
    ----------
    raster_file: str or Sentinel2Raster
        Absolute path of source raster
    Returns
    -------
    """
    src = check_raster(raster_file, consider_S2R=True)

    height = src.shape[0]
    width = src.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset='ul')  # upper left corner
    lons = np.array(xs)
    lats = np.array(ys)
    # src.close()
    return np.stack([lons, lats], axis=0)


def tile_coordinates(transform, shape, size=122):
    """
    Generates list of tuples where each tuple contains necessary data to create rasterio.Affine object i.e. each tuple
    is representation of affine transform of sub-tile of source tile (param `transform`)
    Each tuple contain all information needed for creation array of coordinates while ensuring more effective
    way of storage.
    Parameters
    ----------
    transform: rasterio.Affine
    shape
    size

    Returns
    -------

    """
    if not isinstance(transform, rasterio.Affine):
        raise Exception(f'{transform} must be of type `rasterio.Affine`')
    return [(transform.column_vectors[0], transform.column_vectors[1], (i, j)) for j in
            np.arange(transform.f, transform.f - transform.a * shape[1], transform.e * size) for i in
            np.arange(transform.c, transform.c + transform.a * shape[0], transform.a * size)
            ]


def fast_tiling(tile, size):
    """
    Function for fast tiling of array into smaller square tiles i.e. input tile of shape (num_channels, height, width)
    will be tiled into shape ((height * width) / size, num_channels, size, size) in row by row fashion.
    Used for tiling Sentinel2 tiles into smaller sub-tiles e.g. of size 122x122 pixels
    Parameters
    ----------
    tile: np.ndarray
        Array of shape (num_channels, height, width) or (height, width)
    size: int
        Size of sub-tiles in pixels
    Returns
    -------
    tiled: np.ndarray

    # TODO we can now use einops.rearrange instead of this function
        einops.rearrange(a, 'c (h h1) (w w1) -> (h w) c h1, w1', h1=size, w1=size)
    """
    if not isinstance(tile, np.ndarray):
        raise Exception(f'Tile must be of type numpy.Array')
    if not isinstance(size, int):
        raise Exception('Size must be integer')
    # TODO add checking for size
    if tile.ndim == 2:
        tile = np.expand_dims(tile, axis=0)
    if tile.ndim != 3:
        raise Exception(f'Tile must be 3 or 2 dimensional: (num_channels, height, width) or (height, width)')

    tiled = np.lib.stride_tricks.as_strided(tile, shape=(tile.shape[0], tile.shape[1] // size,
                                                         tile.shape[2] // size, size, size),
                                            strides=tile.itemsize * np.array([tile.shape[1] * tile.shape[2],
                                                                              tile.shape[2] * size, size,
                                                                              tile.shape[2], 1]))
    tiled = tiled.reshape(
        (tile.shape[0], (tile.shape[1] // size) * (tile.shape[2] // size), size, size)).transpose(1, 0, 2, 3)

    return tiled


def raster_vstack(raster_up, raster_down, out_file_path='', export=False):
    """
    Auxiliary function to vertically stack to rasters.
    Parameters
    ----------
    raster_up:
        raster-like object
    raster_down:
        raster-like object
    out_file_path: str
    export:bool
    Returns
    -------

    """
    src_up = check_raster(raster_up, consider_S2R=True)
    src_down = check_raster(raster_down, consider_S2R=True)

    upper_array = src_up.read(1)
    lower_array = src_down.read(1)

    profile = src_up.profile

    # merge arrays
    merged_array = np.concatenate((upper_array, lower_array), axis=0)
    profile.update(height=merged_array.shape[0], width=merged_array.shape[1])

    merged_mem = MemoryFile()
    with merged_mem.open(**profile) as out:
        out.write(np.expand_dims(merged_array, axis=0))

    return merged_mem.open(), os.path.join(out_file_path, '_merged.tif')


def raster_hstack():
    pass


def raster_window(bounds, raster, to_slice=False):
    """
    Creates window object from bounds applied to raster.
    Useful when needed crop part of raster
    (works also with Sentinel2Raster object)
    Parameters
    ----------
    bounds: list
    raster: raster-like object
    to_slice: bool
        Whether to create slice e.g. for ndarray slicing
    Returns
    -------
    window: rasterio.windows.Window if `to_slice=False` otherwise pair of slices (row_slice, col_slice)
    """
    window = from_bounds(*bounds,
                         transform=raster.transform)

    return window if not to_slice else window.toslices()


# proposals to work with sentinel1
'''
def sentinel_1_mosaic():
    from rasterio.merge import merge
    import glob
    readers = []  # here must be rasters in 'r' mode
    mosaic, out_trans = merge(readers)
    # this worked better g = gdal.Warp("output.tif", ['out_up', 'out_down'], format="GTiff", options=["COMPRESS=LZW", "TILED=YES"])


def sentinel_plot(src):
    # plots also coordinates so it is nicer
    from rasterio.plot import show
    show(src.read(), transform=src.transform)


def sentinel_transform_from_gcps():
    import rasterio
    src = rasterio.open("path/to/input/raster")
    with rasterio.open("path/to/output/raster", "w", **src.profile) as dst:
        data = src.read()
        result = data.astype(src.profile["dtype"], casting="unsafe")
        dst.write(result)
        dst.transform = rasterio.transform.from_gcps(src.gcps[0])
        dst.crs = src.gcps[1]
'''

# TODO need some fixes when working with MemoryFiles eg `raster_reproject` can have problem with filename
