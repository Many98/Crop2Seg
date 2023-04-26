import os
import shutil
from typing import List, Tuple, Union
from glob import glob

import gc

import pandas as pd
from tqdm.auto import tqdm
import logging
import numpy as np
import rasterio
import geopandas as gpd
from rasterio import features

from shapely.geometry import box as shapely_box

from datetime import datetime
from einops import rearrange

# ### small boiler plate to add src to sys path
import sys
from pathlib import Path
file = Path(__file__).resolve()
root = str(file).split('src')[0]
sys.path.append(root)
# --------------------------------------


from src.helpers.sentinel2raster import Sentinel2Raster, tile_coordinates, export_to_tif
from src.helpers.sentinel import sentinel

from src.global_vars import SENTINEL_PATH_DATASET, AGRI_PATH_DATASET, DATES, CLOUDS, TILES

logging.getLogger().setLevel(logging.INFO)


# TODO it should be processed slightly different for pretrain dataset, probably iterate using DatasetCreator via subdirs
#  containing different timeseries


class DatasetCreator(object):
    """
    Creates time-series dataset of Sentinel-2 tiles (patches) of shape T x C x H x W.
        C - all S2 bands except B01, B09, B10 i.e. 10 bands
        T  is dependent on tile
        H=W=128 pixels
    Currently only for Sentinel 2 tiles L2A.

    """

    def __init__(self, output_dataset_path: str, features_path: str = AGRI_PATH_DATASET,
                 tiles_path: str = SENTINEL_PATH_DATASET,
                 for_pretraining: bool = False,
                 download: bool = False,
                 delete_source_tiles: bool = False
                 ):
        """

        Parameters
        ----------
        output_dataset_path: str
            Absolute path where will be stored final dataset
        features_path: str
            Absolute path to shapefile.
            Shapefile is expected to have column `value` with integer encoding of classes
                        and column `geometry` with geometry object (which implements python Geo interface) of shape
            Note that `0` is reserved for background (nodata) class
        tiles_path: str
            Absolute path to Sentinel 2 L2A tiles (time-series) i.e. all tiles over same area will be used to
            create time-series
        for_pretraining: bool
            Whether to create dataset for pretraining.
            If `for_pretraining=True` it is expected that in `tiles_path` TODO ...
        download: bool
            Whether to also download particular tiles
        delete_source_tiles: bool
            Whether to delete input tiles after time-series per particular tile is generated
        """
        self.tiles_path = tiles_path  # here are stored sentinel 2 tiles
        self.features_path = features_path  # path of dataset where is stored shapefile
        self.out_path = output_dataset_path  # path to output dataset/ where to create dataset

        self.for_pretraining = for_pretraining

        self.download = download
        self.delete_source = delete_source_tiles

        os.makedirs(self.out_path, exist_ok=True)

        self.data_s2_path = os.path.join(self.out_path, 'DATA_S2')
        os.makedirs(self.data_s2_path, exist_ok=True)

        if not self.for_pretraining:
            self.segmentation_path = os.path.join(self.out_path, 'ANNOTATIONS')
            os.makedirs(self.segmentation_path, exist_ok=True)

        # TODO create also structure for PRETRAIN DATASET
        # TODO when creating pretrain dataset there must be another structure of `tiles_path` e.g. subdirectory
        #  for every time-series

        self.features = None

        if os.path.isfile(os.path.join(self.out_path, 'metadata.json')):
            self.metadata = DatasetCreator.load_metadata(os.path.join(self.out_path, 'metadata.json'))
        else:
            self.metadata = pd.DataFrame({'ID_PATCH': [], 'ID_WITHIN_TILE': [], 'Patch_Cover': [], 'Nodata_Cover': [],
                                          'Snow_Cloud_Cover': [], 'TILE': [], 'dates-S2': [], 'time-series_length': [],
                                          'crs': [], 'affine': [], 'Fold': [], 'Status': []})

    def __call__(self, *args, **kwargs):
        """
        Method to connect all processing steps
        Parameters
        ----------
        args :
        kwargs :

        Returns
        -------

        """
        tile_names = TILES

        for id, tile_name in enumerate(tile_names):

            # sanity check ... skip if there are already exported metadata (it means data should be also exported)
            if self.metadata[self.metadata['TILE'] == tile_name]['TILE'].shape[0] == 82 * 82:
                continue

            if self.download:
                logging.info(f"DOWNLOADING TILES WITH NAME: {tile_name}")
                self._download_timeseries(tile_name)  # TODO this will not work for PRETRAIN DATASET ... another dir struct

            logging.info(f"CONSTRUCTING TIME-SERIES FOR TILE: {tile_name}")
            time_series, bbox, affine, crs, file_names, dates = self._load_s2(tile_name)  # T x (C+1) x H x W
            logging.info(f"LENGTH OF TIME-SERIES IS: {len(file_names)}")

            time_series = self._preprocess(time_series)  # T x (C+1) x H x W

            if not self.for_pretraining:
                logging.info(f"GENERATING SEGMENTATION MASK FOR TILE: {tile_name}")
                segmentation_mask = self._create_segmentation(time_series.shape[-2:], affine, bbox)  # H x W
                patches_segment, _ = self._patchify(segmentation_mask, affine)  # NUM_PATCHES x PATCH_SIZE x PATCH_SIZE

            patches_s2, patches_affine = self._patchify(time_series,
                                                        affine)  # NUM_PATCHES x T x C+1 x PATCH_SIZE x PATCH_SIZE

            del time_series
            del segmentation_mask

            logging.info(f"POSTPROCESSING TIME-SERIES FOR TILE: {tile_name}")
            patches_bool_map, nodata_cover, snow_cloud_cover, patch_cover = self._postprocess_s2(patches_s2)
            if not self.for_pretraining:
                # NOTE that we set as valid only those patches which have background pixels percentage <= 0.7
                patches_bool_map, patch_cover = self._postprocess_segmentation(patches_segment, 0.7)

            # we do not save scene classification, it was only used to get number of nodata_pixels and snow&cloud pixels
            logging.info(f"SAVING TIME-SERIES DATA FOR TILE: {tile_name}")
            self._save_patches(patches_s2[:, :, :-1, ...], patches_bool_map, where=self.data_s2_path,
                               filename=f'S2', id=id)

            del patches_s2

            if not self.for_pretraining:
                self._save_patches(patches_segment, patches_bool_map, where=self.segmentation_path,
                                   filename=f'TARGET', id=id)

            del patches_segment

            logging.info(f"UPDATING METADATA FOR PATCHES OF TILE: {tile_name}")
            self._update_metadata(id, tile_name, dates, crs, patches_affine, patches_bool_map, nodata_cover,
                                  snow_cloud_cover, patch_cover)

            if self.delete_source:
                logging.info(f"REMOVING TILES WITH NAME: {tile_name}")
                self._delete_tiles(tile_name)

            logging.info(f"PATCHES FOR TILE {tile_name} GENERATED SUCCESSFULLY \n"
                         f"------------------------------------------------------\n\n")
            gc.collect()

        if not self.for_pretraining:
            logging.info(f"GENERATING RANDOM FOLDS")
            self._generate_folds()

    @staticmethod
    def load_metadata(metadata_path: str) -> pd.DataFrame:
        """
        static method to load metadata json file
        located at `metadata_path`
        Parameters
        -----------
        metadata_path: str
            Absolute path to metadata.json file
        Returns
        ---------
            pd.DataFrame representation of metadata
        """
        assert 'metadata.json' in metadata_path, '`metadata_path` is expected to have filename `metadata.json`'
        assert os.path.isfile(metadata_path), f'`metadata_path` ({metadata_path}) is now valid path'

        return pd.read_json(metadata_path, orient='records')

    def _delete_tiles(self, tile_name: str) -> None:
        """
        Helper method to delete all tiles with `tile_name`
        Parameters
        ----------
        tile_name : str
             Name of tile to be downloaded. E.g. T33UVR
        """
        file_names = self._get_filenames(tile_name)
        for f in file_names:
            shutil.rmtree(os.path.join(self.tiles_path, f))

    def _download_timeseries(self, tile_name: str) -> None:
        """
        Auxiliary method to download time-series by `tile_name`
        It should call `sentinel` etc functions
        Parameters
        ----------
        tile_name : str
            Name of tile to be downloaded. E.g. T33UVR
        Returns
        -------

        """
        for cloud, date in zip(CLOUDS, DATES):
            try:
                sentinel(polygon=None, tile_name=tile_name, platformname='Sentinel-2', producttype='S2MSI2A',
                         count=5,
                         beginposition=date,
                         cloudcoverpercentage=f'[0 TO {cloud}]', path_to_save=self.tiles_path)
            except RuntimeError as e:
                logging.info(e.__str__())
                logging.info(f'Skipping date:tile {date}:{tile_name}')
                continue

    def _patchify(self, data: np.ndarray, affine: rasterio.Affine,
                  patch_size: int = 128) -> Tuple[np.ndarray, list]:
        """
        Auxiliary method to patchify `time-series` to patches of size `patch_size x patch_size`
        i.e. input should be of shape T x C+1 x H x W and output of shape NUM_PATCHES x T x C+1 x PATCH_SIZE x PATCH_SIZE
        Parameters
        ----------
        data: np.ndarray
            time-series of Sentinel 2 L2A raw data of shape T x C+1 x H x W
        affine: rasterio.Affine
            Affine transform of input raster (tile).
            We need patchify it properly too
        Returns
        -------
        NUM_PATCHES x T x C+1 x PATCH_SIZE x PATCH_SIZE
        """
        # NOTE that each tile has shape 10980x10980 but each tile is overlapped on
        # each side by 4900 m which is 490 pixels
        # Therefore we will always take 484 pixels so it will be divisible by 128 which is about to be patch size
        #
        # T33UVS, T33UWS,  ... go from total left and on right side remove 484 pixels, and take from upper side 484 pixs
        # T33UUR, T33UVR, T33UWR, T33UXR, T33UYR,
        # T33UUQ, T33UVQ, T33UWQ, T33UXQ, T33UYQ

        # IT results in 82x82 patches

        assert patch_size == 128, 'Patch size should be 128'

        transform = rasterio.Affine(affine.a, affine.b,
                                    affine.c,  # (left bbox coordinate) in affine transform can be left unchanged
                                    affine.d, affine.e,
                                    affine.f - (affine.a * 484)  # fix (top bbox coordinate)
                                    )

        coords = tile_coordinates(transform, (10496, 10496), size=patch_size)

        cropped = data[..., 484:, :10496]

        # return rearrange(cropped, 't c (h h1) (w w1) -> (h w) t c h1 w1', h1=128, w1=128)
        return rearrange(cropped, '... (h h1) (w w1) -> (h w) ... h1 w1', h1=patch_size, w1=patch_size), \
               coords

    @staticmethod
    @export_to_tif
    def unpatchify(id: int, data: np.array, metadata_path: str, nodata: int = 0, dtype: str = 'uint8',
                   export: bool = False) -> Tuple[Union[rasterio.io.DatasetReader, rasterio.io.MemoryFile], str]:
        """
        Static method to create raster object from input `data` and using information from metadata located
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
            rdata.write(data)  # write the data
            rdata._set_all_descriptions(f'Unpatchified raster. Id: {id}')

        # with rasterio.open('example.tif', 'w', **profile) as dst:
        #         dst.write(data, 1)

        if export:
            os.makedirs(os.path.join(os.path.split(metadata_path)[0], 'export'), exist_ok=True)

        return memfile.open(), os.path.join(os.path.split(metadata_path)[0], 'export', f'raster_{id}')

    def _save_patches(self, data: np.ndarray, bool_map: np.ndarray, where: str, filename: str, id: int) -> None:
        """
        Auxiliary method to export/save created time-series patch
        It is of shape NUM_PATCHES x T x C x PATCH_SIZE x PATCH_SIZE
        Parameters
        ----------
        data: np.ndarray
            Array of shape NUM_PATCHES x T x C x PATCH_SIZE x PATCH_SIZE
                           or NUM_PATCHES x PATCH_SIZE x PATCH_SIZE
        bool_map: np.ndarray
            1D array bool map where True represents OK state
        where: str
            Absolute path for saved patches
        filename: str
            Filename of exported patch
        id: int
            Id of tile
        """
        for i, patch in enumerate(data):
            if bool_map[i]:
                with open(os.path.join(where, f'{filename}_{(id * data.shape[0]) + i}'), 'wb') as f:
                    np.save(f, patch)

    def _get_filenames(self, tile_name: str) -> List[str]:
        """
        Auxiliary method to get all filenames which contains `tile_name`
        Parameters
        ----------
        tile_name : str
            Name of tile
        Returns
        -------
            List of filenames containing `tile_name`
        """
        ff = glob(os.path.join(self.tiles_path, '*.SAFE'))
        ff = [os.path.split(f)[-1] for f in ff]
        return [f for f in sorted(ff,
                                  key=lambda x: datetime.strptime(x.split('_')[2][:8], '%Y%m%d')) if f.split('_')[5] == tile_name and
                f.split('_')[1].endswith('L2A')]

    def _load_s2(self, tile_name: str) -> Tuple[np.ndarray, rasterio.coords.BoundingBox, rasterio.Affine,
                                                int, List[str], List[str]]:
        """
        Auxiliary method to load time-series corresponding to `tile_name`.
        It also serves for up-sampling to 10m
        Parameters
        ----------
        tile_name : str
            Name of tile
        Returns
        -------
        extracted time-series array of shape T x (C+1) x H x W
        and bounding box of one tile
        and affine transform of one tile
        and epsg of coordinate reference system of one tile
        and list of filenames
        and list of dates
        """
        # load time serie of tile /use Sentinel2Raster interface\
        # do not remove scene classif -> T x C+1 x H x W
        # Sentinel2Raster will be also used to up-sampled 10 m

        file_names = self._get_filenames(tile_name)  # filenames sorted according to date

        rasters = [Sentinel2Raster(os.path.join(self.tiles_path, f)) for f in file_names]

        bbox = rasters[0].bounds
        crs = rasters[0].crs.to_epsg()

        # check whether CRS is UTM33N (epsg=32633)
        assert crs == 32633 and rasters[0].crs.to_epsg() == rasters[-1].crs.to_epsg(), 'Expected UTM33N crs'

        affine = rasters[0].transform
        dates = [r.date for r in rasters]

        logging.getLogger().disabled = True
        time_series = []
        for r in tqdm(rasters, desc="Loading rasters..."):
            time_series.append(r.read())
        logging.getLogger().disabled = False

        return np.stack(time_series, axis=0), bbox, affine, crs, file_names, dates

    def _load_features(self, bbox: rasterio.coords.BoundingBox = None) -> gpd.GeoDataFrame:
        """
        Auxiliary method to load shapefile of features which will be burned onto rasters.
        If bbox is not None then
        Returns
        -------
        GeodataFrame of (filtered) features
        """
        if self.features is None:
            logging.info(f"LOADING INPUT FEATURES FROM {self.features_path}")
            self.features = gpd.read_file(self.features_path)  # it should be .shp file
            logging.info("GENERATING SPATIAL INDEX FOR FEATURES")
            self.features.sindex

        assert isinstance(bbox,
                          rasterio.coords.BoundingBox), 'bbox is expected to be of type `rasterio.coords.BoundingBox`'
        if bbox is not None:
            logging.info(f"FILTERING FEATURES DATASET FOR CURRENT EXTENT")
            indices = self.features.sindex.query(shapely_box(bbox.left, bbox.bottom, bbox.right, bbox.top),
                                                 predicate='intersects')
        # return self.features.cx[bbox.left:bbox.right, bbox.bottom:bbox.top] if bbox is not None else self.features
        # cx method is too slow ... using spatial index instead

        return self.features.iloc[indices] if bbox is not None else self.features

    def _preprocess(self, time_series: np.ndarray) -> np.ndarray:
        """
        Auxiliary method to preprocess given time-series `time_series`.
        Particularly it should remove bands B01, B09, B10 per tile
        and upsample
        Parameters
        ----------
        time_series: np.ndarray
            Array of Sentinel 2 raw data (also with scene classification)

        Returns
        -------
        Preprocessed array of Sentinel 2 raw data
            Bands should be as follows:
            ['B4, central wavelength 665 nm', 'B3, central wavelength 560 nm', 'B2, central wavelength 490 nm',
             'B8, central wavelength 842 nm', 'B5, central wavelength 705 nm', 'B6, central wavelength 740 nm',
             'B7, central wavelength 783 nm', 'B8A, central wavelength 865 nm', 'B11, central wavelength 1610 nm',
             'B12, central wavelength 2190 nm', 'SCL, Scene Classification'
             ]
        """
        # remove not needed bands  (remove B01, B09, B10) per tile / B10 is already removed in L2A products
        assert time_series.ndim == 4, "Time-series is expected to be 4-dimensional [T x C x H x W] array, " \
                                      f"but `ndim={time_series.ndim}`"
        return time_series[:, [i for i in range(13) if i not in [10, 11]], ...]

    def _postprocess_s2(self, time_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        """
        Auxiliary method to postprocess `time_series` of S2 data
        Calculates nodata cover per time, snow&cloud cover per time,
        Parameters
        ----------
        time_series : np.ndarray
            Array of shape NUM_PATCHES x T x C x PATCH_SIZE x PATCH_SIZE
        Returns
        -------
            np.ndarray of shape [NUM_PATCHES], nodata_cover [NUM_PATCHES x T], snow_cloud_cover [NUM_PATCHES x T], None
        """
        # NOTE that patch size is 128 => there is 128 ** 2 pixels

        assert time_series.ndim == 5, '`time_series` argument is expected to be of dimension 3, but is of' \
                                      f'dimension {time_series.ndim}'
        tt = time_series[:, :, -1, ...]
        no_data = np.where(tt <= 1, 1, 0)
        cloud_snow_shadow = np.where(((2 <= tt) & (tt <= 3)) | (8 <= tt), 1, 0)

        return np.ones((time_series.shape[0],), dtype=bool), \
               rearrange(no_data, '... h w -> ... (h w)').sum(-1) / 128 ** 2, \
               rearrange(cloud_snow_shadow, '... h w -> ... (h w)').sum(-1) / 128 ** 2, None

    def _postprocess_segmentation(self, segmentation_mask: np.ndarray, threshold: float) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Auxiliary method to postprocess  `segmentation_mask`
        Calculates patch cover i.e. percentage of pixels different from background class
        Background pixels is expected to have value set to 0
        Parameters
        ----------
        segmentation_mask : np.ndarray
            Array of shape NUM_PATCHES x PATCH_SIZE x PATCH_SIZE
        threshold: float
            Number between 0-1 defining maximal allowed percentage of background pixels
        Returns
        -------
            np.ndarray bool map representing valid patches, patch_cover [NUM_PATCHES]
        """
        assert segmentation_mask.ndim == 3, '`segmentation_mask` argument is expected to be of dimension 3, but is of' \
                                            f'dimension {segmentation_mask.ndim}'

        background = np.where(segmentation_mask == 0, 1, 0)
        background_percentage = rearrange(background, '... h w -> ... (h w)').sum(-1) / 128 ** 2

        return np.where(background_percentage <= threshold, 1, 0).astype(bool), background_percentage

    def _update_metadata(self, id: int, tile_name: str, dates: List[str], crs: int,
                         affine: List[rasterio.Affine], patches_bool_map: np.ndarray,
                         nodata_cover: np.ndarray, snow_cloud_cover: np.ndarray,
                         patch_cover: np.ndarray) -> None:
        """
        Auxiliary method to update `self.metadata` DataFrame and `metadata.json` file

        Parameters
        ----------
        id : int
            Integer representing id of tile
        tile_name: str
            Name of tile
        dates: List[str]
            sorted list of dates within time series
        crs: int
            epsg number of coordinate reference system
        affine: List[rasterio.Affine]
            List of affine transforms for every patch
        patches_bool_map: np.ndarray
            Bool map representing valid patches
        nodata_cover: np.ndarray
            Array of shape NUM_PATCHES x T representing nodata cover
        snow_cloud_cover: np.ndarray
            Array of shape NUM_PATCHES x T representing snow&cloud cover
        patch_cover: np.ndarray
            Array of shape NUM_PATCHES representing patch cover i.e. percentage of all non-background pixels
        Returns
        -------
        """
        update = pd.DataFrame(
            {'ID_PATCH': [int((id * patches_bool_map.shape[0]) + i) for i in range(patches_bool_map.shape[0])],
             'ID_WITHIN_TILE': [int(i) for i in range(patches_bool_map.shape[0])],
             'Patch_Cover': [p for p in patch_cover],
             'Nodata_Cover': [{str(i): v for i, v in enumerate(p)} for p in nodata_cover],
             'Snow_Cloud_Cover': [{str(i): v for i, v in enumerate(p)} for p in snow_cloud_cover],
             'TILE': [tile_name for _ in range(patches_bool_map.shape[0])],
             'dates-S2': [{str(i): d for i, d in enumerate(dates)} for _ in
                          range(patches_bool_map.shape[0])],
             'time-series_length': [int(len(dates)) for _ in range(patches_bool_map.shape[0])],
             'crs': [int(crs) for _ in range(patches_bool_map.shape[0])],
             'affine': affine,
             'Fold': [-1 for _ in range(patches_bool_map.shape[0])],
             'Status': ['OK' if i else 'REMOVED' for i in patches_bool_map]})

        self.metadata = self.metadata.append(update, ignore_index=True)
        self.metadata = self.metadata.astype({'ID_PATCH': 'int32',
                                              'ID_WITHIN_TILE': 'int32',
                                              'Patch_Cover': 'float32',
                                              'time-series_length': 'int16',
                                              'crs': 'int16',
                                              'Fold': 'int16'})
        self.metadata.to_json(os.path.join(self.out_path, 'metadata.json'), orient="records", indent=4)

    def _generate_folds(self) -> None:
        """
        Auxiliary method to distribute patches (randomly) to 5 folds.
        It operates over `self.metadata` DataFrame
        """
        ok_indices = np.array(self.metadata[self.metadata['Status'] == 'OK'].index)
        np.random.shuffle(ok_indices)
        split_5 = np.array_split(ok_indices, 5)

        for i in range(len(split_5)):
            self.metadata.loc[split_5[i], 'Fold'] = i + 1

        self.metadata.to_json(os.path.join(self.out_path, 'metadata.json'), orient="records", indent=4)

    def _create_segmentation(self, shape: Tuple, affine: rasterio.Affine,
                             bbox: rasterio.coords.BoundingBox) -> np.ndarray:
        """
        Auxiliary method to create segmentation mask for one tile of `time_series`
        Output should be of shape H x W
        Parameters
        ----------
        shape: tuple
            Shape of output numpy ndarray
        bbox: rasterio.coords.BoundingBox
            Bounding box of tile for filtering shapefile/geopandas
        affine: rasterio.Affine
            affine transform of tile on which segmentation mask is burned
        Returns
        -------
        Array of shape H x W [uint8] containing segmentation mask
        """
        features_ = self._load_features(bbox)

        assert 'geometry' in features_, 'geometry column is expected to be in features GeoDataFrame'
        assert 'value' in features_, 'value column is expected to be in features GeoDataFrame and contain ' \
                                     'integer classes encoding'

        shapes_ = features_[['geometry', 'value']].values.tolist()

        # Note that `shapes_` input geometry must according to documentation implement Python geo interface
        #  and contain columns `geometry` and `value` which will be burned to output raster
        return rasterio.features.rasterize(shapes_,
                                           out_shape=shape,
                                           fill=0,  # fill value for background
                                           out=None,
                                           transform=affine,
                                           all_touched=False,
                                           # merge_alg=MergeAlg.replace,  # ... used is default
                                           default_value=1,
                                           dtype=rasterio.uint8
                                           )


if __name__ == "__main__":
    c = DatasetCreator(output_dataset_path='/disk2/<username>/S2TSCZCrop')
    c()
    print('Done')
