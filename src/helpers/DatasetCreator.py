import pickle
import os
from typing import List, Tuple

import pandas as pd
from tqdm.auto import tqdm
import logging
import numpy as np
import rasterio
from rasterio import mask
import geopandas as gpd
from rasterio import features
import time
from datetime import datetime
from einops import rearrange

from sentinel2raster import fast_tiling, Sentinel2Raster

from src.global_vars import SENTINEL_PATH_DATASET, AGRI_PATH_DATASET


# we have raster

# dataset for pretraining -> we need to download data and
# then create dataset by loading each tile remove some bands and patchify it to 128x128 pixels

# dataset for fine-tuning -> we need to download data for czech republic ... handled somewhere else

# TODO it should be processed slightly different for pretrain dataset, probably iterate using DatasetCreator via subdirs
#  containing different timeseries


class DatasetCreator(object):
    """
    Creates time-series dataset of Sentinel-2 tiles (patches) of shape T x C+T x H x W.
        C - all S2 bands except B01, B09, B10
        (+T is because there is also scene classification mask after  bands)
        H=W=128 pixels
    Currently only for Sentinel 2 tiles L2A.

    """

    def __init__(self, output_dataset_path: str, features_path: str = AGRI_PATH_DATASET,
                 tiles_path: str = SENTINEL_PATH_DATASET,
                 for_pretraining: bool = False,
                 class_mapping: dict = {0: 'nodata', 15: 'unknown'}):
        """

        Parameters
        ----------
        output_dataset_path: str
            Absolute path where will be stored final dataset
        features_path: str
            Absolute path to shapefile
        tiles_path: str
            Absolute path to Sentinel 2 L2A tiles (time-series) i.e. all tiles over same area will be used to
            create time-series
        for_pretraining: bool
            Whether to create dataset for pretraining.
            If `for_pretraining=True` it is expected that in `tiles_path` TODO ...
        class_mapping: dict
            Specify class mapping used in creation of dataset
        """
        self.tiles_path = tiles_path  # here are stored sentinel 2 tiles
        self.features_path = features_path  # path of dataset where is stored shapefile
        self.out_path = output_dataset_path  # path to output dataset/ where to create dataset

        self.for_pretraining = for_pretraining

        os.makedirs(self.out_path, exist_ok=True)

        if not self.for_pretraining:
            self.data_s2_path = os.path.join(self.out_path, 'DATA_S2')
            os.makedirs(self.data_s2_path, exist_ok=True)

            self.segmentation_path = os.path.join(self.out_path, 'ANNOTATIONS')
            os.makedirs(self.segmentation_path, exist_ok=True)

        # TODO create also structure for PRETRAIN DATASET

        self.additional_classes = class_mapping  # TODO specify it more correctly

        if os.path.isfile(os.path.join(self.out_path, 'metadata.json')):
            self.metadata = pd.read_json(os.path.join(self.out_path, 'metadata.json'))
        else:
            self.metadata = pd.DataFrame({'ID_PATCH': [], 'ID_WITHIN_TILE': [], 'Patch_Cover': [], 'Nodata_Cover': [],
                                          'Snow_Cloud_Cover': [], 'TILE': [], 'dates-S2': [], 'Fold': [],
                                          'Status': []})

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
        tile_names = []

        # TODO adjust it work with information from loaded metadata.json (e.g. if error occured)
        for id, tile_name in enumerate(tile_names):

            self._download_timeseries(tile_name)

            time_series, bboxes, file_names, dates = self._load_s2(tile_name)  # T x (C+1) x H x W

            time_series = self._preprocess(time_series)  # T x (C+1) x H x W

            if not self.for_pretraining:
                segmentation_mask = self._create_segmentation(time_series, bboxes)  # H x W
                patches_segment = self._patchify(segmentation_mask)  # NUM_PATCHES x PATCH_SIZE x PATCH_SIZE

            patches_s2 = self._patchify(time_series)  # NUM_PATCHES x T x C+1 x PATCH_SIZE x PATCH_SIZE

            patches_bool_map, nodata_cover, snow_cloud_cover, patch_cover = self._postprocess_s2(patches_s2)

            if not self.for_pretraining:
                patches_bool_map, patch_cover = self._postprocess_segmentation(patches_segment)

            # we do not save scene classification
            self._save_patches(patches_s2[:, :, :-1, ...], patches_bool_map, where=self.data_s2_path,
                               filename=f'S2_{tile_name}', id=id)

            if not self.for_pretraining:
                self._save_patches(patches_segment, patches_bool_map, where=self.segmentation_path,
                                   filename=f'TARGET_{tile_name}', id=id)

            self._update_metadata(id, tile_name, dates, patches_bool_map, nodata_cover, snow_cloud_cover, patch_cover)

        if not self.for_pretraining:
            self._generate_folds()
        self._generate_normalization()

    def _download_timeseries(self, tile_name: str) -> None:
        """
        auxiliary method to download time-series by `tile_name`
        It should call `sentinel` etc functions
        Parameters
        ----------
        tile_name :

        Returns
        -------

        """
        pass

    def _patchify(self, data: np.ndarray,
                  patch_size: int = 128) -> np.ndarray:
        """
        Auxiliary method to patchify `time-series` to patches of size `patch_size x patch_size`
        i.e. input should be of shape T x C+1 x H x W and output of shape NUM_PATCHES x T x C+1 x PATCH_SIZE x PATCH_SIZE
        Parameters
        ----------
        data: np.ndarray
            time-series of Sentinel 2 L2A raw data of shape T x C+1 x H x W
        Returns
        -------
        NUM_PATCHES x T x C+1 x PATCH_SIZE x PATCH_SIZE
        """
        # finally perform fasttiling 128x128 try not overlapping
        # TODO we can use einops instead
        # TODO note that each tile is overlaped on each side by 4900 m which is 490 pixels
        #  but we will always take 484 pixels so it will be divisible by 128 which is about to be patch size
        # T33UVS, T33UWS,  ... go from total left and on right side remove 490 pixels, and take from upper side 490 pixs
        # T33UUR, T33UVR, T33UWR, T33UXR, T33UYR,
        # T33UUQ, T33UVQ, T33UWQ, T33UXQ, T33UYQ

        assert patch_size != 128, 'Patch size should be 128'

        cropped = data[484:, :10496]

        # todo do same for segmentation mask, will we save it together or in different file ?
        # todo what to do with scene clsssification mask

        # return rearrange(cropped, 't c (h h1) (w w1) -> (h w) t c h1 w1', h1=128, w1=128)
        return rearrange(cropped, '... (h h1) (w w1) -> (h w) ... h1 w1', h1=128, w1=128)

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
                with open(os.path.join(where, f'{filename}_{i}_{id + i}'), 'wb') as f:
                    np.save(f, patch)

    def _get_filenames(self, tile_name: str) -> List[str]:
        """
        Auxiliary method to get all filenames which contains `tile_name`
        Parameters
        ----------
        tile_name :

        Returns
        -------

        """
        return [f for f in sorted(os.listdir(self.tiles_path),
                                  key=lambda x: datetime.strptime(x.split('_')[2][:8], '%Y%m%d')) if f.endswith('.SAFE')
                and f.split('_')[5] == tile_name and f.split('_')[1].endswith('L2A')]

    def _load_s2(self, tile_name: str) -> Tuple[np.ndarray, List[rasterio.coords.BoundingBox], List[str], List[str]]:
        """
        Auxiliary method to load time-series corresponding to `tile_name`.
        It also serves for up-sampling to 10m
        Returns
        -------
        extracted time-series array of shape T x (C+1) x H x W
        and list of bounding boxes
        and list of filenames
        and list of dates
        """
        # load time serie of tile /use Sentinel2Raster interface\
        # do not remove scene classif -> T x C+1 x H x W
        # Sentinel2Raster will be also used to up-sampled 10 m

        file_names = self._get_filenames(tile_name)  # filenames sorted according to date

        rasters = [Sentinel2Raster(f) for f in file_names]

        bboxes = [r.bounds for r in rasters]
        dates = [r.date for r in rasters]
        time_series = [r.read() for r in rasters]

        return np.stack(time_series, axis=0), bboxes, file_names, dates

    def _load_features(self) -> gpd.GeoDataFrame:
        """
        Auxiliary method to load shapefile of features which will be burned onto rasters
        Returns
        -------
        GeodataFrame of features
        """

        return gpd.read_file(self.features_path)  # it should be .shp file

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
        """
        # remove not needed bands  (remove B01, B09, B10) per tile / B10 is already removed in L2A products
        # somehow measure number of nodata/cloud pixels  (we can do this later)

        # TODO we need to check if B01, B09 bands are always on 11, 10
        #  Sentinel2Raster has .descriptions method ...

        return time_series[[i for i in range(13) if i not in [10, 11]]]

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
        pass
        """
        ########################################
                            # CHECK IF IMAGE IS NOT FULL OF NODATA #
                            # IN 122X122 IMAGE MUST BE AT LEAST    #
                            # 2000 NON ZERO PIXELS TO PASS         #
                            ########################################
                            if np.count_nonzero(np.where(sub_image['mask'][-1] == 14, 0, 1)) > 1999:
        """
        return np.ones((time_series.shape[0],), dtype=bool), \
               np.zeros((time_series.shape[0], time_series.shape[1]), dtype=float), \
               np.zeros((time_series.shape[0], time_series.shape[1]), dtype=float), None

    def _postprocess_segmentation(self, segmentation_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Auxiliary method to postprocess  `segmentation_mask`
        Calculates patch cover i.e. percentage of pixels different from background class
        Parameters
        ----------
        segmentation_mask : np.ndarray
            Array of shape NUM_PATCHES x PATCH_SIZE x PATCH_SIZE
        Returns
        -------
            np.ndarray, patch_cover [NUM_PATCHES]
        """
        pass
        """
        ########################################
                            # CHECK IF IMAGE IS NOT FULL OF NODATA #
                            # IN 122X122 IMAGE MUST BE AT LEAST    #
                            # 2000 NON ZERO PIXELS TO PASS         #
                            ########################################
                            if np.count_nonzero(np.where(sub_image['mask'][-1] == 14, 0, 1)) > 1999:
        """
        return np.ones((segmentation_mask.shape[0],), dtype=bool), \
               np.zeros((segmentation_mask.shape[0],), dtype=float)

    def _update_metadata(self, id: int, tile_name: str, dates: List[str], patches_bool_map: np.ndarray,
                         nodata_cover: np.ndarray, snow_cloud_cover: np.ndarray, patch_cover: np.ndarray) -> None:
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

        # TODO set Nodata_Cover and Snow_Cloud_Cover
        update = pd.DataFrame({'ID_PATCH': [id+i for i in range(patches_bool_map.shape[0])],
                               'ID_WITHIN_TILE': [i for i in range(patches_bool_map.shape[0])],
                               'Patch_Cover': [p for p in patch_cover],
                               'Nodata_Cover': [],
                               'Snow_Cloud_Cover': [],
                               'TILE': [tile_name for _ in range(patches_bool_map.shape[0])],
                               'dates-S2': [{str(i): d for i, d in enumerate(dates)} for _ in range(patches_bool_map.shape[0])],
                               'Fold': [0 for _ in range(patches_bool_map.shape[0])],
                               'Status': ['OK' if i else 'REMOVED' for i in patches_bool_map]})

        self.metadata = self.metadata.append(update)
        self.metadata.to_json(os.path.join(self.out_path, 'metadata.json'), orient="split")

    def _generate_folds(self) -> None:
        """
        Auxiliary method to distribute patches (randomly) to 5 folds.
        It operates over `self.metadata` DataFrame
        """
        pass

    def _generate_normalization(self) -> None:
        """
        Auxiliary method to calculate normalization values (mean, std) per fold
        and export `NORM_S2_patch.json file`
        Structure of json
        {
            "Fold_1": {
                "mean": [
                    1165.9398193359375,
                    1375.6534423828125,
                    1429.2191162109375,
                    1764.798828125,
                    2719.273193359375,
                    3063.61181640625,
                    3205.90185546875,
                    3319.109619140625,
                    2422.904296875,
                    1639.370361328125
                ],
                "std": [
                    1942.6156005859375,
                    1881.9234619140625,
                    1959.3798828125,
                    1867.2239990234375,
                    1754.5850830078125,
                    1769.4046630859375,
                    1784.860595703125,
                    1767.7100830078125,
                    1458.963623046875,
                    1299.2833251953125
                ]
            },
            ...
        }
        """
        # see compute_norm_vals in dataset.py
        pass

    def _create_segmentation(self,  # time_series: np.ndarray,
                             bboxes: List[rasterio.coords.BoundingBox]) -> np.ndarray:
        """
        Auxiliary method to create segmentation mask for one tile of `time_series`
        Note that it will operate only on one element of time-series.
        Output should be of shape H x W
        """
