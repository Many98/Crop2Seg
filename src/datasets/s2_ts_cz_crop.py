import json
import os
from datetime import datetime
import logging

from tqdm import tqdm

import numpy as np
from scipy.ndimage.measurements import label
import pandas as pd
import torch
import torch.utils.data as tdata

import rasterio

from typing import Union, Tuple

# ### small boiler plate to add src to sys path
import sys
from pathlib import Path

file = Path(__file__).resolve()
root = str(file).split('src')[0]
sys.path.append(root)
# --------------------------------------

from src.helpers.sentinel2raster import export_to_tif
from src.global_vars import TILES
from src.helpers.utils import get_row_col

logging.getLogger().setLevel(logging.INFO)

# labels
labels = ['Background 0', 'Permanent grassland 1', 'Annual fruit and vegetable 2', 'Summer cereals 3',
          'Winter cereals 4', 'Rapeseed 5', 'Maize 6', 'Annual forage crops 7', 'Sugar beet 8', 'Flax and Hemp 9',
          'Permanent fruit 10', 'Hopyards 11', 'Vineyards 12', 'Other crops 13', 'Not classified 14']

labels_short = ['Background 0', 'Grassland 1', 'Fruit & vegetable 2', 'Summer cereals 3',
                'Winter cereals 4', 'Rapeseed 5', 'Maize 6', 'Forage crops 7', 'Sugar beet 8', 'Flax & Hemp 9',
                'Permanent fruit 10', 'Hopyards 11', 'Vineyards 12', 'Other crops 13', 'Not classified 14']

labels_super_short = ['Background', 'Grassland', 'Fruit/vegetable', 'Summer cereals',
                      'Winter cereals', 'Rapeseed', 'Maize', 'Forage crops', 'Sugar beet', 'Flax/Hemp',
                      'Permanent fruit', 'Hopyards', 'Vineyards', 'Other crops', 'Not classified', 'Boundary']

labels_super_short_2 = ['Background', 'Grassland', 'Fruit/vegetable', 'Summer cereals',
                        'Winter cereals', 'Rapeseed', 'Maize', 'Forage crops', 'Sugar beet', 'Flax/Hemp',
                        'Permanent fruit', 'Hopyards', 'Vineyards', 'Other crops', 'Not classified']


def crop_cmap():
    """
    Auxiliary function to return dictionary with color map used for visualization
    of classes in S2TSCZCrop dataset
    """

    def get_rgb(h):
        return list(np.array(list(int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))) / 255) + [1]

    return {0: [0, 0, 0, 1],  # Background
            1: get_rgb('#a0db8e'),  # Permanent grassland
            2: get_rgb('#cc5500'),  # Annual fruit and vegetable
            3: get_rgb('#e9de89'),  # Summer cereals
            4: get_rgb('#f4ecb1'),  # Winter cereals
            5: get_rgb('#dec928'),  # Rapeseed
            6: get_rgb('#f0a274'),  # Maize
            7: get_rgb('#556b2f'),  # Annual forage crops
            8: get_rgb('#94861b'),  # Sugar beat
            9: get_rgb('#767ee1'),  # Flax and Hemp
            10: get_rgb('#7d0015'),  # Permanent fruit
            11: get_rgb('#9299a9'),  # Hopyards
            12: get_rgb('#dea7b0'),  # Vineyards
            13: get_rgb('#ff0093'),  # Other crops
            14: get_rgb('#c0d8ed'),  # Not classified (removed from training and evaluation)
            15: [1, 1, 1, 1]  # Border of semantic classes
            }


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
    assert 'metadata.json' in metadata_path, '`metadata_path` is expected to have filename `metadata.json`'
    assert os.path.isfile(metadata_path), f'`metadata_path` ({metadata_path}) is now valid path'

    metadata = pd.read_json(metadata_path, orient='records', dtype={'ID_PATCH': 'int32',
                                                                    'ID_WITHIN_TILE': 'int32',
                                                                    'Background_Cover': 'float32',
                                                                    'time-series_length': 'int16',
                                                                    'crs': 'int16',
                                                                    'Fold': 'int8'})

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


class S2TSCZCropDataset(tdata.Dataset):
    """
    Pytorch Dataset class to load samples from the S2TSCZCrop dataset for semantic segmentation of crop types
    from in time-series of Sentinel-2 tiles over Czech republic
    The Dataset yields ((data, dates), target) tuples, where:
        - data contains the image time series
        - dates contains the date sequence of the observations expressed in number
          of days since a reference date or expressed as day of year
          or vector of two sequences with both representations
        - target is the semantic or instance target
    """

    def __init__(
            self,
            folder,
            norm=True,
            norm_values=None,
            cache=False,
            mem16=False,
            folds=None,
            set_type=None,
            reference_date="2018-09-01",
            class_mapping=None,
            mono_date=None,
            from_date=None,
            to_date=None,
            channels_like_pastis=True,
            use_doy=False,
            use_abs_rel_enc=False,
            transform=None,
            add_ndvi=False,
            temporal_dropout=0.0,
            get_affine=False,
            for_inference=False,
            *args, **kwargs
    ):
        """
        Args:
            folder (str): Path to the dataset folder (directory)
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
                When Use `norm_values_folder` argument instead
            norm_values (dict): Defines normalization values for dataset
                {"mean": [...], "std": [...]}
                                Used only of norm is set to true
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            cache (bool): If True, the loaded samples stay in RAM, default False.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                They are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            set_type (str): Type of set. Can be one of 'train', 'val', 'test'
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            from_date (int or str, optional):

            to_date: (int or str, optional):
            channels_like_pastis: bool
                Whether to rearrange channels to be in same order like in PASTIS dataset.
                It should be used if one want to fine-tune using UTAE's original pretrained weights.
                Particularly PASTIS has channels ordered like this:
                    [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
                while S2TSCZCrop has channels ordered as found in .SAFE format:
                    [B04, B03, B02, B08, B05, B06, B07, B8A, B11, B12]
            use_doy: (bool) Whether to use absolute positions for time-series i.e. day of years instead
                           of difference between date and `self.reference_date`
            use_abs_rel_enc: (bool)
                Whether to use both relative day positions and absolute (doy) positions.
                If True parameter `use_doy` will be unused
            transform: (torchvision.transform)
                Transformation which will be applied to input tensor and mask
            add_ndvi: (bool)
                Whether to add NDVI channel at the end of spectral channels
            temporal_dropout: (float)
                Probability of temporal dropout
            get_affine: (bool)
                Whether to return also affine transforms of data

        """
        super().__init__()
        self.folder = folder
        self.norm = norm
        self.norm_values = norm_values
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.use_abs_rel_enc = use_abs_rel_enc
        self.use_doy = False if use_abs_rel_enc else use_doy
        self.set_type = set_type
        self.get_affine = get_affine
        self.for_inference = for_inference

        self.transform = transform
        self.add_ndvi = add_ndvi
        self.temporal_dropout = temporal_dropout

        assert set_type is not None and set_type in ['train', 'test', 'val'], f"`set_type` parameter must be one of" \
                                                                              f"['train', 'test', 'val'] but is {set_type}"

        # simple fix to get same order of channels like in PASTIS dataset
        self.channels_like_pastis = channels_like_pastis
        self.channels_order = [2, 1, 0, 4, 5, 6, 3, 7, 8, 9] if channels_like_pastis else [_ for _ in range(10)]

        self.cache = cache
        self.mem16 = mem16

        self.from_date = from_date  # TODO implement it
        self.to_date = to_date
        if mono_date is not None:
            self.mono_date = (
                datetime(*map(int, mono_date.split("-")))
                if "-" in mono_date
                else int(mono_date)
            )
        else:
            self.mono_date = mono_date

        self.memory = {}
        self.memory_dates = {}

        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )

        # Get metadata
        logging.info("Reading patch metadata . . .")
        if not self.for_inference:
            self.meta_patch = pd.read_json(os.path.join(folder, "metadata.json"), orient='records',
                                           dtype={'ID_PATCH': 'int32',
                                                  'ID_WITHIN_TILE': 'int32',
                                                  'Background_Cover': 'float32',
                                                  'time-series_length': 'int8',
                                                  'crs': 'int16',
                                                  'Fold': 'int8'})

            self.meta_patch = self.meta_patch[(self.meta_patch['Status'] == 'OK') & (self.meta_patch['set'] == set_type)]

            if self.meta_patch.empty:
                create_train_test_split(self.folder)
                self.meta_patch = self.meta_patch[
                    (self.meta_patch['Status'] == 'OK') & (self.meta_patch['set'] == set_type)]
        else:
            self.meta_patch = pd.read_json(os.path.join(folder, "metadata.json"), orient='records',
                                           dtype={'ID_PATCH': 'int32',
                                                  'time-series_length': 'int8',
                                                  'crs': 'int16'})

        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        logging.info("Done.")

        # Select Fold samples
        '''
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )
        '''

        # self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        # Get normalisation values
        if norm:
            if norm_values is None or not isinstance(norm_values, dict):
                raise Exception(f"Norm parameter set to True but normalization values are not provided.")

            self.norm = {'S2': (
                torch.from_numpy(norm_values['mean']).float(),
                torch.from_numpy(norm_values['std']).float(),
            )}
        else:
            self.norm = None
        logging.info("Dataset ready.")

    def __len__(self):
        return self.meta_patch.shape[0]

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def get_dates_relative(self, id):
        """
        Method returns array representing difference between date and `self.reference_date` i.e.
        position of element within time-series is relative to  `self.reference_date`
        """
        d = pd.DataFrame().from_dict(self.meta_patch.loc[id, 'dates-S2'], orient="index")
        d = d[0].apply(
            lambda x: (
                    datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                    - self.reference_date
            ).days
        )

        return d.values

    def get_dates_absolute(self, id):
        """
        Method returns array representing day of year for a date i.e.
        position of element within time-series is absolute to with respect to actual year.
        Using only 365 days long years
        """
        d = pd.DataFrame().from_dict(self.meta_patch.loc[id, 'dates-S2'], orient="index")
        d = pd.to_datetime(d[0].astype(str), format='%Y%m%d').dt.dayofyear

        return d.values

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        if self.get_affine:
            affine = self.meta_patch[self.meta_patch.ID_PATCH == id_patch]['affine'].values[0]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {
                'S2': np.load(
                    os.path.join(
                        self.folder,
                        f"DATA_S2",
                        f"S2_{id_patch}",  # f"S2_{id_patch}.npy",
                    )
                ).astype(np.float32)
            }  # T x C x H x W arrays
            data = {s: torch.from_numpy(a)[:, self.channels_order, ...] for s, a in data.items()}

            if self.add_ndvi:
                # NDVI B08(NIR) - B04(Red) / B08(NIR) + B04(Red)
                # NDVI red edge B08(NIR) - B06(Red edge) / B08(NIR) + B06(Red edge)
                #  we do not normalize it (its by definition between -1, 1)
                #  no data gets 0 as value

                data_ = data['S2']

                if self.channels_like_pastis:
                    ndvi = torch.where(data_[:, 6, ...] + data_[:, 2, ...] == 0, 0.,
                                       (data_[:, 6, ...] - data_[:, 2, ...]) / (data_[:, 6, ...] + data_[:, 2, ...]))
                else:
                    ndvi = torch.where(data_[:, 3, ...] + data_[:, 0, ...] == 0, 0.,
                                       (data_[:, 3, ...] - data_[:, 0, ...]) / (data_[:, 3, ...] + data_[:, 0, ...]))

                ndvi = torch.where((ndvi < -1) | (ndvi > 1), 0, ndvi)

            if self.norm is not None:
                data = {
                    s: (d - self.norm[s][0][None, :, None, None])
                       / self.norm[s][1][None, :, None, None]
                    for s, d in data.items()
                }

            # concat with NDVI
            if self.add_ndvi:
                data = {'S2': torch.cat([data['S2'], ndvi[:, None, ...]], axis=1)}

            if not self.for_inference:
                target = np.load(
                    os.path.join(
                        self.folder, f"ANNOTATIONS", f"TARGET_{id_patch}"  # f"TARGET_{id_patch}.npy"
                    )
                )
                target = torch.from_numpy(target.astype(int))

                if self.class_mapping is not None:
                    target = self.class_mapping(target)

                if self.cache:
                    if self.mem16:
                        self.memory[item] = [{k: v.half() for k, v in data.items()}, target]
                    else:
                        self.memory[item] = [data, target]

        else:
            data, target = self.memory[item]
            if self.mem16:
                data = {k: v.float() for k, v in data.items()}

        # Retrieve date sequences
        if not self.cache or id_patch not in self.memory_dates.keys():

            dates = {
                'S2': torch.from_numpy(self.get_dates_absolute(id_patch) if self.use_doy else
                                       self.get_dates_relative(id_patch))
            }

            if self.use_abs_rel_enc:
                dates2 = {
                    'S2': torch.from_numpy(self.get_dates_absolute(id_patch) if not self.use_doy else
                                           self.get_dates_relative(id_patch))
                }

            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = {'S2': data['S2'][self.mono_date].unsqueeze(0)}
                dates = {'S2': dates['S2'][self.mono_date]}
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = {
                    'S2': int((dates['S2'] - mono_delta).abs().argmin())
                }
                data = {'S2': data['S2'][mono_date['S2']].unsqueeze(0)}
                dates = {'S2': dates['S2'][mono_date['S2']]}

                # TODO fix this fpr dates2 variable

        if self.mem16:
            data = {k: v.float() for k, v in data.items()}

        # we work only with S2 data
        data = data['S2']
        dates = dates['S2']

        if self.use_abs_rel_enc:
            dates2 = dates2['S2']

        assert data.shape[0] == dates.shape[
            0], f'Shape in time dimension does not match for data T={data.shape[0]} and ' \
                f'for dates T={dates.shape[0]}. Id of patch is {id_patch}'

        if self.for_inference:
            return data, dates

        if self.transform and self.set_type == 'train':
            data, target = self.transform(data, target)  # 4d tensor T x C x H x W, 2d tensor H x W

        # TEMPORAL DROPOUT
        if self.set_type == 'train' and self.temporal_dropout > 0.:
            # remove acquisition with probability of temporal_dropout
            probas = torch.rand(data.shape[0])
            drop = torch.where(probas > self.temporal_dropout)[0]
            data = data[drop]
            dates = dates[drop]
            if self.use_abs_rel_enc:
                dates2 = dates2[drop]

        if self.use_abs_rel_enc:
            if self.get_affine:
                return (data, torch.cat([dates[..., None], dates2[..., None]], axis=1)), target, torch.tensor(affine)
            else:
                return (data, torch.cat([dates[..., None], dates2[..., None]], axis=1)), target
        else:
            if self.get_affine:
                return (data, dates), target, torch.tensor(affine)
            else:
                return (data, dates), target

    def rasterize_target(self, item, export=False):
        id_patch = self.id_patches[item]
        _, target = self[item]
        # currently works only for 2d arrays
        r = unpatchify(id=id_patch, data=target.numpy(), metadata_path=os.path.join(self.folder,
                                                                                    "metadata.json"),
                       nodata=0, dtype='uint8', export=export)

        return r


def calc_cover_statistics(folder: str, labels: list = labels_super_short):
    """
    Auxiliary function to calculate per class statistics over `OK` patches
    Parameters
    ----------
    folder: str
        Absolute path to directory where is stored dataset containing patches
    labels: list
        List containing names of classes. Note that 0th class is expected to be background.
        Also note that order and length of list is used to determine class codes and number of classes therefore
        order class names based on their class codes used in ground-truth
    Returns
    -------

    """
    m = pd.read_json(os.path.join(folder, 'metadata.json'))
    m.index = m['ID_PATCH'].astype(int)
    m.sort_index(inplace=True)
    path = os.path.join(folder, 'ANNOTATIONS')
    stats = {f'{k}_Cover': [] for k in labels[1:]}

    for _, v in tqdm(m.iterrows(), total=m.shape[0], desc='Processing...'):
        if v.Status == 'REMOVED':
            for k in list(stats.keys()):
                stats[k].append(np.nan)
            continue

        t = np.load(os.path.join(path, f'TARGET_{str(v["ID_PATCH"])}'))
        for i, k in enumerate(list(stats.keys())):
            stats[k].append(np.count_nonzero(t == i + 1))

    for k in list(stats.keys()):
        m[k] = stats[k]

    m.to_json(os.path.join(folder, 'metadata_and_stats.json'), indet=4, orient='records')


def create_train_test_split(folder: str):
    """
    Auxiliary function to create train/val/test split
    in 14:3:3 ratio i.e. 70:15:15 %.
    Note that this function is tailored for crop types used in S2TSCZCrop dataset i.e. implementation is not general
    Constraints are:
        - adjacent patches must be in same set i.e. there cannot be two adjacent patches in different sets
        - similar even distribution of patches per set over tile
        - similar distribution of crop type classes per set
    Parameters
    ----------
    folder: str
        Absolute path to directory where is stored dataset
    Returns
    -------
    """
    if not os.path.isfile(os.path.join(folder, 'metadata_and_stats.json')):
        logging.info("CALCULATING COVER STATISTICS")
        calc_cover_statistics(folder, labels_super_short)

    m = pd.read_json(os.path.join(folder, 'metadata_and_stats.json'))
    m.index = m['ID_PATCH'].astype(int)
    m.sort_index(inplace=True)

    minority_l = ['Flax_Hemp_Cover', 'Hopyards_Cover', 'Sugar_beat_Cover', 'Permanent_fruit_Cover', 'Vineyards_Cover']

    majority_l = ['Background_Cover', 'Grassland_Cover', 'Winter_cereals_Cover']

    element = np.ones((3, 3))

    final_train_ids = []
    final_val_ids = []
    final_test_ids = []

    for e, t in enumerate(TILES):
        # flax & hemp has lowest number of occurrences (307 parcels) and is scattered all over the republic
        # so we cannot lost any of them and therefore we will treat it specially
        flax = m[(m['Flax_Hemp_Cover'] > 0.0) & (m['TILE'] == t)]
        minority = m[((m[minority_l[0]] > 0.) | (m[minority_l[1]] > 0.) | (m[minority_l[2]] > 0.) |
                      (m[minority_l[3]] > 0.) | (m[minority_l[4]] > 0.) |
                      ((m[majority_l[0]] < 0.2) & (m[majority_l[1]] < 0.3) & (m[majority_l[2]] < 0.3))) &
                     (m['TILE'] == t)]

        flax_ids = flax['ID_PATCH'].values
        minority_ids = minority['ID_PATCH'].values

        grid = np.zeros((82, 82), dtype=int)

        rows_flax = [get_row_col(i - (i // 6724 * 6724))[0] for i in flax_ids]
        cols_flax = [get_row_col(i - (i // 6724 * 6724))[1] for i in flax_ids]
        rows_minority = [get_row_col(i - (i // 6724 * 6724))[0] for i in minority_ids]
        cols_minority = [get_row_col(i - (i // 6724 * 6724))[1] for i in minority_ids]

        grid[rows_minority, cols_minority] = 1
        grid[0:-1:10] = 0
        grid[:, 0:-1:10] = 0
        grid[rows_flax, cols_flax] = 1

        # https://stackoverflow.com/questions/46737409/finding-connected-components-in-a-pixel-array
        labeled, ncomponents = label(grid, element)

        # features on borders should be all set to train set
        border_components = np.unique(np.concatenate([labeled[:, [0, 81]].flatten(), labeled[[0, 81]].flatten()]))
        border_components = [i for i in border_components if i != 0]

        other_components = [i for i in np.unique(labeled) if i not in border_components + [0]]

        other_components = np.random.permutation(other_components)

        border_components_sizes = [np.where(labeled == i)[0].shape[0] for i in border_components]
        other_components_sizes = [np.where(labeled == i)[0].shape[0] for i in other_components]

        total = np.sum(border_components_sizes + other_components_sizes)
        sums = {'train': np.sum(border_components_sizes) / total, 'val': 0., 'test': 0.}
        required = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        other_sorted = [i for _, i in sorted(zip(other_components_sizes, other_components), key=lambda x: x[0])]
        other_sizes_sorted = sorted(other_components_sizes)

        train_components = border_components
        val_components = []
        test_components = []

        for o, s_o in zip(other_sorted, other_sizes_sorted):
            w = [1 - s / r if 1 - s / r > 0. else 0. for r, s in zip(required.values(), sums.values())]
            weights = np.array(w) / np.sum(w)

            np.random.seed(42)
            choice = np.random.choice(3, 1, replace=False, p=weights)[0]

            if choice == 0:
                train_components.append(o)
                sums['train'] += s_o / total
            elif choice == 1:
                val_components.append(o)
                sums['val'] += s_o / total
            elif choice == 2:
                test_components.append(o)
                sums['test'] += s_o / total

        final_grid = np.zeros((82, 82), dtype=int)

        # 1 means train set
        final_grid = np.where(np.isin(labeled, train_components), 1, final_grid)
        # 2 means val set
        final_grid = np.where(np.isin(labeled, val_components), 2, final_grid)
        # 3 means test set
        final_grid = np.where(np.isin(labeled, test_components), 3, final_grid)

        train_ids = [(i * 82 + j) + e * 82 * 82 for i, j in
                     zip(np.where(final_grid == 1)[0], np.where(final_grid == 1)[1])]
        val_ids = [(i * 82 + j) + e * 82 * 82 for i, j in
                   zip(np.where(final_grid == 2)[0], np.where(final_grid == 2)[1])]
        test_ids = [(i * 82 + j) + e * 82 * 82 for i, j in
                    zip(np.where(final_grid == 3)[0], np.where(final_grid == 3)[1])]

        with open(os.path.join(folder, f'patches_distribution_{t}.npy'), 'wb') as f:
            np.save(f, final_grid)

        final_train_ids += train_ids
        final_val_ids += val_ids
        final_test_ids += test_ids

    m2 = pd.read_json(os.path.join(folder, 'metadata.json'), dtype={'ID_PATCH': 'int32',
                                                                    'ID_WITHIN_TILE': 'int32',
                                                                    'Background_Cover': 'float32',
                                                                    'time-series_length': 'int16',
                                                                    'crs': 'int16',
                                                                    'Fold': 'int8'})
    m2[final_train_ids, 'set'] = 'train'
    m2[final_val_ids, 'set'] = 'val'
    m2[final_test_ids, 'set'] = 'test'
    m[final_train_ids, 'set'] = 'train'
    m[final_val_ids, 'set'] = 'val'
    m[final_test_ids, 'set'] = 'test'

    m2.to_json(os.path.join(folder, 'metadata.json'), indent=4, orient='records')
    m.to_json(os.path.join(folder, 'metadata_and_stats.json'), indent=4, orient='records')


def compute_sample_weights(folder: str):
    """
    Auxiliary function to compute weights for every sample based on presence of minority class ...
    will be used for weighted random sampling with replacement
    Parameters
    ----------
    folder: str
        Absolute path to directory where is stored dataset
    Returns
    -------
    """
    stats = pd.read_json(os.path.join(folder, 'metadata_and_stats.json'))
    m = pd.read_json(os.path.join(folder, 'metadata.json'))

    m.index = m["ID_PATCH"].astype(int)
    m.sort_index(inplace=True)

    stats = stats[(stats['Status'] == 'OK') & (stats['set'] == 'train')]

    stats.index = stats["ID_PATCH"].astype(int)
    stats.sort_index(inplace=True)

    s = stats[[i for i in stats.columns if 'Cover' in i and i not in ['Nodata_Cover', 'Snow_Cloud_Cover']]]

    # TODO order of weights is corresponds to order of cover columns in stats dataframe
    weights = np.array([0, 1, 1, 0, 0, 0, 0, 5, 0, 14, 8, 4, 4, 0, 0])

    kk = (s * weights.astype(bool).astype(int)).astype(bool).astype(int) * weights

    kk['total'] = kk.sum(axis=1)

    kk.loc[kk.total == 0, 'total'] = 1

    m.loc[kk.index, 'weight'] = kk.total

    m.to_json(os.path.join(folder, 'metadata.json'), indent=4, orient='records')


def compute_norm_vals(folder: str):
    """
    Auxiliary function to generate mean and std over train dataset
    Parameters
    ----------
    folder: str
        Absolute path to directory where is stored dataset
    Returns
    -------

    """
    norm_vals = {}
    dt = S2TSCZCropDataset(folder=folder, norm=False, set_type='train', channels_like_pastis=False)

    means = []
    stds = []
    for i, b in enumerate(tqdm(dt, desc=f'Calculating mean and standard deviation')):
        data = b[0][0]  # T x C x H x W
        data = data.permute(1, 0, 2, 3).contiguous()  # C x T x H x W
        means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
        stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

    mean = np.stack(means).mean(axis=0).astype(float)
    std = np.stack(stds).mean(axis=0).astype(float)

    norm_vals["train"] = dict(mean=list(mean), std=list(std))

    with open(os.path.join(folder, "NORM_S2_patch.json"), "w") as file:
        file.write(json.dumps(norm_vals, indent=4))


if __name__ == '__main__':
    d = S2TSCZCropDataset(folder='/disk2/<username>/S2TSCZCrop', set_type='train', norm=False,
                          use_doy=False)
    out = d[0]  # (data, dates), target

    r1 = d.rasterize_target(0, export=False)
    r2 = d.rasterize_target(0, export=True)
    print('Done')
