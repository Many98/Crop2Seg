import json
import os
from datetime import datetime
import logging

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata

# ### small boiler plate to add src to sys path
import sys
from pathlib import Path

file = Path(__file__).resolve()
root = str(file).split('src')[0]
sys.path.append(root)
# --------------------------------------

from src.helpers.utils import unpatchify

logging.getLogger().setLevel(logging.INFO)


class S2TSCZCropDataset(tdata.Dataset):
    """
    Pytorch Dataset class to load samples from the S2TSCZCrop dataset for semantic segmentation of crop types
    from in time-series of Sentinel-2 tiles over Czech republic
    The Dataset yields ((data, dates), target) tuples, where:
        - data contains the image time series
        - dates contains the date sequence of the observations expressed in number
          of days since a reference date
        - target is the semantic or instance target
    TODO adjust it to handle also pretrain dataset ... difference should be only in loading TARGET
    """
    def __init__(
            self,
            folder,
            norm=True,
            norm_values_folder="",
            cache=False,
            mem16=False,
            folds=None,
            reference_date="2019-02-01",
            class_mapping=None,
            mono_date=None,
            from_date=None,
            to_date=None,
            channels_like_pastis=True,
            *args, **kwargs
    ):
        """
        Args:
            folder (str): Path to the dataset folder (directory)
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
                When Use `norm_values_folder` argument instead
            norm_values_folder (str or None): Path to folder (directory) where to look for NORM_S2_patch.json
                                     file storing normalization values. Used only of norm is set to true
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
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            from_date (int or str, optional):

            to_date: (int or str, optional):  TODO
            channels_like_pastis: bool
                Whether to rearrange channels to be in same order like in PASTIS dataset.
                It should be used if one want to fine-tune using UTAE's original pretrained weights.
                Particularly PASTIS has channels ordered like this:
                    [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
                while S2TSCZCrop has channels ordered as found in .SAFE format:
                    [B04, B03, B02, B08, B05, B06, B07, B8A, B11, B12]
        """
        super().__init__()
        self.folder = folder
        self.norm = norm
        self.norm_values_folder = norm_values_folder
        self.reference_date = datetime(*map(int, reference_date.split("-")))

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

        # Get metadata  TODO what to do with this
        logging.info("Reading patch metadata . . .")
        self.meta_patch = pd.read_json(os.path.join(folder, "metadata.json"), orient='records',
                                       dtype={'ID_PATCH': 'int32',
                                              'ID_WITHIN_TILE': 'int32',
                                              'Background_Cover': 'float32',
                                              'time-series_length': 'int8',
                                              'crs': 'int16',
                                              'Fold': 'int8'})
        self.meta_patch = self.meta_patch[self.meta_patch['Status'] == 'OK']
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {'S2': None}
        self.date_range = np.array(range(-200, 600))

        dates = self.meta_patch[f"dates-S2"]
        date_table = pd.DataFrame(
            index=self.meta_patch.index, columns=self.date_range, dtype=int
        )
        for pid, date_seq in dates.iteritems():
            d = pd.DataFrame().from_dict(date_seq, orient="index")
            d = d[0].apply(
                lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                ).days
            )
            date_table.loc[pid, d.values] = 1
        date_table = date_table.fillna(0)
        self.date_tables['S2'] = {
            index: np.array(list(d.values()))
            for index, d in date_table.to_dict(orient="index").items()
        }

        logging.info("Done.")

        # Select Fold samples
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        # Get normalisation values
        if norm:
            if not os.path.isfile(os.path.join(norm_values_folder, "NORM_S2_patch.json")):
                raise Exception(f"Norm parameter set to True but normalization values json file for dataset was "
                                f"not found in specified directory {norm_values_folder} .")

            self.norm = {}

            with open(
                    os.path.join(norm_values_folder, "NORM_S2_patch.json"), "r"
            ) as file:
                normvals = json.loads(file.read())
            selected_folds = folds if folds is not None else range(1, 6)
            means = [normvals[f"Fold_{f}"]["mean"] for f in selected_folds]
            stds = [normvals[f"Fold_{f}"]["std"] for f in selected_folds]
            self.norm['S2'] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
            self.norm['S2'] = (
                torch.from_numpy(self.norm['S2'][0]).float()[self.channels_order],
                torch.from_numpy(self.norm['S2'][1]).float()[self.channels_order],
            )
        else:
            self.norm = None
        logging.info("Dataset ready.")

    def __len__(self):
        return self.meta_patch.shape[0]

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

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
            data = {s: torch.from_numpy(a)[self.channels_order] for s, a in data.items()}

            if self.norm is not None:
                data = {
                    s: (d - self.norm[s][0][None, :, None, None])
                       / self.norm[s][1][None, :, None, None]
                    for s, d in data.items()
                }
            # TODO we want to do this bit different
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
                'S2': torch.from_numpy(self.get_dates(id_patch, 'S2'))
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

        if self.mem16:
            data = {k: v.float() for k, v in data.items()}

        # we work only with S2 data
        data = data['S2']
        dates = dates['S2']

        return (data, dates), target

    def rasterize_target(self, item, export=False):
        id_patch = self.id_patches[item]
        _, target = self[item]
        # currently works only for 2d arrays
        r = unpatchify(id=id_patch, data=target.numpy(), metadata_path=os.path.join(self.folder,
                                                                                    "metadata.json"),
                       nodata=0, dtype='uint8', export=export)

        return r


def prepare_dates(date_dict, reference_date):
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
                datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                - reference_date
        ).days
    )
    return d.values


def compute_norm_vals(folder):
    """
    Auxiliary function to generate mean and std over dataset (per fold)
    Parameters
    ----------
    folder :

    Returns
    -------

    """
    norm_vals = {}
    dt = S2TSCZCropDataset(folder=folder, norm=False, channels_like_pastis=False)
    meta_patch_copy = dt.meta_patch.copy(deep=True)

    for fold in range(1, 6):
        dt.meta_patch = meta_patch_copy[meta_patch_copy['Fold'] == fold]
        dt.id_patches = dt.meta_patch.index

        means = []
        stds = []
        for i, b in enumerate(tqdm(dt, desc=f'Processing fold: {fold}')):
            data = b[0][0]  # T x C x H x W
            data = data.permute(1, 0, 2, 3).contiguous()  # C x T x H x W
            means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
            stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals[f"Fold_{fold}"] = dict(mean=list(mean), std=list(std))

    with open(os.path.join(folder, "NORM_S2_patch.json"), "w") as file:
        file.write(json.dumps(norm_vals, indent=4))


if __name__ == '__main__':
    d = S2TSCZCropDataset(folder='/disk2/<username>/S2TSCZCrop', norm=True)
    out = d[0]  # (data, dates), target

    r1 = d.rasterize_target(0, export=False)
    r2 = d.rasterize_target(0, export=True)
    print('Done')
