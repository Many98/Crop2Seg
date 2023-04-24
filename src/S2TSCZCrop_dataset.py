import json
import os
from datetime import datetime
import logging

import pandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata

logging.getLogger().setLevel(logging.INFO)


class S2TSCZCrop_Dataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        norm=True,
        cache=False,
        mem16=False,
        folds=None,
        reference_date="2019-02-01",
        class_mapping=None,
        mono_date=None,
        from_date=None,
        to_date=None
    ):
        """
        TODO adjust it to handle also pretrain dataset ... difference should be only in loading TARGET
        Pytorch Dataset class to load samples from the S2TSCZCrop dataset for semantic segmentation.
        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number
              of days since a reference date
            - target is the semantic or instance target
        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
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
        """
        super(S2TSCZCrop_Dataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))

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
        self.meta_patch = pd.read_json(os.path.join(folder, "metadata.json"), orient='index')
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
            if not os.path.isfile(os.path.join(folder, "NORM_S2_patch.json")):
                logging.info("Normalization values for dataset not found."
                             "Generating normalization values for dataset")
                compute_norm_vals(self.folder)

            self.norm = {}

            with open(
                os.path.join(folder, "NORM_S2_patch.json"), "r"
            ) as file:
                normvals = json.loads(file.read())
            selected_folds = folds if folds is not None else range(1, 6)
            means = [normvals[f"Fold_{f}"]["mean"] for f in selected_folds]
            stds = [normvals[f"Fold_{f}"]["std"] for f in selected_folds]
            self.norm['S2'] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
            self.norm['S2'] = (
                torch.from_numpy(self.norm['S2'][0]).float(),
                torch.from_numpy(self.norm['S2'][1]).float(),
            )
        else:
            self.norm = None
        logging.info("Dataset ready.")

    def __len__(self):
        return self.len

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
                        f"S2_{id_patch}.npy",
                    )
                ).astype(np.float32)
            }  # T x C x H x W arrays
            data = {s: torch.from_numpy(a) for s, a in data.items()}

            if self.norm is not None:
                data = {
                    s: (d - self.norm[s][0][None, :, None, None])
                    / self.norm[s][1][None, :, None, None]
                    for s, d in data.items()
                }
            # TODO we want to do this bit different
            target = np.load(
                os.path.join(
                    self.folder, f"ANNOTATIONS", f"TARGET_{id_patch}.npy"
                )
            )
            target = torch.from_numpy(target[0].astype(int))

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
    Auxiliary function to generate mean and std over dataset
    Parameters
    ----------
    folder :

    Returns
    -------

    """
    norm_vals = {}
    for fold in range(1, 6):
        dt = S2TSCZCrop_Dataset(folder=folder, norm=False, folds=[fold])
        means = []
        stds = []
        for i, b in enumerate(dt):
            logging.info(f"{i}/{len(dt)}", end="\r")
            data = b[0][0]  # T x C x H x W
            data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
            means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
            stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals[f"Fold_{fold}"] = dict(mean=list(mean), std=list(std))

    with open(os.path.join(folder, "NORM_S2_patch.json"), "w") as file:
        file.write(json.dumps(norm_vals, indent=4))


if __name__ == '__main__':
    d = S2TSCZCrop_Dataset(folder='/disk2/<username>/test_dataset', norm=False)
