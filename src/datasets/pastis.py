import json
import os
from datetime import datetime
import logging
import matplotlib
from matplotlib.colors import ListedColormap

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata

logging.getLogger().setLevel(logging.INFO)

labels = ['Background 0', 'Meadow 1', 'Soft winter wheat 2', 'Corn 3',
          'Winter barley 4', 'Winter rapeseed 5', 'Spring barley 6', 'Sunflower 7', 'Grapevine 8', 'Beet 9',
          'Winter triticale 10', 'Winter durum wheat 11', 'Fruits, vegetables, flowers 12', 'Potatoes 13',
          'Leguminous fodder 14', 'Soybeans 15', 'Orchard 16', 'Mixed cereals 17', 'Sorghum 18', 'Void label 19']

labels_super_short = ['Background', 'Meadow', 'Winter wheat (s.)', 'Corn',
                      'Winter barley', 'Winter rapeseed', 'Spring barley', 'Sunflower', 'Grapevine', 'Beet',
                      'Winter triticale', 'Winter wheat (d.)', 'Fruit/vegetable/flower', 'Potatoes',
                      'Leguminous fodder', 'Soybeans', 'Orchard', 'Mixed cereals', 'Sorghum', 'Void']


def crop_cmap():
    """
    Auxiliary function to return dictionary with color map used for visualization
    of classes in S2TSCZCrop dataset
    """
    cm = matplotlib.cm.get_cmap('tab20')
    def_colors = cm.colors
    colors = [[0, 0, 0]] + [list(def_colors[i]) for i in range(1, 19)] + [[1, 1, 1]]

    return {k: v + [1] for k, v in enumerate(colors)}


class PASTISDataset(tdata.Dataset):
    def __init__(
            self,
            folder,
            norm=True,
            target="semantic",
            cache=False,
            mem16=False,
            folds=None,
            norm_folds=None,  # added 24.7.23
            norm_values=None,  # added 29.9.23
            reference_date="2018-09-01",
            class_mapping=None,
            mono_date=None,
            sats=["S2"],
            use_doy=False,
            use_abs_rel_enc=False,
            transform=None,
            add_ndvi=False,
            set_type='train',
            temporal_dropout=0.0,
            *args, **kwargs
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
        panoptic segmentation.
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
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
                * If 'instance' the target tensor is the concatenation of several
                  signals, necessary to train the Parcel-as-Points module:
                    - the centerness heatmap,
                    - the instance ids,
                    - the voronoi partitioning of the patch with regards to the parcels'
                      centers,
                    - the (height, width) size of each parcel
                    - the semantic label of each parcel
                    - the semantic label of each pixel
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
            sats (list): defines the satellites to use (only Sentinel-2 is available
                in v1.0)
            use_doy: (bool) Whether to use absolute positions for time-series i.e. day of years instead
                           of difference between date and `self.reference_date`
            use_abs_rel_enc: (bool)
                Whether to use both relative day positions and absolute (doy) positions.
                If True parameter `use_doy` will be unused
            transform: (torchvision.transform)
                Transformation which will be applied to input tensor and mask
            add_ndvi: (bool)
                Whether to add NDVI channel at the end of spectral channels
        """
        super(PASTISDataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = None
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.target = target
        self.sats = sats
        self.set_type = set_type
        self.temporal_dropout = temporal_dropout

        self.norm_values = norm_values

        self.use_abs_rel_enc = use_abs_rel_enc
        self.use_doy = False if use_abs_rel_enc else use_doy

        self.transform = transform
        self.add_ndvi = add_ndvi

        # Get metadata
        logging.info("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

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
        return self.len

    def get_dates_relative(self, id, sat):
        """
        Method returns array representing difference between date and `self.reference_date` i.e.
        position of element within time-series is relative to  `self.reference_date`

        Note that this work only for S2 tiles
        """
        d = pd.DataFrame().from_dict(self.meta_patch.loc[id, 'dates-S2'], orient="index")
        d = d[0].apply(
            lambda x: (
                    datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                    - self.reference_date
            ).days
        )

        return d.values

    def get_dates_absolute(self, id, sat):
        """
        Method returns array representing day of year for a date i.e.
        position of element within time-series is absolute to with respect to actual year.
        Using only 365 days long years

        Note that this work only for S2 tiles
        """
        d = pd.DataFrame().from_dict(self.meta_patch.loc[id, 'dates-S2'], orient="index")
        d = pd.to_datetime(d[0].astype(str), format='%Y%m%d').dt.dayofyear

        return d.values

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {
                satellite: np.load(
                    os.path.join(
                        self.folder,
                        "DATA_{}".format(satellite),
                        "{}_{}.npy".format(satellite, id_patch),
                    )
                ).astype(np.float32)
                for satellite in self.sats
            }  # T x C x H x W arrays
            data = {s: torch.from_numpy(a) for s, a in data.items()}

            if self.add_ndvi:
                # NDVI B08(NIR) + B04(Red) / B08(NIR) - B04(Red)
                # NDVI red edge B08(NIR) + B06(Red edge) / B08(NIR) - B06(Red edge)
                #  we do not normalize it (its by definition between -1, 1)
                #  no data gets 0 as value

                data_ = data['S2']

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

            if self.target == "semantic":
                target = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )
                target = torch.from_numpy(target[0].astype(int))

                if self.class_mapping is not None:
                    target = self.class_mapping(target)

            elif self.target == "instance":
                heatmap = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "HEATMAP_{}.npy".format(id_patch),
                    )
                )

                instance_ids = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "INSTANCES_{}.npy".format(id_patch),
                    )
                )
                pixel_to_object_mapping = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "ZONES_{}.npy".format(id_patch),
                    )
                )

                pixel_semantic_annotation = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )

                if self.class_mapping is not None:
                    pixel_semantic_annotation = self.class_mapping(
                        pixel_semantic_annotation[0]
                    )
                else:
                    pixel_semantic_annotation = pixel_semantic_annotation[0]

                size = np.zeros((*instance_ids.shape, 2))
                object_semantic_annotation = np.zeros(instance_ids.shape)
                for instance_id in np.unique(instance_ids):
                    if instance_id != 0:
                        h = (instance_ids == instance_id).any(axis=-1).sum()
                        w = (instance_ids == instance_id).any(axis=-2).sum()
                        size[pixel_to_object_mapping == instance_id] = (h, w)
                        object_semantic_annotation[
                            pixel_to_object_mapping == instance_id
                            ] = pixel_semantic_annotation[instance_ids == instance_id][0]

                target = torch.from_numpy(
                    np.concatenate(
                        [
                            heatmap[:, :, None],  # 0
                            instance_ids[:, :, None],  # 1
                            pixel_to_object_mapping[:, :, None],  # 2
                            size,  # 3-4
                            object_semantic_annotation[:, :, None],  # 5
                            pixel_semantic_annotation[:, :, None],  # 6
                        ],
                        axis=-1,
                    )
                ).float()

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
                s: torch.from_numpy(self.get_dates_absolute(id_patch, s) if self.use_doy else
                                    self.get_dates_relative(id_patch, s)) for s in self.sats
            }

            if self.use_abs_rel_enc:
                dates2 = {
                    'S2': torch.from_numpy(self.get_dates_absolute(id_patch, 'S2') if not self.use_doy else
                                           self.get_dates_relative(id_patch, 'S2'))
                }

            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = {s: data[s][self.mono_date].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][self.mono_date] for s in self.sats}
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = {
                    s: int((dates[s] - mono_delta).abs().argmin()) for s in self.sats
                }
                data = {s: data[s][mono_date[s]].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][mono_date[s]] for s in self.sats}

        if self.mem16:
            data = {k: v.float() for k, v in data.items()}

        # we work only with S2 data
        data = data['S2']
        dates = dates['S2']

        if self.use_abs_rel_enc:
            dates2 = dates2['S2']

        if self.transform and self.set_type == 'train':
            data, target = self.transform(data, target)  # 4d tensor T x C x H x W, 2d tensor H x W

        if self.set_type == 'train' and self.temporal_dropout > 0.:
            # remove acquisition with probability of temporal_dropout
            probas = torch.rand(data.shape[0])
            drop = torch.where(probas > self.temporal_dropout)[0]
            data = data[drop]
            dates = dates[drop]
            if self.use_abs_rel_enc:
                dates2 = dates2[drop]

        if self.use_abs_rel_enc:
            return (data, torch.cat([dates[..., None], dates2[..., None]], axis=1)), target
        else:
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


def compute_norm_vals(folder, sat):
    norm_vals = {}
    for fold in range(1, 6):
        dt = PASTISDataset(folder=folder, norm=False, folds=[fold], sats=[sat])
        means = []
        stds = []
        for i, b in enumerate(dt):
            logging.info("{}/{}".format(i, len(dt)), end="\r")
            data = b[0][0][sat]  # T x C x H x W
            data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
            means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
            stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals["Fold_{}".format(fold)] = dict(mean=list(mean), std=list(std))

    with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
        file.write(json.dumps(norm_vals, indent=4))


if __name__ == '__main__':
    d = PASTISDataset('/disk2/<username>/PASTIS', use_doy=True)
    print(d[0])
    print('p')
