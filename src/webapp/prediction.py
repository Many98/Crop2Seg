# ############
# SCRIPT CONTAINING ALL RELEVANT FUNCTIONS FOR APPLYING NEURAL MODEL FOR PREDICTION
# ############

import json

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import os

import matplotlib.pyplot as plt

from src.learning.utils import get_model
from src.visualization.visualize import show, plot_rgb, plot_lulc, plot_proba_mask
from src.datasets.s2_ts_cz_crop import crop_cmap, unpatchify, labels, labels_short, \
    labels_super_short, labels_super_short_2

user_device = 'cpu'

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="utae",
    type=str,
    help="Type of architecture to use. Can be one of: (utae/unet3d/timeunet)",
)
## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 15]")
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int, help="Dimension to which map value vectors before temporal"
                                                             "encoding.")
parser.add_argument("--d_k", default=4, type=int, help="Dimension of learnable query vector.")
parser.add_argument("--input_dim", default=10, type=int, help="Number of input spectral channels")
parser.add_argument("--num_queries", default=1, type=int, help="Number of learnable query vectors. This vectors are"
                                                               "averaged.")
parser.add_argument("--temporal_dropout", default=0., type=float,
                    help="Probability of removing acquisition from time-series")
parser.add_argument("--pretrain", default=False, type=bool, help="Whether to use pretrining dataset")

parser.add_argument(
    "--dataset",
    default="s2tsczcrop",
    type=str,
    help="Type of dataset to use. Can be one of: (s2tsczcrop/pastis)",
)

# Set-up parameters
parser.add_argument(
    "--test",
    action='store_true',
    help="Whether to perform test run (inference)"
         "Weights stored in `--weight_folder` directory  will be used",
)
parser.add_argument(
    "--test_region",
    default='all',
    help="Experimental setting. Can be one of ['all', 'boundary', 'interior']",
)
parser.add_argument(
    "--finetune",
    action='store_true',
    help="Whether to perform finetuning instead of training from scratch."
         "Weights stored in `--weight_folder` directory  will be used",
)
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where is stored dataset.",
)
parser.add_argument(
    "--norm_values_folder",
    default="",
    type=str,
    help="Path to the folder where to look for NORM_S2_patch.json file storing normalization values",
)
parser.add_argument(
    "--weight_folder",
    default=None,
    type=str,
    help="Path to folder containing the network weights in model.pth.tar file and model configuration file in conf.json."
         "If you want to resume training then this folder should also have trainlog.json file.",
)
parser.add_argument(
    "--res_dir",
    default="",
    help="Path to the folder where the results should be stored",
)
# parser.add_argument(
#    "--num_workers", default=8, type=int, help="Number of data loading workers"
# )
parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument(
    "--cache",
    dest="cache",
    action="store_true",
    help="If specified, the whole dataset is kept in RAM",
)
# Training parameters
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--mono_date", default=None, type=str, help="Whether to perform segmentation using only one"
                                                                " element of time-series. Use integer or string in"
                                                                " form (YYYY-MM-DD) ")
parser.add_argument("--ref_date", default="2018-09-01", type=str, help="Reference date (YYYY-MM-DD) used in relative"
                                                                       " positional"
                                                                       " encoding scheme i.e. dates are encoded"
                                                                       " as difference between actual date and reference"
                                                                       " date. If you want to use absolute encoding"
                                                                       "using day of years use `--use_doy` flag")
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Specify fold. (between 1 and 5) Note that this argument is used only as legacy argument \n"
         "and is used only for accessing correct normalization values e.g. if using PASTIS trained"
         "network for fine-tuning",
)
parser.add_argument("--num_classes", default=15, type=int, help="Number of classes used in segmentation task")
parser.add_argument("--ignore_index", default=-1, type=int, help="Index of class to be ignored")
parser.add_argument("--pad_value", default=0, type=float, help="Padding value for time-series")
parser.add_argument("--padding_mode", default="reflect", type=str, help="Type of padding")
parser.add_argument("--conv_type", default="2d", type=str, help="Type of convolutional layer. Must be one of '2d' or"
                                                                " 'depthwise_separable'")
parser.add_argument("--use_mbconv", action='store_true', help="Whether to use MBConv module instead of classical "
                                                              " convolutional layers")
parser.add_argument("--add_squeeze", action='store_true', help="Whether to add squeeze & excitation module")
parser.add_argument("--use_doy", action='store_true', help="Whether to use absolute positional encoding (day of year)"
                                                           " instead of relative encoding w.r.t. reference date")
parser.add_argument("--add_ndvi", action='store_true', help="Whether to add NDVI channel at the end")
parser.add_argument("--use_abs_rel_enc", action='store_true',
                    help="Whether to use both date representations: Relative and"
                         "absolute (DOY)")

parser.add_argument(
    "--val_every",
    default=1,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=0,
    type=int,
    help="Do validation only after that many epochs.",
)

list_args = ["encoder_widths", "decoder_widths", "out_conv"]
parser.set_defaults(cache=False)


def load_model(path, device):
    """
    loads model
    """

    # TODO download model weights from zenodo
    #   prepare config for model
    state = torch.load('',
                       map_location=torch.device(device))
    state_dict = state["state_dict"]

    model = get_model(config)

    model.load_state_dict(state_dict)

    model = model.to(device)

    return model


def load_norm_values(norm_folder):
    """
    loads norm values
    """

    # TODO download norm vals or it should be within repo
    with open(
            os.path.join(norm_folder), "r"
    ) as file:
        normvals = json.loads(file.read())

    means = [normvals[f"train"]["mean"]]
    stds = [normvals[f"train"]["std"]]
    # here is fix for channels order to be like in PASTIS dataset
    channels_order = [2, 1, 0, 4, 5, 6, 3, 7, 8, 9]

    norm_values = {'mean': np.stack(means).mean(axis=0)[channels_order],
                   'std': np.stack(stds).mean(axis=0)[channels_order]}

    return norm_values


def get_dates_relative(times, ref_date):
    """
    Method returns array representing difference between date and `self.reference_date` i.e.
    position of element within time-series is relative to  `self.reference_date`
    """
    ref_date = datetime(*map(int, ref_date.split("-")))
    d = pd.DataFrame().from_dict(times, orient="index")
    d = d[0].apply(
        lambda x: (datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])) - ref_date
        ).days
    )

    return d.values


def get_dates_absolute(times):
    """
    Method returns array representing day of year for a date i.e.
    position of element within time-series is absolute to with respect to actual year.
    Using only 365 days long years
    """
    d = pd.DataFrame().from_dict(times, orient="index")
    d = pd.to_datetime(d[0].astype(str), format='%Y%m%d').dt.dayofyear

    return d.values


def load_preprocess_data(data, dates, norm_values):
    pass


if __name__ == '__main__':
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))

    # ---------------------------
    config.batch_size = 1
    # config.out_conv = [32, 20]  #  here is num of out classes as in pretrained net
    # config.num_classes = 16  #  added boundary class as 15th class
    # config.out_conv = [32, 16]  #  added boundary class as 15th class
    # config.ignore_index = 14  #  added boundary class as 15th class
    config.use_doy = False
    config.test_region = 'all'  # 'boundary' | 'interior'
    config.num_queries = 1
    config.use_abs_rel_enc = False
    config.add_ndvi = False
    config.temporal_dropout = 0.0
    config.num_classes = 15  # 15 for s2tsczcrop
    config.out_conv = [32, 15]
    config.model = 'wtae'
    # ---------------------------

    assert config.num_classes == config.out_conv[
        -1], f'Number of classes {config.num_classes} does not match number of' \
             f' output channels {config.out_conv[-1]}'
    assert config.conv_type in ['2d', 'depthwise_separable'], f'Parameter `conv_type` must be one of ' \
                                                              f' [2d, depthwise_separable] but is {config.conv_type}'

    model = load_model('',
                       device=user_device)

    norm_values = load_norm_values(
        norm_folder='')

    data = np.load('').astype(np.float32)
    channels_order = [2, 1, 0, 4, 5, 6, 3, 7, 8, 9]
    data = torch.from_numpy(data)[:, channels_order, ...]

    # TODO maybe add ndvi

    data = (data - norm_values['mean'][None, :, None, None]) / norm_values['std'][None, :, None, None]

    with open('', 'r') as f:
        dates = json.load(f)

    dates = torch.from_numpy(get_dates_absolute(dates) if config.use_doy else
                             get_dates_relative(dates, config.ref_date))

    if config.use_abs_rel_enc:
        dates2 = torch.from_numpy(get_dates_absolute(dates) if not config.use_doy else
                                  get_dates_relative(dates, config.ref_date))

    data = data[None, ...].to(user_device)
    dates = dates[None, ...].to(user_device)

    model.eval()
    with torch.no_grad():
        out = model(data.float(), batch_positions=dates.float())

    pred_ = torch.nn.Softmax(dim=1)(out.detach().cpu()).max(dim=1)
    pred = pred_[1][0].cpu().numpy()
    proba = pred_[0][0].cpu().numpy()

    plot_lulc(pred, labels_super_short, crop_cmap())
    plt.title('Prediction')

    plot_proba_mask(proba)
    plt.title('Prediction confidence map')

    plot_rgb(data[0, 10, [2, 1, 0], ...].cpu().numpy())
    plt.title(f'Satellite image ({"unknown date"})')

    plt.show()

    print('done')

