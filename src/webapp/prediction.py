# ############
# SCRIPT CONTAINING ALL RELEVANT FUNCTIONS FOR APPLYING NEURAL MODEL FOR PREDICTION
# ############

import json

import torch
import torch.utils.data as data
import numpy as np
from einops import rearrange
import argparse
import os

import geopandas as gpd
import pandas as pd
import rasterio

import streamlit as st


from src.learning.utils import get_model, recursive_todevice
from src.utils import pad_collate
from src.helpers.postprocess import prediction2raster, polygonize
from src.datasets.s2_ts_cz_crop import crop_cmap, labels_super_short, S2TSCZCropDataset

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="utae",
    type=str,
    help="Type of architecture to use. Can be one of: (utae/unet3d/timeunet/wtae)",
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

parser.add_argument("--add_linear", action='store_true',
                    help="Whether to add linear transform to positional encoder")
parser.add_argument("--max_temp",  default=None,
                    type=int,
                    help="Maximal length of time-series. Required only for unet_naive")
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


def get_config(year):
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))

    # ---------------------------
    config.batch_size = 1
    config.use_doy = False
    config.num_queries = 1
    config.use_abs_rel_enc = False
    config.add_ndvi = False
    config.num_classes = 15  # 15 for s2tsczcrop
    config.out_conv = [32, 15]
    config.model = 'timeunet'
    config.ref_date = f"{year - 1}-09-01"
    # ---------------------------

    assert config.num_classes == config.out_conv[
        -1], f'Number of classes {config.num_classes} does not match number of' \
             f' output channels {config.out_conv[-1]}'
    assert config.conv_type in ['2d', 'depthwise_separable'], f'Parameter `conv_type` must be one of ' \
                                                              f' [2d, depthwise_separable] but is {config.conv_type}'

    return config


def load_model(config, device):
    """
    loads model
    """

    state = torch.load('data/inference/timeunet_v1_base/model.pth.tar',
                       map_location=torch.device(device))
    state_dict = state["state_dict"]

    model = get_model(config)

    model.load_state_dict(state_dict)

    model = model.to(device)

    return model


def load_norm_values():
    """
    loads norm values
    """

    with open(
            os.path.join('data/inference/NORM_S2_patch.json'), "r"
    ) as file:
        normvals = json.loads(file.read())

    means = [normvals[f"train"]["mean"]]
    stds = [normvals[f"train"]["std"]]
    # here is fix for channels order to be like in PASTIS dataset
    channels_order = [2, 1, 0, 4, 5, 6, 3, 7, 8, 9]

    norm_values = {'mean': np.stack(means).mean(axis=0)[channels_order],
                   'std': np.stack(stds).mean(axis=0)[channels_order]}

    return norm_values


def generate_prediction(device, year, dataset_folder, affine):

    gdf_pred = None
    if os.path.isfile(f'src/webapp/cache/prediction/prediction_{year}.shp'):
        gdf_pred = gpd.read_file(f'src/webapp/cache/prediction/prediction_{year}.shp')
        if not gdf_pred[gdf_pred['name'] == os.path.split(dataset_folder)[-1]].empty:
            proba = rasterio.open(f'data/export/{os.path.split(dataset_folder)[-1]}.tif').read()[
                0]  # this is just top1 prediction
            st.write('Prediction already generated... Skipping')
            return proba

    config = get_config(year)

    config.device = device

    norm_values = load_norm_values()

    dt_args = dict(
        folder=dataset_folder,
        norm=True,
        norm_values=norm_values,
        reference_date=config.ref_date,
        channels_like_pastis=True,
        use_doy=config.use_doy,
        add_ndvi=config.add_ndvi,
        use_abs_rel_enc=config.use_abs_rel_enc,
        for_inference=True
    )

    dt_test = S2TSCZCropDataset(**dt_args, set_type='test', cache=config.cache)

    collate_fn = lambda x: pad_collate(x, pad_value=config.pad_value, max_size=config.max_temp)
    st.write('Reading time series...')
    test_loader = data.DataLoader(
        dt_test,
        batch_size=config.batch_size,  # TODO set to 1
        drop_last=False,
        collate_fn=collate_fn,
        # num_workers=0,
        # persistent_workers=True
    )
    if device == 'cuda':
        st.write('CUDA device detected.')
        st.write('Target device: GPU')
    else:
        st.write('Target device: CPU')

    st.write('Loading neural net...')

    model = load_model(config, device)

    model.eval()
    t1 = []
    proba = []
    st.write('Starting inference...')
    prediction_progress = st.progress(0, text=f'Generating raw prediction:  {0}%')
    done = 0
    for i, batch in enumerate(test_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)

            (x, dates) = batch

        with torch.no_grad():
            out = model(x, batch_positions=dates)
            pred_ = torch.nn.Softmax(dim=1)(out.detach().cpu())
            proba.append(pred_)
            t1.append(pred_.max(dim=1)[1][0].cpu().numpy())
            #proba.append(pred_[0][0].cpu().numpy())
            done += 1
            prediction_progress.progress(min((done / len(test_loader)), 1.0), f'Generating raw prediction:  {round(min((done / len(test_loader)), 1.0) * 100)}%')

    st.write("Post-processing...")
    t1 = np.stack(t1)
    proba = np.stack(proba)

    t1 = rearrange(t1, '(h w) ... h1 w1 -> ... (h h1) (w w1)', h1=128, w1=128, h=10, w=10)
    proba = rearrange(proba, '(h w) ... h1 w1 -> ... (h h1) (w w1)', h1=128, w1=128, h=10, w=10)[0]

    t1 = t1[..., :1098, :1098]
    proba = proba[..., :1098, :1098]

    st.write("Exporting raster...")
    try:
        prediction2raster(proba, 32633, affine, export=True, export_dir='data/export',
                          export_name=os.path.split(dataset_folder)[-1])
    except Exception as e:
        st.error(f'Error occured when exporting prediction raster ... Skipping')
        st.error(f'Error: {e}')

    st.write("Performing vectorization of raster layer...")
    gdf = polygonize(proba, affine, type_='hard')

    gdf.loc[:, 'name'] = os.path.split(dataset_folder)[-1]

    if gdf_pred is not None:
        gdf_pred = pd.concat([gdf_pred, gdf])
    else:
        gdf_pred = gdf

    gdf_pred.to_file(f'src/webapp/cache/prediction/prediction_{year}.shp')

    return proba

