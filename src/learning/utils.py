import json
import logging
import os
import pickle as pkl
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchnet as tnt


# ### small boiler plate to add src to sys path
import sys
from pathlib import Path

file = Path(__file__).resolve()
root = str(file).split('src')[0]
sys.path.append(root)
# --------------------------------------

from src.utils import get_ntrainparams
from src.learning.metrics import confusion_matrix_analysis
from src.learning.miou import IoU
from src.helpers.postprocess import homogenize

from src.backbones.utae import UTAE  # ours slightly adjusted implementation of utae
from src.backbones.unet3d import UNet3D
from src.backbones.timeunet import TimeUNet_v1, TimeUNet_v2
from src.backbones.convgru import ConvGRU_Seg
from src.backbones.convlstm import ConvLSTM_Seg
from src.backbones.recunet import RecUNet
#from src.backbones.fpn import FPNConvLSTM
from src.backbones.wtae import WTAE
#from src.backbones.etae import ETAE
from src.backbones.unet import Unet_naive

#from src.backbones.utae_original import UTAE

from einops import rearrange

from src.learning.focal_loss import FocalCELoss

from src.global_vars import AGRI_PATH_DATASET


# from src.backbones.utae_original import UTAE  # original implementation


def get_model(config):
    if config.model == "utae":
        model = UTAE(
            input_dim=config.input_dim,  # number of input channels
            encoder_widths=config.encoder_widths,
            decoder_widths=config.decoder_widths,
            out_conv=config.out_conv,
            str_conv_k=config.str_conv_k,
            str_conv_s=config.str_conv_s,
            str_conv_p=config.str_conv_p,
            agg_mode=config.agg_mode,
            encoder_norm=config.encoder_norm,
            n_head=config.n_head,
            d_model=config.d_model,
            d_k=config.d_k,
            encoder=False,
            return_maps=False,
            pad_value=config.pad_value,
            padding_mode=config.padding_mode,
            # ------------------------------
            # From here starts added arguments
            conv_type=config.conv_type,
            use_mbconv=config.use_mbconv,
            add_squeeze_excit=config.add_squeeze,
            use_abs_rel_enc=config.use_abs_rel_enc,
            num_queries=config.num_queries,
            use_doy=config.use_doy,
            add_linear=config.add_linear,
            add_boundary_loss=config.add_boundary_loss
        )
    elif config.model == "wtae":
        model = WTAE(
            input_dim=config.input_dim,  # number of input channels
            encoder_widths=config.encoder_widths,
            decoder_widths=config.decoder_widths,
            out_conv=config.out_conv,
            str_conv_k=config.str_conv_k,
            str_conv_s=config.str_conv_s,
            str_conv_p=config.str_conv_p,
            agg_mode=config.agg_mode,
            encoder_norm=config.encoder_norm,
            n_head=config.n_head,
            d_model=config.d_model,
            d_k=config.d_k,
            encoder=False,
            return_maps=False,
            pad_value=config.pad_value,
            padding_mode=config.padding_mode,
            # ------------------------------
            # From here starts added arguments
            conv_type=config.conv_type,
            use_mbconv=config.use_mbconv,
            add_squeeze_excit=config.add_squeeze,
            use_abs_rel_enc=config.use_abs_rel_enc,
            num_queries=config.num_queries,
            use_doy=config.use_doy,
            add_linear=config.add_linear,
            add_boundary_loss=config.add_boundary_loss
        )
    elif config.model == "timeunet":
        model = TimeUNet_v1(
            input_dim=config.input_dim,  # number of input channels
            encoder_widths=config.encoder_widths,
            decoder_widths=config.decoder_widths,
            out_conv=config.out_conv,
            str_conv_k=config.str_conv_k,
            str_conv_s=config.str_conv_s,
            str_conv_p=config.str_conv_p,
            agg_mode=config.agg_mode,
            encoder_norm=config.encoder_norm,
            n_head=config.n_head,
            d_model=config.d_model,
            d_k=config.d_k,
            encoder=False,
            return_maps=False,
            pad_value=config.pad_value,
            padding_mode=config.padding_mode,
            # ------------------------------
            # From here starts added arguments
            conv_type=config.conv_type,
            use_mbconv=config.use_mbconv,
            add_squeeze_excit=config.add_squeeze,
            use_abs_rel_enc=config.use_abs_rel_enc,
            num_queries=config.num_queries,
            use_doy=config.use_doy,
            add_linear=config.add_linear
        )
    elif config.model == "unet_naive":
        model = Unet_naive(
            input_dim=config.input_dim,  # number of input channels
            temporal_length=config.max_temp,
            #encoder_widths=config.encoder_widths,
            #decoder_widths=config.decoder_widths,
            #out_conv=config.out_conv,
            str_conv_k=config.str_conv_k,
            str_conv_s=config.str_conv_s,
            str_conv_p=config.str_conv_p,
            agg_mode=config.agg_mode,
            encoder=False,
            return_maps=False,
            pad_value=config.pad_value,
            padding_mode=config.padding_mode,
            # ------------------------------
            # From here starts added arguments
            conv_type=config.conv_type,
            use_mbconv=config.use_mbconv,
            add_squeeze_excit=config.add_squeeze
        )
    elif config.model == "unet3d":
        model = UNet3D(
            in_channel=config.input_dim, n_classes=config.num_classes, pad_value=config.pad_value
        )
    elif config.model == "convlstm":
        model = ConvLSTM_Seg(
            num_classes=config.num_classes,
            input_size=(128, 128),
            input_dim=config.input_dim,
            kernel_size=(3, 3),
            hidden_dim=160,
        )
    elif config.model == "convgru":
        model = ConvGRU_Seg(
            num_classes=config.num_classes,
            input_size=(128, 128),
            input_dim=config.input_dim,
            kernel_size=(3, 3),
            hidden_dim=180,
        )
    elif config.model == "uconvlstm":
        model = RecUNet(
            input_dim=config.input_dim,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 20],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            temporal="lstm",
            input_size=128,
            encoder_norm="group",
            hidden_dim=64,
            encoder=False,
            padding_mode="zeros",
            pad_value=0,
        )
    return model


def get_dilated(target: torch.tensor, n_classes: int, device: str, connectivity: int = 4):
    """
    Helper function to get dilated ground truth. Used to determine boundaries of semantic classes
    Parameters
    ----------
    target: torch.tensor
    n_classes: int
        Number of classes
    device: str
        Device can be `cpu`, `cuda`, or `cuda:1` etc.
    connectivity: int
        Connectivity used for dilatation operation. Can be 8 or 4
    """
    if connectivity == 8:
        weights = torch.ones((n_classes, 1, 3, 3), device=device, requires_grad=False)
    else:
        weights = torch.tensor([[0., 1., 0.],
                                [1., 1., 1.],
                                [0., 1., 0.]], device=device, requires_grad=False).view(1, 1, 3, 3).repeat(
            n_classes, 1, 1, 1)

    # y is of shape B x H x W
    one_hot_target = F.one_hot(target.long(), num_classes=n_classes).permute(0, 3, 1, 2)

    return F.conv2d(one_hot_target.float(), weights, groups=n_classes, padding=(1, 1)).bool().long()


def iterate(
        model, data_loader, criterion, config, optimizer=None, scheduler=None, mode="train", device=None,
        test_region='all', **kwargs
):
    """
    helper function implementing training/validation/testing loop
    Parameters
    ----------
    model
    data_loader
    criterion
    config
    optimizer
    scheduler
    mode
    device
    test_region: str
        New parameter to test performance on `all` pixels or `boundary` pixels or `interior` pixels
        Currently works by reclassification to ignore class
        Note that code expects that there is set ignore index
    """
    #import matplotlib.pyplot as plt
    #from src.datasets.pastis import crop_cmap, labels_super_short
    #from src.visualization.visualize import plot_lulc

    loss_meter = tnt.meter.AverageValueMeter()
    iou_meter = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device=config.device,
    )
    iou_meter_top2 = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device=config.device,
    )

    if config.add_boundary_loss:
        iou_meter_boundary = IoU(
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
            cm_device=config.device,
        )

        criterion_b = FocalCELoss(gamma=2.0)
    #catcher = {'pred': [], 'conf': [], 'label': []}
    #catcher = {'hash': [], 'area': [], 'raster_val': [], 'Legenda': []}
    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)

        if config.get_affine:
            (x, dates), y, affine = batch
        else:
            (x, dates), y = batch
        y = y.long()

        if config.add_boundary_loss:
            dilated = get_dilated(y, config.num_classes, x.device, 4)  # shape is B x NUM_CLASSES x H x W
            y_b = torch.where(dilated.sum(1) > 1, 1, 0)  # 0 is background; 1 is boundary
        # -----------HERE ARE EXPERIMENTAL CHANGES ---------------------
        # boundary is removed from training but not from performance evaluation
        '''
        dilated = get_dilated(y, config.num_classes, x.device, 4)  # shape is B x NUM_CLASSES x H x W
        ignore_label = [i for i in range(config.num_classes)][config.ignore_index]
        # reclassify boundary to ignore
        y2 = torch.where(dilated.sum(1) > 1, ignore_label, y)
        '''

        # ----------------------------------------------
        # add boundary class as 15th class
        # boundary will be of course used as 15th class in train and test
        # dilated = get_dilated(y, config.num_classes, x.device, 4)  # shape is B x NUM_CLASSES x H x W

        # optionally remove background class from boundary calculation totally
        # dilated[:, 0, ...] = 0
        #
        # finally reclassify boundary to 15th class named boundary
        # y = torch.where(dilated.sum(1) > 1, 15, y)

        # ----------------------------------------------
        # reclassify boundary as boundary
        # y = torch.where(dilated.sum(1) > 1, 0, y)

        if mode != "train":
            with torch.no_grad():
                out = model(x, batch_positions=dates)
        else:
            optimizer.zero_grad()
            out = model(x, batch_positions=dates)

        if config.add_boundary_loss:
            out, out_b = out
            loss_b = criterion_b(out_b, y_b)

        # EXPERIMENTAL SETTING
        # loss = criterion(out, y2)
        loss = criterion(out, y)
        if mode == "train":

            loss = loss + loss_b if config.add_boundary_loss else loss
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            pred = out.argmax(dim=1)
            #catcher['pred'].append(pred.flatten().detach().cpu().numpy())
            #catcher['label'].append(y.flatten().detach().cpu().numpy())
            #catcher['conf'].append(rearrange(nn.Softmax(dim=1)(out), 'b c h w -> (b h w) c').detach().cpu().numpy().max(axis=1))
            pred_ = out.topk(2, dim=1).indices
            if config.add_boundary_loss:
                pred_b = out_b.argmax(dim=1)

            if config.get_affine:  # performs postprocessing (homogenization of prediction)
                pp = []

                for p, a in zip(pred, affine):
                    p = homogenize(p.detach().cpu().numpy(),
                                   vector_data_path=AGRI_PATH_DATASET, type_='hard',
                                   affine=a.cpu().numpy(), array_out=True)
                    pp.append(torch.from_numpy(p))
                    #catcher['hash'].append(p[0])
                    #catcher['area'].append(p[1])
                    #catcher['raster_val'].append(p[2])
                    #catcher['Legenda'].append(p[3])

                '''
                for p, a in zip(out, affine):
                    p = homogenize(nn.Softmax(dim=0)(p).detach().cpu().numpy(),
                                   vector_data_path=AGRI_PATH_DATASET, type_='soft',
                                   affine=a.cpu().numpy(), array_out=True)
                    pp.append(torch.from_numpy(p))
                '''
                pred = torch.stack(pp).to(config.device)

        # Specify test region default is all
        if test_region in ['boundary', 'interior']:
            dilated = get_dilated(y, config.num_classes, pred.device, 4)  # shape is B x NUM_CLASSES x H x W

            if test_region == 'boundary':
                # reclassify interior to ignore
                ignore_label = [i for i in range(config.num_classes)][config.ignore_index]
                y = torch.where(dilated.sum(1) == 1, ignore_label, y)
            elif test_region == 'interior':
                # reclassify boundary to ignore
                ignore_label = [i for i in range(config.num_classes)][config.ignore_index]
                y = torch.where(dilated.sum(1) > 1, ignore_label, y)

                #wh = torch.where(dilated.sum(1) > 1, 0, 1).flatten().detach().cpu().numpy()
                #whh = np.where(wh == 0)
                #catcher['pred'][-1] = catcher['pred'][-1][whh]
                #catcher['label'][-1] = catcher['label'][-1][whh]
                #catcher['conf'][-1] = catcher['conf'][-1][whh]
        # ----------------------------------------------
        pred_top2 = torch.where(y == pred_[:, 1, ...], pred_[:, 1, ...], pred_[:, 0, ...])
        iou_meter.add(pred, y)
        iou_meter_top2.add(pred_top2, y)
        loss_meter.add(loss.item())

        if config.add_boundary_loss:
            iou_meter_boundary.add(pred_b, y_b)

        if (i + 1) % config.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            miou_top2, acc_top2 = iou_meter_top2.get_miou_acc()
            if config.add_boundary_loss:
                miou_b, acc_b = iou_meter_boundary.get_miou_acc()
                logging.info(
                    "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}, Acc_t2 : {:.2f}, mIoU_t2 {:.2f},\
                     Acc_b : {:.2f}, mIoU_b {:.2f}".format(
                        i + 1, len(data_loader), loss_meter.value()[0], acc, miou, acc_top2, miou_top2, acc_b,
                        miou_b
                    )
                )
            else:
                logging.info(
                    "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}, Acc_t2 : {:.2f}, mIoU_t2 {:.2f}".format(
                        i + 1, len(data_loader), loss_meter.value()[0], acc, miou, acc_top2, miou_top2
                    )
                )
    t_end = time.time()
    #ll = np.concatenate(catcher['label'])
    #pred = np.concatenate(catcher['pred'])
    #from src.visualization.visualize import reliability_plot, bin_strength_plot
    #import matplotlib.pyplot as plt
    #conff = np.concatenate(catcher['conf'])
    #plot = reliability_plot(conff, pred, ll)
    #plot2 = bin_strength_plot(conff, pred, ll)
    total_time = t_end - t_start
    logging.info("Epoch time : {:.1f}s".format(total_time))
    miou, acc = iou_meter.get_miou_acc()
    miou_top2, acc_top2 = iou_meter_top2.get_miou_acc()
    if config.add_boundary_loss:
        miou_b, acc_b = iou_meter_boundary.get_miou_acc()
        metrics = {
            "{}_accuracy".format(mode): acc,
            "{}_accuracy_b".format(mode): acc_b,
            "{}_accuracy_top2".format(mode): acc_top2,
            "{}_loss".format(mode): loss_meter.value()[0],
            "{}_IoU".format(mode): miou,
            "{}_IoU_b".format(mode): miou_b,
            "{}_IoU_top2".format(mode): miou_top2,
            "{}_epoch_time".format(mode): total_time,
        }
    else:
        metrics = {
            "{}_accuracy".format(mode): acc,
            "{}_accuracy_top2".format(mode): acc_top2,
            "{}_loss".format(mode): loss_meter.value()[0],
            "{}_IoU".format(mode): miou,
            "{}_IoU_top2".format(mode): miou_top2,
            "{}_epoch_time".format(mode): total_time,
        }

    if mode == "test":
        if config.add_boundary_loss:
            return metrics, iou_meter.conf_metric.value(), iou_meter_top2.conf_metric.value() , iou_meter_boundary.conf_metric.value()
        else:
            return metrics, iou_meter.conf_metric.value(), iou_meter_top2.conf_metric.value()  # confusion matrix

    else:
        return metrics


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config, folds=None):
    os.makedirs(config.res_dir, exist_ok=True)
    if folds is None:
        for fold in range(1, 6):
            os.makedirs(os.path.join(config.res_dir, f"Fold_{fold}"), exist_ok=True)
    else:
        os.makedirs(os.path.join(config.res_dir, f"Fold_{folds}"), exist_ok=True)


def checkpoint(fold, log, config):
    with open(
            os.path.join(config.res_dir, "Fold_{}".format(fold), "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def save_results(fold, metrics, conf_mat, config, name="", top2=False):
    with open(
            os.path.join(config.res_dir, f"Fold_{fold}", f"{name}test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)

    if conf_mat is not None:
        pkl.dump(
            conf_mat,
            open(
                os.path.join(config.res_dir, f"Fold_{fold}", f"{name}conf_mat{'_top2' if top2 else ''}.pkl"), "wb"
            ),
        )


def overall_performance(config, fold=None, name="", top2=False):
    cm = np.zeros((config.num_classes, config.num_classes))

    if fold is None:
        for f in range(1, 6):
            try:
                cm += pkl.load(
                    open(
                        os.path.join(config.res_dir, f"Fold_{f}", f"{name}conf_mat{'_top2' if top2 else ''}.pkl"),
                        "rb",
                    )
                )
            except:
                pass
    else:
        try:
            cm += pkl.load(
                open(
                    os.path.join(config.res_dir, f"Fold_{fold}", f"{name}conf_mat{'_top2' if top2 else ''}.pkl"),
                    "rb",
                )
            )
        except:
            return

    if config.ignore_index is not None:
        cm = np.delete(cm, config.ignore_index, axis=0)
        cm = np.delete(cm, config.ignore_index, axis=1)

    per_class, perf = confusion_matrix_analysis(cm)

    logging.info("Overall performance:")
    logging.info(f"Acc: {perf['Accuracy']},  IoU (macro): {perf['MACRO_IoU']}")

    perf['folds'] = f'Performance calculated on folds: {"all" if fold is None else fold}'

    with open(os.path.join(config.res_dir, f"{name}overall{'_top2' if top2 else ''}.json"), "w") as file:
        file.write(json.dumps(perf, indent=4))

    with open(os.path.join(config.res_dir, f"{name}per_class{'_top2' if top2 else ''}.json"), "w") as file:
        file.write(json.dumps(per_class, indent=4))


def model_characteristics(model: nn.Module, device: str = 'cuda'):
    """
    Function for estimation of model characteristics like  FLOPs, MACs and num of params.
    Sample is of shape B x T x C x H x W = 1 x 30 x 10 x 128 x 128
    Parameters
    ----------
    model: nn.Module
        torch.nn.Module object
    device: str
        Name of device. Can be `cpu`, `cuda`
    """
    from thop import profile
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    model = model.to(device)

    sample_data = torch.randn(1, 30, 10, 128, 128).to(device)
    sample_batch_positions = torch.randint(0, 30, (1, 30)).to(device)

    print(f'Number of trainable parameters is: {get_ntrainparams(model)}')

    print('\n\n\n')

    macs, params = profile(model, inputs=(sample_data, sample_batch_positions))
    print(f'| module | # parameters      | #macs     |\n'
          f'| model  |  {params / 1e6}M         | {macs / 1e9}G')

    print('\n\n\n')

    flops = FlopCountAnalysis(model, (sample_data, sample_batch_positions))

    print(flop_count_table(flops))


def inference_time(model: nn.Module, device: str = 'cuda', repetitions: int = 100):
    """
    Function for estimation of model inference time i.e. time of one forward pass
    Sample is of shape B x T x C x H x W = 1 x 30 x 10 x 128 x 128
    Parameters
    ----------
    model: nn.Module
        torch.nn.Module object
    device: str
        Name of device. Can be `cpu`, `cuda`
    repetitions: int
        Number of repetitions
    """
    model = model.to(device)

    sample_data = torch.randn(1, 30, 10, 128, 128).to(device)
    sample_batch_positions = torch.randint(0, 30, (1, 30)).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = torch.zeros((repetitions, 1))

    # GPU-WARM-UP
    for _ in range(10):
        _ = model(sample_data, sample_batch_positions)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(sample_data, sample_batch_positions)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = torch.sum(timings) / repetitions
    std_syn = torch.std(timings)
    print(f'Average time of forward pass is {mean_syn} +- {std_syn} ms')
