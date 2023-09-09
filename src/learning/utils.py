import json
import logging
import os
import pickle as pkl
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchnet as tnt

# ### small boiler plate to add src to sys path
import sys
from pathlib import Path

file = Path(__file__).resolve()
root = str(file).split('src')[0]
sys.path.append(root)
# --------------------------------------


from src.learning.metrics import confusion_matrix_analysis
from src.learning.miou import IoU

from src.backbones.utae import UTAE
from src.backbones.unet3d import UNet3D


def get_model(config):
    if config.model == "utae":
        model = UTAE(
            input_dim=10,  # number of input channels
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
            use_transpose_conv=config.use_transpose_conv,
            use_mbconv=config.use_mbconv,
            add_squeeze_excit=config.add_squeeze
        )
    elif config.model == "unet3d":
        model = UNet3D(
            in_channel=10, n_classes=config.num_classes, pad_value=config.pad_value
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
        model, data_loader, criterion, config, optimizer=None, mode="train", device=None, test_region='all'
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
    mode
    device
    test_region: str
        New parameter to test performance on `all` pixels or `boundary` pixels or `interior` pixels
        Currently works by reclassification to ignore class
        Note that code expects that there is set ignore index
    """
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

    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch
        y = y.long()

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

        # EXPERIMENTAL SETTING
        # loss = criterion(out, y2)
        loss = criterion(out, y)
        if mode == "train":
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = out.argmax(dim=1)
            pred_ = out.topk(2, dim=1).indices

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

        # ----------------------------------------------
        pred_top2 = torch.where(y == pred_[:, 1, ...], pred_[:, 1, ...], pred_[:, 0, ...])
        iou_meter.add(pred, y)
        iou_meter_top2.add(pred_top2, y)
        loss_meter.add(loss.item())

        if (i + 1) % config.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            miou_top2, acc_top2 = iou_meter_top2.get_miou_acc()
            logging.info(
                "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}, Acc_t2 : {:.2f}, mIoU_t2 {:.2f}".format(
                    i + 1, len(data_loader), loss_meter.value()[0], acc, miou, acc_top2, miou_top2
                )
            )

    t_end = time.time()
    total_time = t_end - t_start
    logging.info("Epoch time : {:.1f}s".format(total_time))
    miou, acc = iou_meter.get_miou_acc()
    miou_top2, acc_top2 = iou_meter_top2.get_miou_acc()
    metrics = {
        "{}_accuracy".format(mode): acc,
        "{}_accuracy_top2".format(mode): acc_top2,
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_IoU".format(mode): miou,
        "{}_IoU_top2".format(mode): miou_top2,
        "{}_epoch_time".format(mode): total_time,
    }

    if mode == "test":
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
    if not os.path.isfile(os.path.join(config.res_dir, "Fold_{}".format(fold), f"{name}test_metrics.json")):
        with open(
                os.path.join(config.res_dir, "Fold_{}".format(fold), f"{name}test_metrics.json"), "w"
        ) as outfile:
            json.dump(metrics, outfile, indent=4)
    pkl.dump(
        conf_mat,
        open(
            os.path.join(config.res_dir, "Fold_{}".format(fold), f"{name}conf_mat{'_top2' if top2 else ''}.pkl"), "wb"
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
                return
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
