import json
import logging
import os
import pickle as pkl
import time

import numpy as np
import torch
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


def iterate(
        model, data_loader, criterion, config, optimizer=None, mode="train", device=None
):
    loss_meter = tnt.meter.AverageValueMeter()
    iou_meter = IoU(
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

        if mode != "train":
            with torch.no_grad():
                out = model(x, batch_positions=dates)
        else:
            optimizer.zero_grad()
            out = model(x, batch_positions=dates)

        loss = criterion(out, y)
        if mode == "train":
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = out.argmax(dim=1)
        iou_meter.add(pred, y)
        loss_meter.add(loss.item())

        if (i + 1) % config.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            logging.info(
                "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}".format(
                    i + 1, len(data_loader), loss_meter.value()[0], acc, miou
                )
            )

    t_end = time.time()
    total_time = t_end - t_start
    logging.info("Epoch time : {:.1f}s".format(total_time))
    miou, acc = iou_meter.get_miou_acc()
    metrics = {
        "{}_accuracy".format(mode): acc,
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_IoU".format(mode): miou,
        "{}_epoch_time".format(mode): total_time,
    }

    if mode == "test":
        return metrics, iou_meter.conf_metric.value()  # confusion matrix
    else:
        return metrics


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)
    for fold in range(1, 6):
        os.makedirs(os.path.join(config.res_dir, "Fold_{}".format(fold)), exist_ok=True)


def checkpoint(fold, log, config):
    with open(
            os.path.join(config.res_dir, "Fold_{}".format(fold), "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def save_results(fold, metrics, conf_mat, config):
    with open(
            os.path.join(config.res_dir, "Fold_{}".format(fold), "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(
        conf_mat,
        open(
            os.path.join(config.res_dir, "Fold_{}".format(fold), "conf_mat.pkl"), "wb"
        ),
    )


def overall_performance(config, fold=None):
    cm = np.zeros((config.num_classes, config.num_classes))
    if fold is None:
        for f in range(1, 6):
            cm += pkl.load(
                open(
                    os.path.join(config.res_dir, "Fold_{}".format(f), "conf_mat.pkl"),
                    "rb",
                )
            )
    else:
        cm += pkl.load(
            open(
                os.path.join(config.res_dir, "Fold_{}".format(fold), "conf_mat.pkl"),
                "rb",
            )
        )

    if config.ignore_index is not None:
        cm = np.delete(cm, config.ignore_index, axis=0)
        cm = np.delete(cm, config.ignore_index, axis=1)

    per_class, perf = confusion_matrix_analysis(cm)

    logging.info("Overall performance:")
    logging.info(f"Acc: {perf['Accuracy']},  IoU (macro): {perf['MACRO_IoU']}")

    perf['folds'] = f'Performance calculated on folds: {"all" if fold is None else fold}'

    with open(os.path.join(config.res_dir, "overall.json"), "w") as file:
        file.write(json.dumps(perf, indent=4))

    with open(os.path.join(config.res_dir, "per_class.json"), "w") as file:
        file.write(json.dumps(per_class, indent=4))
