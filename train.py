import json
import logging
import os
import numpy as np
import argparse
import pprint

import torch
import torch.nn as nn
import torch.utils.data as data

from segmentation_models_pytorch.losses import FocalLoss, TverskyLoss, LovaszLoss

from src.utils import pad_collate, get_ntrainparams
from src.datasets.s2_ts_cz_crop import S2TSCZCropDataset
from src.learning.weight_init import weight_init
from src.learning.smooth_loss import SmoothCrossEntropy2D
from src.learning.recall_loss import RecallCrossEntropy
from src.learning.utils import iterate, overall_performance, save_results, prepare_output, checkpoint, get_model

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="utae",
    type=str,
    help="Type of architecture to use. Can be one of: (utae/unet3d)",
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
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)

# Set-up parameters
parser.add_argument(
    "--test",
    action='store_true',
    help="Whether to perform test run (inference)"
         "Weights stored in `--weight_folder` directory  will be used",
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
    default="",
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
    default=1,
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
parser.add_argument("--use_transpose_conv", action='store_true', help="Whether to use transposed 3D convolutions for"
                                                                      " up-sampling attention masks instead of simple"
                                                                      " bi-linear interpolation")
parser.add_argument("--use_mbconv", action='store_true', help="Whether to use MBConv module instead of classical "
                                                              " convolutional layers")
parser.add_argument("--add_squeeze", action='store_true', help="Whether to add squeeze & excitation module")
parser.add_argument("--use_doy", action='store_true', help="Whether to use absolute positional encoding (day of year)"
                                                           " instead of relative encoding w.r.t. reference date")
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


def main(config):
    """
    Main training function.

    This function can run in 4 states:
        - Training from scratch
        - Fine-tuning
        - Resume training/fine-tuning
        - Pure inference (testing)
    Notes:
        To perform only inference use `--test` flag
        To perform fine-tuning use `--finetune` flag
    """
    # In S2TSCZCrop dataset  we will use classical train/val/test splits
    # `fold` parameter is not used

    # TODO this fold sequence will be used only if PASTIS is used e.g. for finetuning
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    device = torch.device(config.device)
    is_test_run = config.test
    finetuning = config.finetune
    start_epoch = 1
    best_mIoU = 0
    trainlog = {}

    # weight_folder => user wants resume training
    if not config.weight_folder or finetuning:
        prepare_output(config, folds=1)

    if config.weight_folder:
        logging.info(f"LOADING WEIGHTS FROM {os.path.join(config.weight_folder, 'model.pth.tar')}")
        state = torch.load(
            os.path.join(
                config.weight_folder, "model.pth.tar"
            )
        )

        state_dict = state["state_dict"]

        weight_folder = config.weight_folder
        num_epochs = config.epochs

        if not finetuning:
            logging.info(f"LOADING STATE JSON FROM {os.path.join(config.weight_folder, 'conf.json')}")
            with open(os.path.join(config.weight_folder, 'conf.json'), 'r') as f:
                config = json.load(f)
                config.update({"weight_folder": weight_folder})

        if not is_test_run and not finetuning:
            logging.info("RESUMING TRAINING...")
            try:
                with open(os.path.join(config.weight_folder, 'trainlog.json'), 'r') as f:
                    trainlog = json.load(f)
            except:
                trainlog = {}

            start_epoch = state["epoch"] + 1
            best_mIoU = state.get("best_mIoU", 0)
            optimizer_state_resume = state["optimizer"]
            config.update({"epochs": num_epochs})

        config = argparse.Namespace(**config)

    fold_sequence = fold_sequence[config.fold - 1]

    config.fold = 1  # we do not need fold parameter therefore here hardcoded to 1. but still using for consistency

    if not os.path.isfile(os.path.join(config.norm_values_folder, "NORM_S2_patch.json")):
        raise Exception(f"Norm parameter set to True but normalization values json file for dataset was "
                        f"not found in specified directory {config.norm_values_folder} .")

    with open(
            os.path.join(config.norm_values_folder, "NORM_S2_patch.json"), "r"
    ) as file:
        normvals = json.loads(file.read())

    if 'Fold' in list(normvals.keys())[0]:
        means = [normvals[f"Fold_{f}"]["mean"] for f in fold_sequence[0]]
        stds = [normvals[f"Fold_{f}"]["std"] for f in fold_sequence[0]]
    elif 'train' in list(normvals.keys())[0]:
        means = [normvals[f"train"]["mean"]]
        stds = [normvals[f"train"]["std"]]
    else:
        raise Exception('Unknown structure of normalization values json file')

    # TODO here is fix for channels order to be like in PASTIS dataset
    channels_order = [2, 1, 0, 4, 5, 6, 3, 7, 8, 9]
    norm_values = {'mean': np.stack(means).mean(axis=0)[channels_order],
                   'std': np.stack(stds).mean(axis=0)[channels_order]}

    # TODO here was mistake in original implementation, authors did not use train set norm values on
    #  validation and test sets but norm values from validation and test sets respectively
    # Dataset definition
    dt_args = dict(
        folder=config.dataset_folder,
        norm=True,
        norm_values=norm_values,
        reference_date=config.ref_date,
        mono_date=config.mono_date,
        from_date=None,
        to_date=None,
        channels_like_pastis=True,
        use_doy=config.use_doy
    )

    collate_fn = lambda x: pad_collate(x, pad_value=config.pad_value)

    if not is_test_run:
        dt_train = S2TSCZCropDataset(**dt_args,
                                     # folds=fold_sequence[0],
                                     set_type='train',
                                     cache=config.cache)

        train_loader = data.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            # num_workers=2,
            # persistent_workers=True
        )

    # dt_ = S2TSCZCropDataset(**dt_args, folds=fold_sequence[1], cache=config.cache)

    dt_val = S2TSCZCropDataset(**dt_args, set_type='val', cache=config.cache)
    dt_test = S2TSCZCropDataset(**dt_args, set_type='test', cache=config.cache)

    # dt_val, dt_test = data.random_split(dt_, [0.5, 0.5])

    val_loader = data.DataLoader(
        dt_val,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        # num_workers=1,
        # persistent_workers=True
    )
    test_loader = data.DataLoader(
        dt_test,
        batch_size=config.batch_size,
        # shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        # num_workers=0,
        # persistent_workers=True
    )

    if not is_test_run:
        logging.info(
            f"Train: {len(dt_train)} samples, Val: {len(dt_val)} samples, Test: {len(dt_test)} samples"
        )
    else:
        logging.info(
            f"Test: {len(dt_test)} samples"
        )

    # Model definition
    model = get_model(config)

    if config.weight_folder:
        model.load_state_dict(state_dict)

    # HERE COMES FINE-TUNING CODE
    # -------------------------------
    # for now just initialize UTAE with weights from pretrained network
    if finetuning:
        for name, p in model.named_parameters():
            p.requires_grad = True
    # --------------------------------

    config.N_params = get_ntrainparams(model)

    if not config.weight_folder:
        with open(os.path.join(config.res_dir, f"Fold_{config.fold}", "conf.json"), "w") as file:
            file.write(json.dumps(vars(config), indent=4))

    if not is_test_run:
        logging.info(f"TOTAL TRAINABLE PARAMETERS : {config.N_params}")
        # print(model)
        """
        logging.info("Trainable layers:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                logging.info(name)
        """

    model = model.to(device)

    if not config.weight_folder:
        model.apply(weight_init)

    if not is_test_run:
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        if config.weight_folder and not is_test_run and not finetuning:
            optimizer.load_state_dict(optimizer_state_resume)

    # TODO maybe try scheduler as well

    # note that we ensure ignoring some classes by setting weight to 0
    weights = torch.ones(config.num_classes, device=device).float()
    weights[config.ignore_index] = 0

    # ---- ATTEMPTS TO RESOLVE CLASS IMBALANCE PROBLEM ----------------
    #
    # ---- By adjusting weights ----------

    # first attempt
    # weights[:-1] = 1 / torch.tensor([0.3, 0.08, 0.015, 0.08, 0.22, 0.1, 0.09, 0.02, 0.05, 0.001, 0.015, 0.008,
    #                                 0.015, 0.05], device=device)

    # second attempt (v2)
    weights[:-1] = 1 / torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.1, 0.1, 0.1, 0.04, 0.1, 0.1,
                                     0.1, 0.1], device=device)

    #criterion = nn.CrossEntropyLoss(weight=weights)

    # ---- By using specific loss functions ----------

    # Focal loss ; ref: https://arxiv.org/pdf/1708.02002v2.pdf  (modified CE loss)
    criterion = FocalLoss(mode='multiclass',
                          gamma=2.0,
                          ignore_index=[i for i in range(15)][config.ignore_index])

    '''
    # Recall Loss; ref: https://arxiv.org/pdf/2106.14917.pdf (similarly like FocalLoss it tries to dynamically weight
    #                                                           CrossEntropy)
    criterion = RecallCrossEntropy(n_classes=config.num_classes,
                                   ignore_index=[i for i in range(15)][config.ignore_index])
    
    # Lovasz loss; ref: https://arxiv.org/pdf/1705.08790.pdf  (it optimizes IoU)
    criterion = LovaszLoss(mode='multiclass', per_image=False,
                           ignore_index=[i for i in range(15)][config.ignore_index])

    # Tversky Loss ; ref: https://arxiv.org/pdf/1706.05721.pdf
    #   (modification of dice-coefficient or jaccard coefficient by weighting FP and FN)
    criterion = TverskyLoss(mode='multiclass', classes=None,
                            smooth=0.0, ignore_index=[i for i in range(15)][config.ignore_index],
                            alpha=0.5,  # alpha weights FP
                            beta=0.5,  # beta weights FN
                            gamma=1.0
                            )
    '''
    # -----------------------------------------------------------------------

    # SmoothCrossEntropy2D - our modification of classical 2D CE with specific labels smoothing on
    #  borders of crop fields which should help with pixel mixing problem (on boundaries of semantic classes)
    # criterion = SmoothCrossEntropy2D(weight=weights, background_treatment=False)

    if not is_test_run:
        # Training loop
        logging.info(f"STARTING FROM EPOCH: {start_epoch} \n"
                     f"TRAINING PLAN: {config.epochs} EPOCHS TO BE COMPLETED")
        for epoch in range(start_epoch, config.epochs + start_epoch):
            logging.info(f"EPOCH {epoch}/{config.epochs + start_epoch - 1}")

            model.train()
            train_metrics = iterate(
                model,
                data_loader=train_loader,
                criterion=criterion,
                config=config,
                optimizer=optimizer,
                mode="train",
                device=device,
            )
            if epoch % config.val_every == 0 and epoch > config.val_after:
                logging.info("VALIDATION ... ")
                model.eval()
                val_metrics = iterate(
                    model,
                    data_loader=val_loader,
                    criterion=criterion,
                    config=config,
                    mode="val",
                    device=device,
                )

                logging.info(
                    "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                        val_metrics["val_loss"],
                        val_metrics["val_accuracy"],
                        val_metrics["val_IoU"],
                    )
                )

                trainlog[epoch] = {**train_metrics, **val_metrics}
                checkpoint(config.fold, trainlog, config)
                if val_metrics["val_IoU"] >= best_mIoU:
                    best_mIoU = val_metrics["val_IoU"]
                    torch.save(
                        {
                            "best_mIoU": best_mIoU,
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            config.res_dir, f"Fold_{config.fold}", "model.pth.tar"
                        ),
                    )
            else:
                trainlog[epoch] = {**train_metrics}
                checkpoint(config.fold, trainlog, config)

        model.load_state_dict(
            torch.load(
                os.path.join(
                    config.res_dir, f"Fold_{config.fold}", "model.pth.tar"
                )
            )["state_dict"]
        )

    # Inference
    logging.info("TESTING BEST EPOCH ...")

    model.eval()

    test_metrics, conf_mat = iterate(
        model,
        data_loader=test_loader,
        criterion=criterion,
        config=config,
        mode="test",
        device=device,
    )
    logging.info(
        "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
            test_metrics["test_loss"],
            test_metrics["test_accuracy"],
            test_metrics["test_IoU"],
        )
    )
    save_results(config.fold, test_metrics, conf_mat.cpu().numpy(), config)

    overall_performance(config, config.fold)


if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))


    assert not config.finetune or not config.test, f'Use only one flag. Either `--finetune` or `--test`'
    assert os.path.isdir(config.dataset_folder), f'Path {config.dataset_folder} for dataset is not valid'
    assert os.path.isdir(config.norm_values_folder), f'Path {config.norm_values_folder} where to look for normalization' \
                                                     f'values is not valid'
    assert os.path.isfile(os.path.join(config.norm_values_folder, 'NORM_S2_patch.json')), \
        f'There is no NORM_S2_patch.json file with normalization values in {config.norm_values_folder}. \n' \
        f'You should call appropriate `compute_norm_vals` function (from pastis.py or s2_ts_cz_crop.py)' \
        f'to generate normalization values for corresponding dataset'
    if config.weight_folder:
        assert os.path.isdir(config.weight_folder), f'Path {config.weight_folder} where should be stored weights of ' \
                                                    f'network and conf.json file is not valid'
    else:
        assert os.path.isdir(config.res_dir), f'Path {config.res_dir} for export of results is not valid'
        assert config.num_classes == config.out_conv[
            -1], f'Number of classes {config.num_classes} does not match number of' \
                 f' output channels {config.out_conv[-1]}'
    assert config.fold in [1, 2, 3, 4, 5], f'Parameter `fold` must be one of [1, 2, 3, 4, 5] but is {config.fold}'
    assert config.conv_type in ['2d', 'depthwise_separable'], f'Parameter `conv_type` must be one of ' \
                                                              f' [2d, depthwise_separable] but is {config.conv_type}'

    pprint.pprint(config)
    main(config)
