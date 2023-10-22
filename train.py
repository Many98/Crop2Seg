import json
import logging
import os
import numpy as np
import argparse
import pprint

import torch
import torch.nn as nn
import torch.utils.data as data

# from segmentation_models_pytorch.losses import FocalLoss, TverskyLoss, LovaszLoss

from src.utils import pad_collate, get_ntrainparams, Transform
from src.datasets.s2_ts_cz_crop import S2TSCZCropDataset
from src.datasets.pastis import PASTISDataset
from src.learning.weight_init import weight_init
from src.learning.smooth_loss import SmoothCrossEntropy2D
from src.learning.recall_loss import RecallCrossEntropy
from src.learning.utils import iterate, overall_performance, save_results, prepare_output, checkpoint, get_model, \
    iterate_pretrain

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
parser.add_argument("--seg_model", default='unet', type=str,
                    help="Model to use for segmentation")
parser.add_argument("--temp_model", default='ltae', type=str,
                    help="Model to use for temporal encoding")

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

    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    device = torch.device(config.device)
    is_test_run = config.test
    finetuning = config.finetune
    start_epoch = 1
    best_mIoU = 0
    best_loss = 1e9
    trainlog = {}

    # weight_folder => user wants resume training
    if not config.weight_folder or finetuning:
        prepare_output(config, folds=config.fold)

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
        test_region = config.test_region

        if not finetuning:
            logging.info(f"LOADING STATE JSON FROM {os.path.join(config.weight_folder, 'conf.json')}")
            with open(os.path.join(config.weight_folder, 'conf.json'), 'r') as f:
                config = json.load(f)
                config.update({"weight_folder": weight_folder})
                config.update({"test_region": test_region})

        if not is_test_run and not finetuning:
            logging.info("RESUMING TRAINING...")
            try:
                with open(os.path.join(config.weight_folder, 'trainlog.json'), 'r') as f:
                    trainlog = json.load(f)
            except:
                trainlog = {}

            start_epoch = state["epoch"] + 1
            best_mIoU = state.get("best_mIoU", 0)  # TODO adjust for pretraining
            optimizer_state_resume = state["optimizer"]
            config.update({"epochs": num_epochs})

        config = argparse.Namespace(**config) if not isinstance(config, argparse.Namespace) else config

    fold_sequence = fold_sequence[config.fold - 1]

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
        # here is fix for channels order to be like in PASTIS dataset
        channels_order = [i for i in range(10)]
    elif 'train' in list(normvals.keys())[0]:
        means = [normvals[f"train"]["mean"]]
        stds = [normvals[f"train"]["std"]]
        # here is fix for channels order to be like in PASTIS dataset
        channels_order = [2, 1, 0, 4, 5, 6, 3, 7, 8, 9]
    else:
        raise Exception('Unknown structure of normalization values json file')

    norm_values = {'mean': np.stack(means).mean(axis=0)[channels_order],
                   'std': np.stack(stds).mean(axis=0)[channels_order]}

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
        use_doy=config.use_doy,
        add_ndvi=config.add_ndvi,
        use_abs_rel_enc=config.use_abs_rel_enc,
        temporal_dropout=config.temporal_dropout,
        pretrain=config.pretrain
    )

    if config.add_ndvi:
        config.input_dim += 1

    collate_fn = lambda x: pad_collate(x, pad_value=config.pad_value)

    train_folds, val_fold, test_fold = fold_sequence

    if not is_test_run:
        transform = None  # Transform(add_noise=True)

        if config.dataset.lower() == 'pastis':
            dt_train = PASTISDataset(**dt_args, folds=train_folds,
                                     # norm_folds=train_folds,
                                     set_type='train',
                                     cache=config.cache)
        else:
            dt_train = S2TSCZCropDataset(**dt_args,
                                         # folds=fold_sequence[0],
                                         set_type='train',
                                         transform=transform,
                                         cache=config.cache)
            sample_weights = torch.from_numpy(dt_train.meta_patch.weight.values).float()
            sampler = data.WeightedRandomSampler(weights=sample_weights,
                                                 num_samples=5 * len(sample_weights),
                                                 replacement=True
                                                 )

        train_loader = data.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            # sampler=sampler,
            drop_last=True,
            collate_fn=collate_fn,
            # num_workers=2,
            # persistent_workers=True
        )

    if config.dataset.lower() == 'pastis':
        dt_val = PASTISDataset(**dt_args, folds=val_fold, set_type='val',
                               cache=config.cache)
        dt_test = PASTISDataset(**dt_args, folds=test_fold, set_type='test')
    else:
        dt_val = S2TSCZCropDataset(**dt_args, set_type='val', cache=config.cache)
        dt_test = S2TSCZCropDataset(**dt_args, set_type='test', cache=config.cache)

    val_loader = data.DataLoader(
        dt_val,
        batch_size=config.batch_size,
        drop_last=True,
        collate_fn=collate_fn,
        # num_workers=1,
        # persistent_workers=True
    )
    test_loader = data.DataLoader(
        dt_test,
        batch_size=config.batch_size,
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

    if config.pretrain:
        config.out_conv = [config.out_conv[0], config.input_dim]

    # Model definition
    model = get_model(config)

    # HERE COMES FINE-TUNING CODE
    # -------------------------------
    # use it adjust state_dict; load model weights or freeze layers etc.
    # for now just initialize UTAE with weights from pretrained network
    if finetuning:
        '''
        # for name, p in model.named_parameters():
        #    p.requires_grad = False
        for name, p in model.in_conv.named_children():
            p.requires_grad = False
        for name, p in model.down_blocks.named_children():
            p.requires_grad = False
        model.out_conv.conv.conv[3] = nn.Conv2d(32, config.num_classes, kernel_size=(3, 3), stride=(1, 1),
                                                padding=(1, 1), padding_mode='reflect')
        model.out_conv.conv.conv[4] = nn.BatchNorm2d(config.num_classes, eps=1e-05, momentum=0.1, affine=True,
                                                     track_running_stats=True)
        for name, p in model.out_conv.named_children():
            p.requires_grad = True
        '''

        layers_to_remove = []
        for k in state_dict:
            if "out_conv" in k:  # or "in_conv" in k or "spatial_red" in k
                layers_to_remove.append(k)

        for key in layers_to_remove:
            del state_dict[key]

        model.out_conv.conv.conv[3] = nn.Conv2d(32, config.num_classes, kernel_size=(3, 3), stride=(1, 1),
                                                padding=(1, 1), padding_mode='reflect')
        model.out_conv.conv.conv[4] = nn.BatchNorm2d(config.num_classes, eps=1e-05, momentum=0.1, affine=True,
                                                     track_running_stats=True)

        model.apply(weight_init)
        model_dict = model.state_dict()

        non_pretrained_dict = {k: v for k, v in model_dict.items() if "out_conv" in k}
        state_dict.update(non_pretrained_dict)

        model.load_state_dict(state_dict)

        for p in model.in_conv.parameters():
            p.requires_grad = False
        for p in model.down_blocks.parameters():
            p.requires_grad = False
        for p in model.spatial_reduction.parameters():
            p.requires_grad = False

        '''
        layers_to_remove = []
        for k in state_dict:
            if "down_block" in k or "up_block" in k or "out_conv" in k or "linear_clf" in k:  # or "in_conv" in k or "spatial_red" in k
                layers_to_remove.append(k)

        for key in layers_to_remove:
            del state_dict[key]

        model.apply(weight_init)
        model_dict = model.state_dict()

        non_pretrained_dict = {k: v for k, v in model_dict.items() if "down_block" in k or
                               "up_block" in k or "out_conv" in k} # or "in_conv" in k or "spatial_red" in k
        state_dict.update(non_pretrained_dict)

        model.load_state_dict(state_dict)
        '''
        # --------------------------------
    else:
        if config.weight_folder:
            model.load_state_dict(state_dict)

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
        # optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        # optimizer = torch.optim.RAdam(model.parameters(), lr=config.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=-1, verbose=False)

        if config.weight_folder and not is_test_run and not finetuning:
            optimizer.load_state_dict(optimizer_state_resume)

    if not config.pretrain:
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
        # weights[:-1] = 1 / torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.1, 0.1, 0.1, 0.04, 0.1, 0.1,
        #                                 0.1, 0.1], device=device)

        criterion = nn.CrossEntropyLoss(weight=weights,
                                        # label_smoothing=0.2
                                        )

        # ---- By using specific loss functions ----------

        # Focal loss ; ref: https://arxiv.org/pdf/1708.02002v2.pdf  (modified CE loss)
        '''
        criterion = FocalLoss(mode='multiclass',
                              gamma=2.0,
                              ignore_index=[i for i in range(config.num_classes)][config.ignore_index])


        # Recall Loss; ref: https://arxiv.org/pdf/2106.14917.pdf (similarly like FocalLoss it tries to dynamically weight
        #                                                           CrossEntropy)
        criterion = RecallCrossEntropy(n_classes=config.num_classes,
                                       ignore_index=[i for i in range(config.num_classes)][config.ignore_index])

        # Lovasz loss; ref: https://arxiv.org/pdf/1705.08790.pdf  (it optimizes IoU)
        criterion = LovaszLoss(mode='multiclass', per_image=False,
                               ignore_index=[i for i in range(config.num_classes)][config.ignore_index])

        # Tversky Loss ; ref: https://arxiv.org/pdf/1706.05721.pdf
        #   (modification of dice-coefficient or jaccard coefficient by weighting FP and FN)
        criterion = TverskyLoss(mode='multiclass', classes=None,
                                smooth=0.0, ignore_index=[i for i in range(config.num_classes)][config.ignore_index],
                                alpha=0.5,  # alpha weights FP
                                beta=0.5,  # beta weights FN
                                gamma=1.0
                                )
        '''
        # -----------------------------------------------------------------------

        # SmoothCrossEntropy2D - our modification of classical 2D CE with specific labels smoothing on
        #  boundaries of crop fields which should help with pixel mixing problem (on boundaries of semantic classes)
        # criterion = SmoothCrossEntropy2D(weight=weights, background_treatment=False)
    else:

        criterion = nn.MSELoss()

    if not is_test_run:
        # Training loop
        logging.info(f"STARTING FROM EPOCH: {start_epoch} \n"
                     f"TRAINING PLAN: {config.epochs} EPOCHS TO BE COMPLETED")
        for epoch in range(start_epoch, config.epochs + start_epoch):
            logging.info(f"EPOCH {epoch}/{config.epochs + start_epoch - 1}")

            model.train()

            if not config.pretrain:
                train_metrics = iterate(
                    model,
                    data_loader=train_loader,
                    criterion=criterion,
                    config=config,
                    optimizer=optimizer,
                    scheduler=None,
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
            else:
                train_metrics = iterate_pretrain(
                    model,
                    data_loader=train_loader,
                    criterion=criterion,
                    config=config,
                    optimizer=optimizer,
                    scheduler=None,
                    mode="train",
                    device=device,
                )
                if epoch % config.val_every == 0 and epoch > config.val_after:
                    logging.info("VALIDATION ... ")
                    model.eval()

                    val_metrics = iterate_pretrain(
                        model,
                        data_loader=val_loader,
                        criterion=criterion,
                        config=config,
                        mode="val",
                        device=device,
                    )

                    logging.info(
                        "Loss {:.4f}".format(
                            val_metrics["val_loss"],
                        )
                    )

                    trainlog[epoch] = {**train_metrics, **val_metrics}
                    checkpoint(config.fold, trainlog, config)
                    if val_metrics["val_loss"] <= best_loss:
                        best_loss = val_metrics["val_loss"]
                        torch.save(
                            {
                                "best_loss": best_loss,
                                "epoch": epoch,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            },
                            os.path.join(
                                config.res_dir, f"Fold_{config.fold}", "model_pretrain.pth.tar"
                            ),
                        )
                    if epoch % 20 == 0:
                        torch.save(
                            {
                                "epoch": epoch,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            },
                            os.path.join(
                                config.res_dir, f"Fold_{config.fold}", f"model_pretrain_{epoch}.pth.tar"
                            ),
                        )
                else:
                    trainlog[epoch] = {**train_metrics}
                    checkpoint(config.fold, trainlog, config)

        if config.pretrain:
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        config.res_dir, f"Fold_{config.fold}", "model_pretrain.pth.tar"
                    )
                )["state_dict"]
            )
        else:
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

    if config.pretrain:
        test_metrics = iterate_pretrain(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            mode="test",
            device=device,
        )
        logging.info(
            f"Loss {test_metrics['test_loss']}"

        )
        save_results(config.fold, test_metrics, None, config,
                     name=f"{config.test_region}_", top2=False)
        save_results(config.fold, test_metrics, None, config,
                     name=f"{config.test_region}_", top2=True)

    else:
        test_metrics, conf_mat, conf_mat_top2 = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            mode="test",
            device=device,
        )
        logging.info(
            "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}, Acc_top2 {:.2f},  IoU_top2 {:.4f}".format(
                test_metrics["test_loss"],
                test_metrics["test_accuracy"],
                test_metrics["test_IoU"],
                test_metrics["test_accuracy_top2"],
                test_metrics["test_IoU_top2"],
            )
        )
        save_results(config.fold, test_metrics, conf_mat.cpu().numpy(), config,
                     name=f"{config.test_region}_", top2=False)
        save_results(config.fold, test_metrics, conf_mat_top2.cpu().numpy(), config,
                     name=f"{config.test_region}_", top2=True)

        overall_performance(config, name=f"{config.test_region}_", top2=False)
        overall_performance(config, name=f"{config.test_region}_", top2=True)


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
    assert config.fold in [1, 2, 3, 4, 5, None], f'Parameter `fold` must be one of [1, 2, 3, 4, 5] but is {config.fold}'
    assert config.conv_type in ['2d', 'depthwise_separable'], f'Parameter `conv_type` must be one of ' \
                                                              f' [2d, depthwise_separable] but is {config.conv_type}'

    pprint.pprint(config)

    if config.test or config.dataset != 'pastis':
        folds = [1]
    else:
        folds = list(range(1, 6)) if config.fold is None else [config.fold]

    for f in folds:
        config.fold = f
        main(config)

