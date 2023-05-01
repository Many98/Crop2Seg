import json
import logging
import os
import numpy as np
import argparse
import pprint

import torch
import torch.nn as nn
import torch.utils.data as data

from src.utils import pad_collate, get_ntrainparams
from src.datasets.s2_ts_cz_crop import S2TSCZCropDataset
from src.learning.weight_init import weight_init
from src.learning.smooth_loss import SmoothCrossEntropy2D
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
    help="Path to folder containing the network weights and conf.json file to resume training",
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
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--mono_date", default=None, type=str)
parser.add_argument("--ref_date", default="2018-09-01", type=str)
parser.add_argument(
    "--fold",
    default=1,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument("--num_classes", default=15, type=int)
parser.add_argument("--ignore_index", default=-1, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument("--conv_type", default="2d", type=str)
parser.add_argument("--use_transpose_conv", action='store_true')
parser.add_argument("--use_mbconv", action='store_true')
parser.add_argument("--add_squeeze", action='store_true')
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
    # as S2TSCZCrop dataset is bigger than PASTIS we will use different train/val/test splits
    # 1st four random folds will be used for training and 5th fold will be splitted to val and test
    # user can specify which fold sequence use by `fold` parameter default is 1
    fold_sequence = (
        ((1, 2, 3, 4), (5,)),
        ((1, 2, 3, 5), (4,)),
        ((1, 2, 5, 4), (3,)),
        ((1, 5, 3, 4), (2,)),
        ((5, 2, 3, 4), (1,)),
    )

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    device = torch.device(config.device)
    is_test_run = config.test
    finetuning = config.finetune
    start_epoch = 1
    best_mIoU = 0

    # weight_folder => user wants resume training
    if not config.weight_folder or finetuning:
        prepare_output(config)

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
            start_epoch = state["epoch"] + 1
            best_mIoU = state.get("best_mIoU", 0)
            optimizer_state_resume = state["optimizer"]
            config.update({"epochs": num_epochs})

    fold_sequence = fold_sequence[config.fold - 1]

    if not os.path.isfile(os.path.join(config.norm_values_folder, "NORM_S2_patch.json")):
        raise Exception(f"Norm parameter set to True but normalization values json file for dataset was "
                        f"not found in specified directory {config.norm_values_folder} .")

    with open(
            os.path.join(config.norm_values_folder, "NORM_S2_patch.json"), "r"
    ) as file:
        normvals = json.loads(file.read())

    means = [normvals[f"Fold_{f}"]["mean"] for f in fold_sequence[0]]
    stds = [normvals[f"Fold_{f}"]["std"] for f in fold_sequence[0]]

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
        channels_like_pastis=True
    )

    collate_fn = lambda x: pad_collate(x, pad_value=config.pad_value)

    if not is_test_run:
        dt_train = S2TSCZCropDataset(**dt_args, folds=fold_sequence[0], cache=config.cache)

        train_loader = data.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            #num_workers=2,
            #persistent_workers=True
        )

    dt_ = S2TSCZCropDataset(**dt_args, folds=fold_sequence[1], cache=config.cache)

    # TODO test it
    dt_val, dt_test = data.random_split(dt_, [0.5, 0.5])

    val_loader = data.DataLoader(
        dt_val,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        #num_workers=1,
        #persistent_workers=True
    )
    test_loader = data.DataLoader(
        dt_test,
        batch_size=config.batch_size,
        # shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        #num_workers=0,
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

    weights = torch.ones(config.num_classes, device=device).float()
    weights[config.ignore_index] = 0
    criterion = nn.CrossEntropyLoss(weight=weights)

    # TODO test `SmoothCrossEntropy2D`
    # criterion = SmoothCrossEntropy2D(weight=weights)

    if not is_test_run:
        # Training loop
        trainlog = {}

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
                            "fold": config.fold,
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
    if config.weight_folder:
        assert os.path.isdir(config.weight_folder), f'Path {config.weight_folder} where should be stored weights of ' \
                                                    f'network and conf.json file is not valid'
    else:
        assert os.path.isdir(config.res_dir), f'Path {config.res_dir} for export of results is not valid'
        assert config.num_classes == config.out_conv[
            -1], f'Number of classes {config.num_classes} does not match number of' \
                 f' output channels {config.out_conv[-1]}'
    assert config.fold in [1, 2, 3, 4, 5], f'Parameter `fold` must be one of [1, 2, 3, 4, 5] but is {config.fold}'

    pprint.pprint(config)
    main(config)

    # TODO make sure ignore index works as expected (also in CM calculation)
