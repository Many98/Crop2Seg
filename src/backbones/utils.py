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
