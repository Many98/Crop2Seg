from src.backbones import utae, unet3d, timeunet


def get_model(config):
    if config.model == "utae":
        model = utae.UTAE(  # TODO Temporaly changed to timeunet_v2
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
            conv_type=config.conv_type,  # TODO here added conv_type
            use_transpose_conv=config.use_transpose_conv,
            use_mbconv=config.use_mbconv
        )
    elif config.model == "unet3d":
        model = unet3d.UNet3D(
            in_channel=10, n_classes=config.num_classes, pad_value=config.pad_value
        )
    return model
