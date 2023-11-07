"""
Modified U-TAE Implementation
Based on Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import torch
import torch.nn as nn

from src.backbones.tae import LTAE4WTAE
from src.backbones.temporal_aggregator import TemporalAggregator
from src.backbones.conv import ConvBlock, UpConvBlock, DownConvBlock
from src.backbones.mbconv import MBConvBlock, MBUpConvBlock, MBDownConvBlock


class WTAE(nn.Module):
    """
    Modified implementation of U-TAE architecture for spatio-temporal encoding of
    satellite image time series.
    """

    def __init__(
            self,
            input_dim,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 20],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            agg_mode="att_group",
            encoder_norm="group",
            n_head=16,
            d_model=256,
            d_k=4,
            encoder=False,
            return_maps=False,
            pad_value=0,
            padding_mode="reflect",
            # ------------------------------
            # From here starts added arguments
            conv_type='2d',
            use_mbconv=False,
            add_squeeze_excit=False,
            use_abs_rel_enc=False,
            num_queries=1,
            use_doy=False,
            add_linear=False,
            add_boundary_loss=False,
            *args, **kwargs
    ):
        """
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
            return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to `conv_type`).
            ------------------------------
            # From here starts added arguments
            conv_type (str): Defines type of convolution used. Can be one of `2d`, `depthwise_separable`
                                (default `2d`)
            use_mbconv (bool): Whether to use MBConv blocks instead of classical conv blocks
            add_squeeze_excit (bool): Whether to add squeeze and excitation. Note that is is added only to convolutional
                                      part of encoder
            use_abs_rel_enc (bool): Whether to use both date representations: Relative and absolute (DOY)
            use_doy: (bool): Whether to use DOY instead of difference from ref. date positional encoding
            num_queries (int): Number of learnable query vectors which are at the end averaged
            add_linear (bool): Whether to add linear transformation to PositionalEncoder
            add_boundary_loss (bool): Whether to simultaneously return boundary prediction
        """
        super().__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value
        self.encoder = encoder
        self.conv_type = conv_type
        self.use_doy = False if use_abs_rel_enc else use_doy
        self.use_abs_rel_enc = use_abs_rel_enc
        self.add_linear = add_linear
        self.add_boundary_loss = add_boundary_loss

        self.use_mbconv = use_mbconv
        self.add_squeeze_excit = add_squeeze_excit

        if use_mbconv:
            in_conv_block = MBConvBlock
            down_conv_block = MBDownConvBlock
            up_conv_block = MBUpConvBlock
            out_conv_block = MBConvBlock
        else:
            in_conv_block = ConvBlock
            down_conv_block = DownConvBlock
            up_conv_block = UpConvBlock
            out_conv_block = ConvBlock

        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = in_conv_block(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            # [10, 64, 64] => 2 ConvLayers
            pad_value=pad_value,  # 0
            norm=encoder_norm,
            padding_mode=padding_mode,
            conv_type=conv_type,
            add_squeeze=True if add_squeeze_excit else False
        )

        self.spatial_reduction = nn.ModuleList(
            down_conv_block(
                d_in=encoder_widths[i],  # [64, 64, 64]
                d_out=encoder_widths[i + 1],  # [64, 64, 128]
                k=str_conv_k,  # 4
                s=str_conv_s,  # 2
                p=str_conv_p,  # 1
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
                conv_type='depthwise_separable',
                add_squeeze=True if add_squeeze_excit else False
            )
            for i in range(self.n_stages - 1)
        )

        self.down_blocks = nn.ModuleList(
            down_conv_block(
                d_in=encoder_widths[i],  # [64, 64, 64]
                d_out=encoder_widths[i + 1],  # [64, 64, 128]
                k=str_conv_k,  # 4
                s=str_conv_s,  # 2
                p=str_conv_p,  # 1
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
                conv_type=conv_type,
                add_squeeze=True if add_squeeze_excit else False
            )
            for i in range(self.n_stages - 1)
        )

        self.up_blocks = nn.ModuleList(
            up_conv_block(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,  # kernel_size 4
                s=str_conv_s,  # stride 2
                p=str_conv_p,  # padding 1
                norm="batch",
                padding_mode=padding_mode,
                conv_type="2d",  # conv_type
                add_squeeze=False
            )
            for i in range(self.n_stages - 1, 0, -1)
        )

        self.temporal_encoder = LTAE4WTAE(
            in_channels=encoder_widths[-1],  # 128
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            use_abs_rel_enc=use_abs_rel_enc,
            num_queries=num_queries,
            use_doy=self.use_doy,
            add_linear=add_linear
        )

        self.temporal_aggregator = TemporalAggregator(mode=agg_mode)

        # self.attention_drop = nn.Dropout(0.1)

        self.out_conv = out_conv_block(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode,
                                       conv_type="2d",
                                       add_squeeze=False)  # [32, 32, 20]  | 20 is number of classes in PASTIS

        if add_boundary_loss:
            self.boundary_conv = out_conv_block(nkernels=[decoder_widths[0]] + [32, 2], padding_mode=padding_mode,
                                                conv_type="2d",
                                                add_squeeze=False)

    def forward(self, input, batch_positions=None, return_att=False, *args, **kwargs):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        out = self.in_conv.smart_forward(input)
        # feature_maps = [out]

        # SPATIAL REDUCTION
        reduced = out
        for i in range(self.n_stages - 1):
            reduced = self.spatial_reduction[i].smart_forward(reduced)
            # feature_maps.append(reduced)

        # TEMPORAL ENCODER
        # out here is is new embedding (what is called attention, dot product of softmax(QK^T)V )
        # att is just softmax(QK^T)

        att = self.temporal_encoder(reduced,
                                    batch_positions=batch_positions, pad_mask=pad_mask)
        # att = torch.mean(att, 2)
        aggregated = self.temporal_aggregator(
            out, pad_mask=pad_mask, attn_mask=att
        )

        feature_maps = [aggregated]

        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # SPATIAL DECODER
        if self.return_maps:
            maps = [out]

        for i in range(self.n_stages - 1):
            out = self.up_blocks[i](out, feature_maps[-(i + 2)])
            if self.return_maps:  # skip is temporally aggregated input feature maps using attention masks
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            if self.add_boundary_loss:
                out_ = self.out_conv(out)
                out_b = self.boundary_conv(out)
                if return_att:
                    return out_, out_b, att
                if self.return_maps:
                    return out_, out_b, maps
                else:
                    return out_, out_b
            else:
                out = self.out_conv(out)
                if return_att:
                    return out, att
                if self.return_maps:
                    return out, maps
                else:
                    return out
