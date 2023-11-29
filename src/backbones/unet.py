import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.models.layers import trunc_normal_, get_act_layer

from src.backbones.conv import ConvBlock, UpConvBlock, DownConvBlock
from src.backbones.mbconv import MBConvBlock, MBUpConvBlock, MBDownConvBlock

from einops import rearrange


class Unet(nn.Module):
    """
    U-Net segmentation model
    """

    def __init__(
            self,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 20],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            encoder_norm="group",
            encoder=False,
            return_maps=False,
            pad_value=0,
            padding_mode="reflect",
            # ------------------------------
            # From here starts added arguments
            conv_type='2d',
            use_mbconv=False,
            add_squeeze_excit=False,
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
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
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

        self.use_mbconv = use_mbconv
        self.add_squeeze_excit = add_squeeze_excit

        if use_mbconv:
            down_conv_block = MBDownConvBlock
            up_conv_block = MBUpConvBlock
            out_conv_block = MBConvBlock
        else:
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

        # self.attention_drop = nn.Dropout(0.1)

        self.out_conv = out_conv_block(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode,
                                       conv_type="2d",
                                       add_squeeze=False)  # [32, 32, 20]  | 20 is number of classes in PASTIS

    def forward(self, input, *args, **kwargs):

        feature_maps = [input]

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
            out = self.out_conv(out)

            if self.return_maps:
                return out, maps
            else:
                return out


class Unet_naive(nn.Module):
    """
    Naive U-Net segmentation model which treats temporal dimension as new spectral dimension.
    No temporal order is utilized
    Note that this implementation can be very memory consuming
    TODO fix it for training because we need to work it with changing value for time dimension
    """

    def __init__(
            self,
            input_dim,
            temporal_length=61,
            encoder_widths=[8, 8, 8, 16],
            decoder_widths=[4, 4, 8, 16],
            out_conv=[4, 20],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            encoder=False,
            return_maps=False,
            pad_value=0,
            padding_mode="reflect",
            # ------------------------------
            # From here starts added arguments
            conv_type='2d',
            use_mbconv=False,
            add_squeeze_excit=False,
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
            temporal_length: int
                Fixed (maximal) length of time series
        """
        super().__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        encoder_widths = [i * temporal_length // 2 for i in encoder_widths]
        self.encoder_widths = encoder_widths
        decoder_widths = [i * temporal_length // 2 for i in decoder_widths]
        self.decoder_widths = decoder_widths
        out_conv[0] = out_conv[0] * temporal_length
        self.temporal_length = temporal_length
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value
        self.encoder = encoder
        self.conv_type = conv_type

        self.input_dim = input_dim

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
            nkernels=[input_dim * temporal_length] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,  # 0
            norm='batch',
            padding_mode=padding_mode,
            conv_type=conv_type,
            add_squeeze=True if add_squeeze_excit else False
        )

        self.down_blocks = nn.ModuleList(
            down_conv_block(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,  # 4
                s=str_conv_s,  # 2
                p=str_conv_p,  # 1
                pad_value=pad_value,
                norm='batch',
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

        self.out_conv = out_conv_block(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode,
                                       conv_type="2d",
                                       add_squeeze=False)

    def forward(self, input, *args, **kwargs):
        if input.shape[1] < self.temporal_length:
            input = torch.cat([input, torch.zeros(input.shape[0], self.temporal_length - input.shape[1],
                                                  *input.shape[2:]).to(input.device)], dim=1)
        out = self.in_conv.smart_forward(rearrange(input, 'b t c h w -> b (t c) h w'))

        feature_maps = [out]

        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i](feature_maps[-1])
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
            out = self.out_conv(out)

            if self.return_maps:
                return out, maps
            else:
                return out


# ---------------------------------------------------------------------
# MODELS FROM EXCHANGER https://github.com/TotalVariation/Exchanger4SITS/
# ---------------------------------------------------------------------


class ConvModule_ex(nn.Module):
    """
    ConvModule used in https://github.com/TotalVariation/Exchanger4SITS
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            norm_layer=nn.BatchNorm2d,
            act_type='relu',
    ):
        super(ConvModule_ex, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.norm = norm_layer(out_channels)
        self.act = get_act_layer(act_type)()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BasicConvBlock_ex(nn.Module):
    """
    BasicConvBlock used in https://github.com/TotalVariation/Exchanger4SITS
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            num_convs=2,
            stride=1,
            dilation=1,
            norm_layer=nn.BatchNorm2d,
            act_type='relu',
    ):
        super(BasicConvBlock_ex, self).__init__()

        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule_ex(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    norm_layer=norm_layer,
                    act_type=act_type
                )
            )

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        out = self.convs(x)
        return out


class DeconvModule_ex(nn.Module):
    """
    DeconvModule used in https://github.com/TotalVariation/Exchanger4SITS
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            norm_layer=nn.BatchNorm2d,
            act_type='relu',
            *,
            kernel_size=4,
            scale_factor=2
    ):
        super(DeconvModule_ex, self).__init__()

        assert (kernel_size - scale_factor >= 0) and \
               (kernel_size - scale_factor) % 2 == 0, \
            f'kernel_size should be greater than or equal to scale_factor ' \
            f'and (kernel_size - scale_factor) should be even numbers, ' \
            f'while the kernel size is {kernel_size} and scale_factor is ' \
            f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        norm = norm_layer(out_channels)
        activate = get_act_layer(act_type)()

        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        out = self.deconv_upsamping(x)
        return out


class InterpConv_ex(nn.Module):
    """
    InterpConv used in https://github.com/TotalVariation/Exchanger4SITS
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer=nn.BatchNorm2d,
                 act_type='relu',
                 *,
                 conv_first=False,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 upsample_cfg=dict(
                     scale_factor=2, mode='bilinear', align_corners=False)):
        super(InterpConv_ex, self).__init__()

        conv = ConvModule_ex(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
            act_type=act_type
        )
        upsample = nn.Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        out = self.interp_upsample(x)
        return out


class UpConvBlock_ex(nn.Module):
    """
    UpConvBlock used in https://github.com/TotalVariation/Exchanger4SITS for U-Net
    """
    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 norm_layer=nn.BatchNorm2d,
                 act_type='relu',
                 upsample_layer=InterpConv_ex,
                 ):
        super(UpConvBlock_ex, self).__init__()

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
            act_type=act_type,
        )

        self.upsample = upsample_layer(
            in_channels=in_channels,
            out_channels=skip_channels,
            norm_layer=norm_layer,
            act_type=act_type
        )

    def forward(self, skip, x):
        """Forward function."""

        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)

        return out


class UNet_ex(nn.Module):
    """
    U-Net used in https://github.com/TotalVariation/Exchanger4SITS
    """
    def __init__(self, in_channels: int, base_channels: int = 64, num_stages: int=4, strides: list = [1, 1, 1, 1],
                 enc_num_convs: list = [2, 2, 2, 2], dec_num_convs: list = [2, 2, 2],
                 downsamples: list = [True, True, True], enc_dilations: list = [1, 1, 1, 1],
                 dec_dilations: list = [1, 1, 1], norm_type: str = 'bn', act_type: str = 'gelu',
                 upsample_type: str = 'interp', **kwargs):
        super(UNet_ex, self).__init__()

        in_channels = in_channels
        base_channels = base_channels
        num_stages = num_stages
        strides = strides
        enc_num_convs = enc_num_convs
        dec_num_convs = dec_num_convs
        downsamples = downsamples
        enc_dilations = enc_dilations
        dec_dilations = dec_dilations
        norm_type = norm_type
        act_type = act_type
        upsample_type = upsample_type

        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages, ' \
            f'while the strides is {strides}, the length of ' \
            f'strides is {len(strides)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(enc_num_convs) == num_stages, \
            'The length of enc_num_convs should be equal to num_stages, ' \
            f'while the enc_num_convs is {enc_num_convs}, the length of ' \
            f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(dec_num_convs) == (num_stages - 1), \
            'The length of dec_num_convs should be equal to (num_stages-1), ' \
            f'while the dec_num_convs is {dec_num_convs}, the length of ' \
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(downsamples) == (num_stages - 1), \
            'The length of downsamples should be equal to (num_stages-1), ' \
            f'while the downsamples is {downsamples}, the length of ' \
            f'downsamples is {len(downsamples)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(enc_dilations) == num_stages, \
            'The length of enc_dilations should be equal to num_stages, ' \
            f'while the enc_dilations is {enc_dilations}, the length of ' \
            f'enc_dilations is {len(enc_dilations)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(dec_dilations) == (num_stages - 1), \
            'The length of dec_dilations should be equal to (num_stages-1), ' \
            f'while the dec_dilations is {dec_dilations}, the length of ' \
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is ' \
            f'{num_stages}.'

        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.base_channels = base_channels

        if norm_type == 'bn':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.out_dims = []

        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock_ex(
                        conv_block=BasicConvBlock_ex,
                        in_channels=base_channels * 2 ** i,
                        skip_channels=base_channels * 2 ** (i - 1),
                        out_channels=base_channels * 2 ** (i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        norm_layer=norm_layer,
                        act_type=act_type,
                        upsample_layer=InterpConv_ex if upsample_type == 'interp' else DeconvModule_ex,
                    )
                )

            enc_conv_block.append(
                BasicConvBlock_ex(
                    in_channels=in_channels,
                    out_channels=base_channels * 2 ** i,
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    norm_layer=norm_layer,
                    act_type=act_type,
                ))

            self.encoder.append((nn.Sequential(*enc_conv_block)))
            in_channels = base_channels * 2 ** i
            self.out_dims.append(in_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        self._check_input_divisible(x)
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        return dec_outs

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
               and (w % whole_downsample_rate == 0), \
            f'The input image size {(h, w)} should be divisible by the whole ' \
            f'downsample rate {whole_downsample_rate}, when num_stages is ' \
            f'{self.num_stages}, strides is {self.strides}, and downsamples ' \
            f'is {self.downsamples}.'
