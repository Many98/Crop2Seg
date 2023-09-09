import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import reduce, rearrange, repeat

from src.backbones.squeeze_and_excitation import SqueezeAndExcitation
from src.backbones.temp_shared_block import TemporallySharedBlock, TemporallySharedBlock3D

from src.backbones.utils import experimental


class DepthwiseSeparableConv2D(nn.Module):
    """
    2D depth-wise separable convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros', stride=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   padding=padding, padding_mode=padding_mode, stride=stride,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


@experimental
class SparseConv2d(nn.Conv2d):
    """
    Ineffective implementation of sparse convolution using classical "dense" convolution

    Motivation is that we need to somehow forbid information leakage from unmasked image regions to masked image regions
    caused because of use of convolution (VIT does not have such problem)

    # TODO consider using MinkowskiEngine for sparse convolutions instead
        https://nvidia.github.io/MinkowskiEngine/demo/interop.html
    References:
        https://github.com/facebookresearch/ConvNeXt-V2
    # see also this https://github.com/keyu-tian/SparK

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        raise NotImplementedError
        #self.register_buffer('mask', self.weight.data.clone())
        #_, _, kH, kW = self.weight.size()
        #self.mask.fill_(1)
        #self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        #self.mask[:, :, kH // 2 + 1:] = 0

        # TODO we will also need to upsample masks
        # m = torch.randint(2, (16, 16)).float()
        # nn.Upsample(scale_factor=8)(m.unsqueeze(0).unsqueeze(0))

    def forward(self, x: Tensor, *args) -> Tensor:
        # we expect x of shape B x T x C x H x W
        # T dim will be handled by shared encoder which rearanges into shape BT x C x H x W
        # TODO in shared conv must be handled masking of different elements of time series
        # TODO we will need to find out how MASK_TOKEN will look like (zeros vs random)
        # first args[0] should be mask of shape BT x H x W which is of course composed only from 0 and 1
        x *= args[0].unsqueeze(1)  # mask the input
        out = super().forward(x)  # now information is leaked into masked parts
        # TODO note that after conv there can be different H and W so args[1] needs to contain different mask
        return out * args[1].unsqueeze(1)  # again apply mask to remove leaked information


class ConvLayer(nn.Module):
    """
    Building block of `ConvBlocks`
    Particularly it stacks all parts of "classical" convolution layer
    i.e. few normalizations and convolutions
    """
    def __init__(
            self,
            nkernels,
            norm="batch",
            k=3,
            s=1,
            p=1,
            n_groups=4,
            last_relu=True,
            padding_mode="reflect",
            conv_type='2d',
            add_squeeze=False
    ):
        super().__init__()
        self.conv_type = conv_type
        layers = []

        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None

        self.add_squeeze = add_squeeze

        if conv_type == 'depthwise_separable':
            conv = DepthwiseSeparableConv2D
        else:
            conv = nn.Conv2d  # otherwise classical 2d convolution is used
        for i in range(len(nkernels) - 1):
            layers.append(
                conv(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )

            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())

        if add_squeeze:
            layers.append(SqueezeAndExcitation(nkernels[i + 1]))

        self.conv = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)


class ConvLayer3D(nn.Module):
    """
    Building block of `ConvBlocks`
    Applies 3D convolution in spatial and channel dimension i.e. filter is sliding in CxHxW
    Particularly it stacks all parts of "classical" convolution layer
    i.e. few normalizations and convolutions
    """
    def __init__(
            self,
            nkernels,
            norm="batch",
            k=3,
            k_3d=3,
            s=1,
            p=1,
            n_groups=4,
            last_relu=True,
            padding_mode="reflect",
            conv_type='2d',
            add_squeeze=False
    ):
        super().__init__()
        #self.conv_type = conv_type
        layers = []

        if norm == "batch":
            nl = nn.BatchNorm3d
        elif norm == "instance":
            nl = nn.InstanceNorm3d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None

        self.add_squeeze = add_squeeze

        conv = nn.Conv3d
        for i in range(len(nkernels) - 1):
            layers.append(
                conv(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=(k_3d, k, k),
                    padding=(1, p, p),
                    stride=(1, s, s),
                    padding_mode=padding_mode,
                )
            )

            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())

        if add_squeeze:
            layers.append(SqueezeAndExcitation(nkernels[i + 1]))

        self.conv = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    """
    Building block of UTAE.
    Particularly it is convolutional block used within encoder and decoder part of UTAE.
    It is responsible for applying: classical convolution while not changing resolution of feature maps

    Note that `ConvBlock` is child of `TemporallySharedBlock` which effectively
    just merge batch B and time T dimension into one bigger batch dimension and operates
    over it with classical convolutions. This merge of B and T dimension is applied only if input id 5D
    i.e. it is time-series
    """
    def __init__(
            self,
            nkernels,
            pad_value=None,
            norm="batch",
            last_relu=True,
            padding_mode="reflect",
            conv_type='2d',
            add_squeeze=False
    ):
        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
            conv_type=conv_type,
            add_squeeze=add_squeeze
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)


class ConvBlock3D(TemporallySharedBlock3D):
    """
    Building block of UTAE.
    Particularly it is convolutional block used within encoder and decoder part of UTAE.
    It is responsible for applying: 3D convolution while not changing resolution of feature maps

    Note that `ConvBlock` is child of `TemporallySharedBlock` which effectively
    just merge batch B and time T dimension into one bigger batch dimension and operates
    over it with classical convolutions. This merge of B and T dimension is applied only if input id 5D
    i.e. it is time-series
    """
    def __init__(
            self,
            nkernels,
            pad_value=None,
            norm="batch",
            last_relu=True,
            padding_mode="reflect",
            conv_type='2d',
            add_squeeze=False
    ):
        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer3D(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
            conv_type=conv_type,
            add_squeeze=add_squeeze
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    """
    Building block of UTAE.
    Particularly it is convolutional block used within encoder part of UTAE.
    It is responsible for applying: classical convolution while downsize resolution of feature maps

    Note that `DownConvBlock` is child of `TemporallySharedBlock` which effectively
    just merge batch B and time T dimension into one bigger batch dimension and operates
    over it with classical convolutions. This merge of B and T dimension is applied only if input id 5D
    i.e. it is time-series
    """
    def __init__(
            self,
            d_in,
            d_out,
            k,
            s,
            p,
            pad_value=None,
            norm="batch",
            padding_mode="reflect",
            conv_type='2d',
            add_squeeze=False
    ):
        super().__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
            conv_type=conv_type
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
            conv_type=conv_type
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
            conv_type=conv_type
        )
        self.add_squeeze = add_squeeze

        if add_squeeze:
            self.sae = SqueezeAndExcitation(d_out)

    def forward(self, input: Tensor) -> Tensor:
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)

        out = self.sae(out) if self.add_squeeze else out

        return out


class DownConvBlock3D(TemporallySharedBlock3D):
    """
    Building block of UTAE.
    Particularly it is convolutional block used within encoder part of UTAE.
    It is responsible for applying: 3D convolution while downsize resolution of feature maps

    Note that `DownConvBlock` is child of `TemporallySharedBlock` which effectively
    just merge batch B and time T dimension into one bigger batch dimension and operates
    over it with classical convolutions. This merge of B and T dimension is applied only if input id 5D
    i.e. it is time-series
    """
    def __init__(
            self,
            d_in,
            d_out,
            k,
            k_3d,
            s,
            p,
            pad_value=None,
            norm="batch",
            padding_mode="reflect",
            conv_type='2d',
            add_squeeze=False
    ):
        super().__init__(pad_value=pad_value)
        self.down = ConvLayer3D(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            k_3d=k_3d,
            s=s,
            p=p,
            padding_mode=padding_mode,
            conv_type=conv_type
        )
        self.conv1 = ConvLayer3D(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
            conv_type=conv_type
        )
        self.conv2 = ConvLayer3D(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
            conv_type=conv_type
        )
        self.add_squeeze = add_squeeze

        if add_squeeze:
            self.sae = SqueezeAndExcitation(d_out)

    def forward(self, input: Tensor) -> Tensor:
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)

        out = self.sae(out) if self.add_squeeze else out

        return out


class UpConvBlock(nn.Module):
    """
    Building block of UTAE.
    Particularly it is convolutional block used within decoder part of UTAE.
    It is responsible for applying: 1x1 skip convolution
                                    "up-convolution" while increase resolution of feature maps
                                    and some classical convolution
    """
    def __init__(
            self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect",
            conv_type='2d', add_squeeze=False
    ):
        super().__init__()
        d = d_out if d_skip is None else d_skip

        # 1x1 convolution
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode, conv_type=conv_type
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode, conv_type=conv_type
        )

        self.add_squeeze = add_squeeze

        if add_squeeze:
            self.sae = SqueezeAndExcitation(d_out)

    def forward(self, input: Tensor, skip: Tensor) -> Tensor:
        out = self.up(input)  # this will upsamle (real) attentions (new embeddings)
        # below is concat of upsampled attentions (from last layer and then some feature maps)and skip feature map (after 1x1 conv)
        # where skip feature maps are just aggregated input feature maps using `TemporalAggregator` with attention masks

        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)

        out = self.sae(out) if self.add_squeeze else out
        return out
