import torch
from torch import nn
from torch import Tensor

from src.backbones.temp_shared_block import TemporallySharedBlock
from src.backbones.conv import ConvLayer
from src.backbones.squeeze_and_excitation import SqueezeAndExcitation


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x += res
        return x


class MBConv(nn.Sequential):
    """
    Implementation of MBConv block according to
        https://github.com/FrancescoSaverioZuppichini/BottleNeck-InvertedResidual-FusedMBConv-in-PyTorch

    Basically MBConv exploits inverted residual blocks and linear bottleneck
    """

    def __init__(self, in_channels: int, out_channels: int, expansion: int = 4, n_groups: int = 4,
                 add_squeeze: bool = True,
                 norm: str = 'group'):

        residual = ResidualAdd if in_channels == out_channels else nn.Sequential

        expanded_channels = in_channels * expansion

        # squeeze and excitation
        sae = SqueezeAndExcitation(expanded_channels, reduction_ratio=16) if add_squeeze else nn.Identity()

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
            nl = nn.Identity

        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        # narrow -> wide  # 1x1 conv
                        nn.Conv2d(
                            in_channels,
                            expanded_channels,
                            kernel_size=1,
                        ),
                        nl(expanded_channels),  # normalization
                        nn.ReLU(),  # TODO maybe Relu6 replace with Relu

                        # wide -> wide
                        nn.Conv2d(
                            expanded_channels,
                            expanded_channels,
                            groups=expanded_channels,
                            kernel_size=3,
                            padding=1,
                            padding_mode='reflect'
                        ),
                        nl(expanded_channels),  # normalization
                        nn.ReLU(),  # TODO maybe Relu6 replace with Relu

                        # Squeeze and excitation block
                        sae,

                        # wide -> narrow
                        nn.Conv2d(
                            expanded_channels,
                            out_channels,
                            kernel_size=1,
                        ),
                        nl(out_channels),  # normalization

                    ),
                ),
                #nn.ReLU(),  # TODO maybe Relu remove
            )
        )


class MBConvLayer(nn.Module):
    """
    Alternative to ConvLayer used in UTAE
    based on MBConv module
    """

    def __init__(
            self,
            nkernels,
            norm

    ):
        super(MBConvLayer, self).__init__()
        layers = []

        for i in range(len(nkernels) - 1):
            layers.append(
                MBConv(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    expansion=4,
                    norm=norm
                )
            )

        self.conv = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)


class MBConvBlock(TemporallySharedBlock):
    """
    Alternative implementation of temporally shared ConvBlock based on MBConv block
    """
    def __init__(
            self,
            nkernels,
            pad_value=None,
            norm="group",
            *args,
            **kwargs
    ):
        super(MBConvBlock, self).__init__(pad_value=pad_value)
        self.conv = MBConvLayer(
            nkernels=nkernels,
            norm=norm
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)


class MBDownConvBlock(TemporallySharedBlock):
    """
    Alternative implementation of temporally shared DownConvBlock based on MBConv block
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
            *args,
            **kwargs
    ):
        super(MBDownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(  # here we use "normal" convlayer for downsampling
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
            conv_type=conv_type
        )
        self.conv1 = MBConvLayer(
            nkernels=[d_in, d_out],
            norm=norm
        )
        self.conv2 = MBConvLayer(
            nkernels=[d_out, d_out],
            norm=norm
        )

    def forward(self, input: Tensor) -> Tensor:
        out = self.down(input)
        out = self.conv1(out)
        out = self.conv2(out)

        return out


class MBUpConvBlock(nn.Module):
    """
    Alternative implementation of temporally shared UpConvBlock based on MBCOnv block
    """
    def __init__(
            self, d_in, d_out, k, s, p, d_skip=None, norm="batch", *args,
            **kwargs
    ):
        super(MBUpConvBlock, self).__init__()
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
        self.conv1 = MBConvLayer(
            nkernels=[d_out + d, d_out],
            norm=norm
        )
        self.conv2 = MBConvLayer(
            nkernels=[d_out, d_out],
            norm=norm
        )

    def forward(self, input: Tensor, skip: Tensor) -> Tensor:
        out = self.up(input)  # this will upsamle (real) attentions (new embeddings)
        # below is concat of upsampled attentions (from last layer and then some feature maps)and skip feature map (after 1x1 conv)
        # where skip feature maps are just aggregated input feature maps using `TemporalAggregator` with attention masks

        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = self.conv2(out)

        return out
