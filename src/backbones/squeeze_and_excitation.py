from torch import nn

from torch import Tensor
from einops.layers.torch import Rearrange, Reduce


class SqueezeAndExcitation(nn.Module):
    """
    Original (kind of) implementation of squeeze & excitation module i.e. squeeze & excitation is
    performed within channel dimension C
    """

    def __init__(self, channel, reduction_ratio=16):
        super().__init__()

        self.sae = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.sae(x)
        return x * y.expand_as(x)
