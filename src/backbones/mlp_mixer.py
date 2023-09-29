
from torch import nn, Tensor

from timm.models.layers import trunc_normal_, get_act_layer, DropPath

from torch.nn import LayerNorm


class MLPMixerLayer(nn.Module):
    def __init__(self,
                 num_tokens,
                 embed_dims,
                 token_expansion,
                 channel_expansion,
                 drop_path=0.,
                 drop_out=0.,
                 **kwargs):

        super(MLPMixerLayer, self).__init__()

        token_mix_dims = int(token_expansion * embed_dims)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.token_mixer = nn.Sequential(
            nn.Linear(num_tokens, token_mix_dims),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(token_mix_dims, num_tokens),
            nn.Dropout(drop_out)
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(channel_mix_dims, embed_dims),
            nn.Dropout(drop_out)
        )

        self.drop_path1 = DropPath(drop_prob=drop_path)
        self.drop_path2 = DropPath(drop_prob=drop_path)

        self.norm1 = LayerNorm(embed_dims)
        self.norm2 = LayerNorm(embed_dims)

    def forward(self, x):
        # x is expected to be of shape B x T x C
        x = x + self.drop_path1(
            self.token_mixer(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path2(self.channel_mixer(self.norm2(x)))
        return x


class MLPMixer(nn.Module):
    def __init__(self,
                 num_tokens,
                 embed_dims,
                 token_expansion=0.5,
                 channel_expansion=4.0,
                 depth=1,
                 drop_path=0.,
                 drop_out=0.,
                 **kwargs):
        super(MLPMixer, self).__init__()
        layers = [
            MLPMixerLayer(num_tokens, embed_dims, token_expansion, channel_expansion,
                          drop_path, drop_out)
            for _ in range(depth)
        ]
        self.layers = nn.Sequential(*layers)

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.layers(x)
