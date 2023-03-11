import torch
from torch import nn
from torch import Tensor
from typing import Union
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from src.backbones.positional_encoding import PositionalEncoder
import copy
import math


class SqueezeAndExcitation(nn.Module):
    """
    Squeeze and excitation module (in channels)
    """

    def __init__(self, channel, reduction_ratio=16):
        super(SqueezeAndExcitation, self).__init__()

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


class SqueezeAndExcitationInTime(nn.Module):
    """
    Squeeze and excitation module in time (sliding window).
    Expected input has dimension B x T x C x H x W.
    It is then patched by kernel of `kernel_size` where first entry is for T (time dim)
    and second and third entry is for H and W dimensions.
    Input will be then patched and reduced (average pool) into shape (B*C*H*W) x T

    Learned weights are then used to aggregate time series in time dim
    """

    def __init__(self,
                 in_channels=128,
                 n_head=16,
                 d_k=4,
                 mlp=[256, 128],
                 dropout=0.2,
                 d_model=256,
                 T=1000,
                 positional_encoding=True,
                 kernel_size=32, reduction_ratio=2, upscale_ratio=100):
        """
        Parameters:

        """
        super(SqueezeAndExcitationInTime, self).__init__()

        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.n_head = n_head
        self.T = T

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)  # 1x1 convolution
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        # patched squeeze and excitation in time (sliding window in time dim)
        # serves as attention module in time
        # expects input in shape bhw t d

        self.sae = nn.Sequential(
            Reduce('bhw (t t2) d -> (bhw t) t2', 'mean', t2=kernel_size),
            nn.Linear(kernel_size, kernel_size // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(kernel_size // reduction_ratio, kernel_size, bias=False),
            nn.Sigmoid(),  # TODO sigmoid should be probably applied w.r.t whole time series i.e. in forward
        )

        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, batch_positions=None, pad_mask=None) -> (Tensor, Tensor):
        # TODO implement heads as well
        # TODO instead of reducing with mean in channel dim we can use classical squeeze and excitation in channel
        #  and use learned weights to reduce channels dims

        b, t, c, h, w = x.size()  # t dim is only one changing

        if t % self.kernel_size[0] != 0:
            pad_in_time = ((t // self.kernel_size[0]) + 1) * self.kernel_size[0] - t
            # x = torch.cat([x, torch.zeros((b, pad_in_time, c, h, w))], dim=1)  # pad with zeros
            # x = torch.cat([x, x[:, -(pad_in_time+1):-1, :]], dim=1)  # pad with last entry from tensor

            # pad input in time with reflection
            x = nn.ReflectionPad3d((0, 0, 0, 0, 0, pad_in_time))(x.transpose(1, 2)).transpose(1, 2)

            # TODO probably padding with zeros would perform better

        # we need only t within real time-series
        weights = self.sae(x).view(b, t, 1, h / self.kernel_size[1], w / self.kernel_size[2])[:, :t, :]

        # TODO tile it up
        return torch.matmul(weights, x)


class SqueezeAndExcitationInTime_v2(nn.Module):
    """
    Squeeze and excitation module in time  (adaptive average pool version).
    Expected input has dimension B x T x C x H x W.
    It is then patched by kernel of `kernel_size` where first entry is for T (time dim)
    and second and third entry is for H and W dimensions.
    Input will be then patched and reduced (average pool) into shape (B*C*H*W) x T

    Learned weights are then used to aggregate time series in time dim
    """

    def __init__(self,
                 in_channels=128,
                 n_head=16,
                 d_k=4,
                 mlp=[256, 128],
                 dropout=0.2,
                 d_model=256,
                 T=1000,
                 positional_encoding=True,
                 adaptive_seq_len=64, reduction_ratio=16, upscale_ratio=100):
        """
        Parameters:

        """
        super(SqueezeAndExcitationInTime_v2, self).__init__()

        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.n_head = n_head
        self.d_hidden = d_k
        self.T = T
        self.adaptive_seq_len = adaptive_seq_len

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)  # 1x1 convolution
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.fc_in = nn.Linear(self.d_model, self.n_head * self.d_hidden)
        nn.init.normal_(self.fc_in.weight, mean=0, std=math.sqrt(2.0 / d_k))

        self.fc_out = nn.Linear(self.n_head * self.d_hidden, self.d_model)

        # squeeze and excitation in time (with adaptive average pool)
        # serves as attention module in time
        # expects input in shape bhw t d

        self.sae = nn.Sequential(
            Reduce('n_headbhw t d -> n_headbhw t', 'mean'),
            # mean reduce in channel dim  # TODO we can use classical excit module to reduce it
            nn.AdaptiveAvgPool1d(adaptive_seq_len),
            nn.Linear(adaptive_seq_len, adaptive_seq_len // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(adaptive_seq_len // reduction_ratio, adaptive_seq_len, bias=False),
        )

        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, batch_positions=None, pad_mask=None) -> (Tensor, Tensor):
        # TODO implement heads as well
        # TODO instead of reducing with mean in channel dim we can use classical squeeze and excitation in channel
        #  and use learned weights to reduce channels dims
        b, seq_len, d, h, w = x.size()  # d=128 by default

        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                    .repeat((1, 1, h))
                    .unsqueeze(-1)
                    .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(b * h * w, seq_len)  # BHW x T
            )

            # TODO this is because of heads
            pad_mask = pad_mask.repeat(
                (self.n_head, 1)
            )  # replicate pad_mask for each head (n_head*B*H*W) x T

        # this expects x in shape B x T x d x H x W
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                    .repeat((1, 1, h))
                    .unsqueeze(-1)
                    .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        sz_b, seq_len, _, _, _ = x.size()

        # out has now shape (B*H*W) x T x d_in; d_in=256

        # TODO for now without heads | This is critical I think
        out = self.fc_in(out)  # out has now shape (B*H*W) x T x n_head*d_hidden; d_hidden=4

        # TODO if we let out be in shape (B*H*W) x T x n_head x d_hidden then sae will reduce it
        #  to BHW x adaptive_seq_len so no n_heads are no left
        #  therefore probably firstly reshape to (B*n_head*H*W) x T x d_hidden
        #   then perform sae which returns shape (B*n_head*H*W) x adaptive_seq_len
        # attn = self.sae(out)  # returns attention mask of shape BHW x adaptive_seq_len

        out = rearrange(out, 'bhw t (n_head d_hidden) -> (n_head bhw) t d_hidden', d_hidden=self.d_hidden,
                        n_head=self.n_head)
        attn = self.sae(out)  # returns attention mask of shape n_head*B*H*W x adaptive_seq_len

        # adaptive average back to T dim
        attn = nn.AdaptiveAvgPool1d(seq_len)(attn)  # BHWn*head x T

        # normalize weights
        attn = nn.Sigmoid()(attn)

        attn = attn.masked_fill(pad_mask, -1e6)  # pad_mask shape (n_head*B*H*W) x T TODO why do we do masked fill ??

        # n_head*BHW x 1 x T  X  n_head*BHW T d_hidden
        out = torch.matmul(attn.unsqueeze(1), out)  # n_head*BHW x 1 x d_hidden

        out = rearrange(out, '(n_head bhw) t d_hidden -> bhw t (n_head d_hidden)', n_head=self.n_head).squeeze(1)

        out = self.fc_out(out)  # BHW x d_in

        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out

        '''
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)
        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w
        '''

        out = rearrange(out, '(b h w) d -> b d h w', b=b, h=h, w=w)

        attn = rearrange(attn, '(b h w n_head) t -> n_head b t h w', n_head=self.n_head, b=b, h=h, w=w)
        # attn is attention mask
        # out is new representation of input (new embedding)

        return out, attn


class SqueezeAndExcitationInTime_v3(nn.Module):
    """
    Squeeze and excitation module in time  (fixed linear version).
    Expected input has dimension B x T x C x H x W.
    It is then patched by kernel of `kernel_size` where first entry is for T (time dim)
    and second and third entry is for H and W dimensions.
    Input will be then patched and reduced (average pool) into shape (B*C*H*W) x T

    Learned weights are then used to aggregate time series in time dim
    """

    def __init__(self,
                 in_channels=128,
                 n_head=16,
                 d_k=4,
                 mlp=[256, 128],
                 dropout=0.2,
                 d_model=256,
                 positional_encoding=True,
                 T=128, reduction_ratio=16, upscale_ratio=100):
        """
        Parameters:

        """
        super(SqueezeAndExcitationInTime_v3, self).__init__()

        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.n_head = n_head
        self.T = T

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)  # 1x1 convolution
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        # squeeze and excitation in time (with adaptive average pool)
        # expects input in shape bhw t c
        self.sae = nn.Sequential(
            Rearrange('bhw t c -> bhw c t'),
            Reduce('bhw c t -> bhw t', 'mean'),  # mean reduce in channel dim
            nn.Linear(self.T, self.T // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.T // reduction_ratio, self.T, bias=False),
        )

        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, batch_positions=None, pad_mask=None) -> Tensor:
        # TODO implement heads as well
        # TODO instead of reducing with mean in channel dim we can use classical squeeze and excitation in channel
        #  and use learned weights to reduce channels dims

        sz_b, t, c = x.size()

        # now pad it to size of T with zeros


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        # x: [B x T x input_size]
        x_proj = self.fc1(x)  # [B x T x hidden_size]
        scores = self.fc2(torch.tanh(x_proj))  # [B x T x 1]
        weights = torch.softmax(scores, dim=1)  # [B x T x 1]
        attended = torch.sum(weights * x, dim=1)  # [B x input_size]
        attended = attended.unsqueeze(1)  # [B x 1 x input_size]
        return attended, weights


class MultiHeadAttentionV1(nn.Module):
    """
    another attention produced by ChatGPT
    """
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiHeadAttentionV1, self).__init__()
        self.num_heads = num_heads
        self.linear_in = nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.linear_out = nn.Linear(hidden_dim * num_heads, input_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, pad_mask: Tensor = None, return_comp: bool = False) -> (Tensor, Tensor):
        batch_size, seq_len, input_dim = x.size()  # BHW x T x d_in

        # Apply the learned linear projection to create multiple "heads"
        heads = self.linear_in(x).contiguous().view(batch_size, seq_len, self.num_heads, -1)  # BHW x T x num_heads x d_hidden

        # Compute the attention scores for each head
        query = torch.tanh(heads)  # BHW x T x num_heads x d_hidden
        key = torch.tanh(heads)  # BHW x T x num_heads x d_hidden
        energy = torch.sum(query * key, dim=-1)  # BHW x T x num_heads
        attn = self.softmax(energy)  # BHW x T x num_head

        attn = rearrange(attn, 'bhw t n_head -> (n_head bhw) t')

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (self.num_heads, 1)
            )  # replicate pad_mask for each head (n_head*B*H*W) x T

            attn = attn.masked_fill(pad_mask, -1e6)

        attn = rearrange(attn, '(n_head bhw) t -> bhw t n_head', n_head=self.num_heads, bhw=batch_size)  # BHW x T x num_heads

        # Apply the attention scores to the input sequence and concatenate the heads
        weighted_heads = attn.unsqueeze(-1) * heads
        weighted_sum = weighted_heads.sum(dim=1).contiguous().view(batch_size, -1)  # BHW x d_hidden * num_heads

        # Linearly transform and output the concatenated heads
        output = self.linear_out(weighted_sum)  # BHW x d_in

        return output, rearrange(attn, 'bhw t n_head -> n_head bhw t')  # rearrange attn to be consistent with LTAE


class MultiHeadAttentionV2(nn.Module):
    """
    Actually it is mean/mean reduction
    """

    def __init__(self, d_in, hidden_size, num_heads):
        super(MultiHeadAttentionV2, self).__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(d_in, hidden_size * num_heads)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size * num_heads, d_in)

    def forward(self, v: Tensor, pad_mask=None, return_comp=False) -> (Tensor, Tensor):
        # x: BHW x T x d_in
        batch_size, seq_len, d_in = v.size()
        heads = self.fc1(v).view(batch_size, seq_len, self.num_heads, -1)  # BHW x T x num_heads x hidden_size

        scores = self.fc2(torch.tanh(heads))  # BHW x T x num_heads x 1
        attn = torch.softmax(scores, dim=1)  # BHW x T x num_heads x 1

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (self.num_heads, 1)
            )  # replicate pad_mask for each head (n_head*B*H*W) x T

            pad_mask = rearrange(pad_mask, '(n_head bhw) t -> bhw t n_head ()', n_head=self.num_heads, bhw=batch_size)
            # pad_mask shape before is (n_head*B*H*W) x T
            attn = attn.masked_fill(pad_mask, -1e6)

        # this uses directly input x
        # out = torch.sum(attn * x.unsqueeze(2), dim=1)  # [B x num_heads x d_in]

        # or we can use
        out = torch.sum(attn * heads, dim=1)  # [B x num_heads x hidden_size]

        out = out.contiguous().view(batch_size, self.num_heads * self.hidden_size)  # [B x num_heads*hidden_size]

        out = self.fc_out(out)  # [B x d_in]

        return out, attn.squeeze(-1).transpose(0, 1)


class MultiHeadAttentionV3(nn.Module):
    """
    Another additive attention type generated by chatGPT
    """
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttentionV3, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        #self.head_size = hidden_size // num_heads

        self.query_projection = nn.Linear(input_size, hidden_size * num_heads, bias=False)
        self.key_projection = nn.Linear(input_size, hidden_size * num_heads, bias=False)
        self.value_projection = nn.Linear(input_size, hidden_size * num_heads, bias=False)
        self.output_projection = nn.Linear(hidden_size * num_heads, input_size, bias=False)

        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: Tensor, pad_mask=None, return_comp=False) -> (Tensor, Tensor):
        batch_size, seq_len, input_dim = x.size()  # BHW x T x d_in

        query = self.query_projection(x)  # BHW x T x d_hidden * num_heads
        key = self.key_projection(x)  # BHW x T x d_hidden * num_heads
        value = self.value_projection(x)  # BHW x T x d_hidden * num_heads

        # Split the query, key, and value projections into `num_heads` parts
        query = query.view(batch_size, -1, self.num_heads, self.hidden_size)  # BHW x T x num_heads x d_hidden
        key = key.view(batch_size, -1, self.num_heads, self.hidden_size)  # BHW x T x num_heads x d_hidden
        value = value.view(batch_size, -1, self.num_heads, self.hidden_size)  # BHW x T x num_heads x d_hidden

        # Transpose the dimensions for the matrix multiplication
        query = query.transpose(1, 2)  # BHW x num_heads x T x d_hidden
        key = key.transpose(1, 2)  # BHW x num_heads x T x d_hidden
        value = value.transpose(1, 2)  # BHW x num_heads x T x d_hidden

        # Compute the dot product of the query and key vectors
        energy = torch.tanh(query + key)  # BHW x num_heads x T x d_hidden  # TODO not sure if here should be +
        energy = self.energy_layer(energy)  # BHW x num_heads x T x 1

        # Apply softmax to get the attention weights
        attn = torch.softmax(energy, dim=2)  # BHW x num_heads x T x 1

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (self.num_heads, 1)
            )  # replicate pad_mask for each head (n_head*B*H*W) x T

            attn = attn.masked_fill(pad_mask.unsqueeze(-1), -1e6)

        # Compute the context vector as the weighted sum of the value vectors
        output = (attn * value).sum(dim=2)  # BHW x num_heads x d_hidden

        # Concatenate the heads and project the result
        output = output.transpose(1, 2)  # BHW x d_hidden x num_heads
        output = output.contiguous().view(batch_size, -1)  # BHW x d_hidden * num_heads
        output = self.output_projection(output)  # BHW x d_in

        return output, rearrange(attn, 'bhw n_head t x -> n_head bhw (t x)')  # x=1
