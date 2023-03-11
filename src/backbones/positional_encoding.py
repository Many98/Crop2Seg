import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
from einops import rearrange


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d_model
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d_model).float() // 2) / d_model
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table  # B x T x d_model  where B = b*h*w


class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, d_model: int, T: int = 100, repeat=None):
        super(AbsolutePositionalEncoder, self).__init__()

        self.d = d_model
        self.repeat = repeat

        self.fc = nn.Linear(365, d_model)

    def forward(self, batch_positions):
        # here we expect batch_position to provide info about day of year
        #  TODO probably we should consider year with 366 days but whatever
        #  input shape B x T
        b, t = batch_positions.size()
        bp = rearrange(batch_positions, 'b t -> (b t)')
        bp = F.one_hot(bp.to(torch.int64), num_classes=365).to(torch.float32)  # output shape B x T x 365
        bp = rearrange(bp, '(b t) x-> b t x', b=b, t=t)

        pos_emb = self.fc(bp)

        if self.repeat is not None:
            pos_emb = torch.cat(
                [pos_emb for _ in range(self.repeat)], dim=-1
            )

        return pos_emb  # B x T x d_model


class LearnedPositionalEncoder(object):
    def __init__(self, d_model: int, T: int = 100, repeat=None):
        """
        Positional embeddings are learned rather than predefined

        d_model: dimension of embeddings
        T: maximal length of sequence to be learned
        """
        super(LearnedPositionalEncoder).__init__()

        self.d = d_model
        self.T = T
        self.repeat = repeat

        # TODO check dmensions
        self.positional_encodings = nn.Parameter(torch.zeros(1, T, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:, :x.shape[1], :]

        return x + einops.repeat(pe, 't d_model -> new_dim t d_model', new_dim=x.shape[0])
