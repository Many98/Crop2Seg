import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PositionalEncoder(nn.Module):
    """
    Original implementation of positional encoder used in UTAE
    """
    def __init__(self, d_model, T=1000, repeat=None, offset=0, add_linear=False):
        super().__init__()
        self.d = d_model
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d_model).float() // 2) / d_model
        )
        self.updated_location = False
        self.add_linear = add_linear

        if add_linear:
            self.fc = nn.Linear(d_model * repeat, d_model * repeat) if repeat is not None else nn.Linear(d_model, d_model)

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
        if self.add_linear:
            pos_emb = self.fc(sinusoid_table)
            return pos_emb  # B x T x d_model  where B = b*h*w
        else:
            return sinusoid_table  # B x T x d_model  where B = b*h*w


class AbsolutePositionalEncoder(nn.Module):
    """
    Alternative implementation of positional encoder using absolute encoding i.e. number of day within year
    """
    def __init__(self, d_model: int, repeat=None):
        super().__init__()

        self.d = d_model
        self.repeat = repeat

        self.fc = nn.Linear(365, d_model)

    def forward(self, batch_positions):
        # here we expect batch_position to provide info about day of year
        #  input shape B x T
        b, t = batch_positions.size()
        bp = rearrange(batch_positions, 'b t -> (b t)')
        bp = F.one_hot(bp.to(torch.int64), num_classes=365).to(torch.float32)  # output shape (B*T) x 365
        bp = rearrange(bp, '(b t) x-> b t x', b=b, t=t)

        pos_emb = self.fc(bp)

        if self.repeat is not None:
            pos_emb = torch.cat(
                [pos_emb for _ in range(self.repeat)], dim=-1
            )

        return pos_emb  # B x T x d_model
