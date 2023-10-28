import torch
from torch import nn
from torch import Tensor


class TemporalAggregator(nn.Module):
    """
    Original (slightly modified) temporal aggregator as implemented in UTAE
    """
    def __init__(self, mode="mean"):
        super().__init__()
        self.mode = mode

    def forward(self, x: Tensor, pad_mask: None or Tensor = None, attn_mask: None or Tensor = None) -> (Tensor, None):
        # attn_mask.shape = hxBxTxHxW
        # x.shape = BxTxCxHxW
        up = nn.Upsample(
            size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":  # This is main method used in paper
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.contiguous().view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = up(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])

                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW

                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW

                # here was tested max instead of weighted mean but results were not promising
                # attn = repeat(attn, 'head b t h w -> head b t new_dim h w', new_dim=out.shape[3])
                # out = torch.gather(out, 2, attn.argmax(2, keepdim=True)).squeeze(2)  # hxBxC/hxHxW

                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out

            elif self.mode == "att_mean":  # This is Mean attention method (see paper)
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = up(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":  # This is Skip mean method simply calculate temporal mean to aggregate input feature maps
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.contiguous().view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = up(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.contiguous().view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = up(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


class TemporalAggregator3D(nn.Module):
    """
    Modification of original TemporalAggregator for testing/experimental purposes.
    Particularly use of "up-convolution" instead of simple bilinear interpolation
    was tested. To tackle problem of changing size of T dimension in every batch
    it was proposed to use 3D "up-convolution" instead of 2D
    Notes:
        Results indicate that this change does not enhance performance
    """
    def __init__(self, mode="mean"):
        super().__init__()
        self.mode = mode
        # we need to use 3D upconvolution because number of channels (time dime) is every time different
        # for attn_mask probably therefore authors used simple upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=1, out_channels=1, kernel_size=[3, 4, 4], stride=[1, 2, 2], padding=[1, 1, 1]
            ),
            # nn.ReLU(),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.Softmax(2)
        )

    def forward(self, x: Tensor, pad_mask: None or Tensor = None, attn_mask: None or Tensor = None) -> (Tensor, Tensor):
        # attn_mask.shape = hxBxTxHxW
        # x.shape = BxTxCxHxW
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":  # This is main method used in paper
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.contiguous().view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = self.up(attn.unsqueeze(1)).squeeze(1)  # we need to unsqueeze because of 3D Conv
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.contiguous().view(n_heads, b, t, *x.shape[-2:])
                attn2 = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn2[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out, attn

            elif self.mode == "att_mean":  # This is Mean attention method (see paper)
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = self.up(attn.unsqueeze(1)).squeeze(1)
                attn2 = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn2[:, :, None, :, :]).sum(dim=1)
                return out, attn
            elif self.mode == "mean":
                # This is Skip mean method simply calculate temporal mean to aggregate input feature maps
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out, None
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.contiguous().view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = self.up(attn.unsqueeze(1)).squeeze(1)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.contiguous().view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out, attn
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = self.up(attn.unsqueeze(1)).squeeze(1)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out, attn
            elif self.mode == "mean":
                return x.mean(dim=1), None