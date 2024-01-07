from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor


class FocalCELoss(nn.Module):
    """
    FocalLoss copied from github.com/VSainteuf/utae-paps
    """

    def __init__(self, gamma=1.0, size_average=True, ignore_index: int = -100, weight: Optional[Tensor] = None):
        super(FocalCELoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, preds, target):
        # preds shape (B, C), target shape (B,)
        target = target.view(-1, 1)

        if preds.ndim > 2:  # e.g., (B, C, H, W)
            preds = preds.permute(0, 2, 3, 1).flatten(0, 2)

        keep = target[:, 0] != self.ignore_index
        preds = preds[keep, :]
        target = target[keep, :]

        logpt = F.log_softmax(preds, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.weight is not None:
            w = self.weight.expand_as(preds)
            w = w.gather(1, target)
            loss = -1 * (1 - pt) ** self.gamma * w * logpt
        else:
            loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
