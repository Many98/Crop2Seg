from torch.nn import CrossEntropyLoss
from torch import Tensor
import torch.nn.functional as F


class SmoothCrossEntropy2D(CrossEntropyLoss):
    """
    Adjusted CrossEntropy to handle specific smoothing of targets on borders of regions i.e.
    we want to use it for smoothing of hard labels on edges of agricultural fields.
    """

    def __init__(self, weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
                 label_smoothing=0.1):
        self.super(CrossEntropyLoss, self).__init__(weight=weight, size_average=size_average, ignore_index=ignore_index,
                                                    reduce=reduce, reduction=reduction)
        self.ls = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input is expected of shape B x N_CLASSES x H x W
        # target is expected of shape B x H x W ... containing labels corresponding to classes
        assert input.dim() == 4, '`input` is expected to have 4 dimensions (B x N_CLASSES x H x W)'
        assert target.dim() == 3, '`target` is expected to have 3 dimensions (B x H x W)'

        # cc = (np.arange(v.max()) == v[...,None]-1).astype(int).transpose(2, 0, 1)  -> one hot
        # ot just cc = torch.nn.functional.one_hot(v).permute(2, 0, 1)
        # dd = ndimage.binary_dilation(kk, np.ones((1, 3, 3), dtype=bool))  -> dilate
        # dd = ndimage.binary_dilation(kk, [[False, True, False], [True, True, True], [False, True, False]])
        #       -> or this it depends if we choose 4-connectivity or 8-connectivity

        # eps=0.1/9  # 9 classes
        # exp_small = eps * (9-dd.sum(0))
        # exp_large = (1-exp_small)/dd.sum(0)

        # target = torch.where(torch.tensor(dd)==1, torch.tensor(exp_large), torch.tensor(eps)).sum(0)

        # we will need to this on the fly in loss func

        one_hot_target = F.one_hot(target.long(), num_classes=input.shape[1]).permute(0, 3, 1, 2)

        # 4-connectivity
        weights = torch.tensor([[0., 1., 0.],
                                [1., 1., 1.],
                                [0., 1., 0.]]).view(1, 1, 3, 3).repeat(input.shape[1], 1, 1, 1)

        # 8-connectivity
        # weights = torch.ones((input.shape[1], 1, 3, 3))

        # perform dilation
        dilated = F.conv2d(one_hot_target.float(), weights, groups=input.shape[1], padding=(1, 1)).bool().long()

        eps = self.ls / input.shape[1]

        exp_small = eps * (input.shape[1] - dilated.sum(1)) #(input.shape[1] - dilated.sum(0))
        exp_large = (1 - exp_small) / dilated.sum(1) #dilated.sum(0)

        target = torch.where(dilated.permute(1, 0, 2, 3) == 1, exp_large, eps).permute(1, 0, 2, 3)

        return self.forward(input, target)
