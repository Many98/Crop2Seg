from torch.nn import CrossEntropyLoss
from torch import Tensor
import torch.nn.functional as F
import torch


class SmoothCrossEntropy2D(CrossEntropyLoss):
    """
    Adjusted CrossEntropy to handle specific smoothing of targets on borders of regions i.e.
    we want to use it for smoothing of hard labels on edges of agricultural fields.
    """

    def __init__(self, weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
                 label_smoothing=0.1, background_index=0, background_label_value=0.5,
                 class_proportions=(0.3111, 0.0193, 0.0809, 0.2809, 0.1084, 0.0892, 0.0350, 0.0170, 0.0007,
                                    0.0047, 0.0015, 0.0044, 0.0394, 0.0074)):
        super().__init__(weight=weight, size_average=size_average, ignore_index=ignore_index,
                         reduce=reduce, reduction=reduction)
        self.ls = label_smoothing
        self.background_index = background_index
        self.background_label_value = background_label_value
        self.class_proportions = class_proportions

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input is expected of shape B x N_CLASSES x H x W
        # target is expected of shape B x H x W ... containing labels corresponding to classes
        assert input.dim() == 4, f'`input` is expected to have 4 dimensions (B x N_CLASSES x H x W) but is of ' \
                                 f'shape {input.shape}'
        assert target.dim() == 3, f'`target` is expected to have 3 dimensions (B x H x W) but is of shape {target.shape}'

        one_hot_target = F.one_hot(target.long(), num_classes=input.shape[1]).permute(0, 3, 1, 2)

        # 4-connectivity
        weights = torch.tensor([[0., 1., 0.],
                                [1., 1., 1.],
                                [0., 1., 0.]]).view(1, 1, 3, 3).repeat(input.shape[1], 1, 1, 1)

        # TODO consider 8-connectivity
        # weights = torch.ones((input.shape[1], 1, 3, 3))

        # perform dilation
        dilated = F.conv2d(one_hot_target.float(), weights, groups=input.shape[1], padding=(1, 1)).bool().long()

        eps = self.ls / input.shape[1]

        exp_small = eps * (input.shape[1] - dilated.sum(1))  # (input.shape[1] - dilated.sum(0))
        exp_large = (1 - exp_small) / dilated.sum(1)  # dilated.sum(0)

        target_out = torch.where(dilated.permute(1, 0, 2, 3) == 1, exp_large, eps).permute(1, 0, 2, 3)

        # here added special value for backround class because background can contain other crop types
        # therefore added 
        distrib = list(self.class_proportions * self.background_label_value)
        distrib = [self.background_label_value] + distrib
        background_distribution = torch.tensor(distrib)
        target_out = torch.where(target == self.background_index, background_distribution[None, ...], target_out)

        return super().forward(input, target_out)
