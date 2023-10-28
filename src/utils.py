import collections.abc
import re
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils import data

from torchvision.transforms.functional import hflip, vflip, rotate, crop
from torchvision.transforms import RandomCrop

import warnings

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate(batch, pad_value=0):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Transform(torch.nn.Module):
    def __init__(self, add_noise: bool = False, crop: bool = False, crop_size: int = 64):
        super().__init__()

        self.add_noise = add_noise
        self.crop = crop
        self.crop_size = crop_size

    def __call__(self, img, mask, weight=None):
        deg = np.random.choice([-180, -150, -120, -90, -75, -45, -25, -10, 0, 0, 0,
                                0, 10, 25, 45, 75, 90, 120, 150, 180], 1)
        flip = np.random.choice([0, 1, 2], 1)

        if self.add_noise and np.random.sample() > 0.5:
            img = img + 0.01*torch.randn(img.shape)

        if flip == 1:
            img = hflip(img)
            mask = hflip(mask)
        elif flip == 2:
            img = vflip(img)
            mask = vflip(mask)

        img = rotate(img, deg)
        mask = rotate(mask[None, :], deg)[0]

        if self.crop:
            if weight > 4:  # means lots of minority classes
                img = crop(img, top=0, left=0, height=self.crop_size, width=self.crop_size)
                mask = crop(mask, top=0, left=0, height=self.crop_size, width=self.crop_size)
            else:

                img = RandomCrop(size=self.crop_size, )(img)
                mask = RandomCrop(size=self.crop_size, )(mask)

        return img, mask

