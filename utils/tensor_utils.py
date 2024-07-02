import math

from torchvision.utils import make_grid
import numpy as np
import torch

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1), normalize=False):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 3D(C,H,W) any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    if normalize:
        tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
        tensor = (tensor - min_max[0]) / \
                 (min_max[1] - min_max[0])  # to range [0,1]
    else:
        tensor = tensor.squeeze().float().cpu()

    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)

def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.item()
        return tensors
    elif isinstance(tensors, dict):
        new_tensors = {}
        for k, v in tensors.items():
            v = tensors_to_scalars(v)
            new_tensors[k] = v
        return new_tensors
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors

def move_to_cuda(batch, device):
    # base case: object can be directly moved using `cuda` or `to`
    if callable(getattr(batch, 'cuda', None)):
        return batch.cuda(device, non_blocking=True)
    elif callable(getattr(batch, 'to', None)):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, device)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, device)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cuda(v, device)
        return batch
    return batch