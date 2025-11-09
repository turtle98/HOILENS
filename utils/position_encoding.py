# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
from torch import Tensor


def compute_sinusoidal_pe(pos_tensor: Tensor, temperature: float = 10000, dims = 64) -> Tensor:
    """
    Compute positional embeddings for points or bounding boxes

    Parameters:
    -----------
    pos_tensor: Tensor
        Coordinates of 2d points (x, y) normalised to (0, 1). The shape is (n_q, bs, 2).
    temperature: float, Default: 10000.
        The temperature parameter in sinusoidal functions.

    Returns:
    --------
    pos: Tensor
        Sinusoidal positional embeddings of shape (n_q, bs, 128).
    """
    scale = 2 * math.pi
    dim_t = torch.arange(dims, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / dims)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos