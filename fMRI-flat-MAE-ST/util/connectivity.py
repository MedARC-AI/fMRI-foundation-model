from pathlib import Path

import numpy as np
import torch
from torch import nn

from util.hcp_flat import load_hcp_flat_mask


class Connectome(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("mask", load_hcp_flat_mask())
        self.register_buffer("parc", load_schaefer200())

    def forward(self, x: torch.Tensor):
        return connectome(x, self.parc, self.mask)

    forward_features = forward


def connectome(
    x: torch.Tensor, parc: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute batch pearson function connectome.

    Args:
        x: (N, 1, T, H, W)
        parc: (H, W), values in 0, ..., R
        mask: (H, W)

    Returns:
        conn: (N, R * (R-1) / 2)
    """
    # x: (N, T, L)
    x = x[..., mask]
    x = x.flatten(1, -2)
    # parc: (L,)
    parc = parc[mask]

    # one_hot: (L, R)
    n_rois = parc.max()
    one_hot = (parc[:, None] == torch.arange(1, n_rois + 1, device=x.device)).float()
    one_hot = one_hot / one_hot.sum(dim=0)

    # x: (N, T, R)
    x = x @ one_hot
    x = x - x.mean(dim=1, keepdim=True)
    x = x / (torch.linalg.norm(x, dim=1, keepdim=True) + 1e-8)

    # conn: (N, R, R)
    conn = x.transpose(1, 2) @ x
    row, col = torch.triu_indices(n_rois, n_rois, offset=1)

    # conn: (N, R*(R-1)/2)
    conn = conn[:, row, col]
    return conn


def load_schaefer200() -> torch.Tensor:
    parc = np.load(
        Path(__file__).parents[1] / "Schaefer2018_200Parcels_17Networks_order.npy"
    )
    parc = torch.as_tensor(parc)
    return parc
