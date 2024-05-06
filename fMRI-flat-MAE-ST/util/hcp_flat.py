from pathlib import Path

import numpy as np
import torch


def load_hcp_flat_mask() -> torch.Tensor:
    mask = np.load(Path(__file__).parents[1] / "hcp-flat_mask.npy")
    mask = torch.as_tensor(mask)
    return mask
