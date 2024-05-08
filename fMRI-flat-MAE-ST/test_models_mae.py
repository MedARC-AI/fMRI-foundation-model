import pytest
import torch
from models_mae import mae_vit_small_patch16_fmri
from util.hcp_flat import load_hcp_flat_mask


@pytest.fixture(scope="module")
def random_data() -> torch.Tensor:
    # N, C, T, H, W
    x = torch.randn(2, 1, 16, 144, 320)

    mask = load_hcp_flat_mask()
    x = mask * x
    return x


def test_mae_vit_small_patch16_hcpflat(random_data: torch.Tensor):
    model = mae_vit_small_patch16_fmri(
        t_patch_size=2, cls_embed=True, sep_pos_embed=True
    )
    loss, pred, mask = model.forward(random_data)
    print(f"loss: {loss:.3e}")
