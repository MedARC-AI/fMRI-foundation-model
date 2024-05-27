import pytest
import torch
from models_vit import vit_small_patch16_fmri
from util.hcp_flat import load_hcp_flat_mask


@pytest.fixture(scope="module")
def random_data() -> torch.Tensor:
    # N, C, T, H, W
    x = torch.randn(2, 1, 16, 144, 320)

    mask = load_hcp_flat_mask()
    x = mask * x
    return x


@pytest.mark.parametrize("global_pool", ["cls", "avg", "spatial"])
def test_vit_small_patch16_fmri(random_data: torch.Tensor, global_pool: str):
    model = vit_small_patch16_fmri(
        t_patch_size=2, cls_embed=True, sep_pos_embed=True, global_pool=global_pool,
    )
    features = model.forward_features(random_data)
    output = model.forward_head(features)
    assert features.shape == (2, 1 + 8 * 149, 384)
    assert output.shape == (2, 400)

    features = model.mask_fill(features[:, 1:, :])
    assert features.shape == (2, 8, 180, 384)
