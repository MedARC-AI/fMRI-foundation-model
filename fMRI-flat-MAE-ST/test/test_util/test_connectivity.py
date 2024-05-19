import pytest
import torch

from util.connectivity import Connectome
from util.hcp_flat import load_hcp_flat_mask


@pytest.fixture(scope="module")
def random_data() -> torch.Tensor:
    # N, C, T, H, W
    x = torch.randn(2, 1, 16, 144, 320)

    mask = load_hcp_flat_mask()
    x = mask * x
    return x


def test_connectome(random_data: torch.Tensor):
    model = Connectome()
    x = model.forward_features(random_data)
    assert x.shape == (2, 19900)
    assert x.min() >= -1.0 and x.max() <= 1.0
