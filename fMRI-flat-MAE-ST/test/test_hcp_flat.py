import pytest
from torch.utils.data import default_collate

from util.hcp_flat import create_hcp_flat


@pytest.mark.parametrize("shuffle", [True, False])
def test_create_hcp_flat(shuffle: bool):
    dataset = create_hcp_flat(shuffle=shuffle)
    dataset = dataset.batched(2, collation_fn=default_collate)
    img, meta = next(iter(dataset))
    print(img.shape, meta)
    assert img.shape == (2, 1, 16, 144, 320)
    assert isinstance(meta, dict)
    assert set(meta.keys()) == {"key", "sub", "mod", "task", "mag", "dir", "start"}
