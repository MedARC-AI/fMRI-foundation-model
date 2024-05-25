import pytest
from torch.utils.data import default_collate

from util.hcp_flat import create_hcp_flat


@pytest.mark.parametrize(
    "shuffle,clip_mode",
    [(False, "seq"), (True, "seq"), (True, "event")],
)
def test_create_hcp_flat(shuffle: bool, clip_mode: str):
    dataset = create_hcp_flat(shuffle=shuffle, clip_mode=clip_mode)
    dataset = dataset.batched(2, collation_fn=default_collate)
    img, meta = next(iter(dataset))
    print(img.shape, meta)
    assert img.shape == (2, 1, 16, 144, 320)
    assert isinstance(meta, dict)
    expected_keys = {"key", "sub", "mod", "task", "mag", "dir", "start"}
    if clip_mode == "event":
        expected_keys.add("trial_type")
    assert set(meta.keys()) == expected_keys
