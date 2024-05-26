import pytest
import torch
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


@pytest.mark.parametrize("target", ["task", "trial_type"])
def test_create_hcp_flat_target(target: str):
    dataset = create_hcp_flat(shuffle=True, clip_mode="event", target=target)
    dataset = dataset.batched(2, collation_fn=default_collate)
    img, target = next(iter(dataset))
    assert img.shape == (2, 1, 16, 144, 320)
    assert target.shape == (2,)
    assert target.dtype == torch.int64
