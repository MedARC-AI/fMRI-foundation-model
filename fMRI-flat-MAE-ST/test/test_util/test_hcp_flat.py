from typing import Optional

import pytest
import torch
from torch.utils.data import default_collate

from util.hcp_flat import create_hcp_flat


@pytest.mark.parametrize(
    "sub_list,clip_mode,gsr,shuffle",
    [
        ("train", "seq", True, False),
        ("test", "seq", True, False),
        (None, "seq", True, False),
        (None, "event", True, False),
        (None, "seq", False, False),
        (None, "seq", True, True),
    ]
)
def test_create_hcp_flat(
    sub_list: Optional[str], clip_mode: str, gsr: bool, shuffle: bool
):
    dataset = create_hcp_flat(
        sub_list=sub_list,
        clip_mode=clip_mode,
        gsr=gsr,
        shuffle=shuffle,
    )
    dataset = dataset.batched(2, collation_fn=default_collate)
    batch = next(iter(dataset))
    img = batch["image"]
    meta = batch["meta"]
    print(img.shape, meta)
    assert img.shape == (2, 1, 16, 144, 320)
    assert isinstance(meta, dict)
    expected_keys = {"key", "sub", "mod", "task", "mag", "dir", "start"}
    if clip_mode == "event":
        expected_keys.add("trial_type")
    assert set(meta.keys()) == expected_keys


@pytest.mark.parametrize("target", ["task", "trial_type"])
def test_create_hcp_flat_target(target: str):
    dataset = create_hcp_flat(shards=1, clip_mode="event", target=target)
    dataset = dataset.batched(2, collation_fn=default_collate)
    batch = next(iter(dataset))
    img = batch["image"]
    target = batch["target"]
    assert img.shape == (2, 1, 16, 144, 320)
    assert target.shape == (2,)
    assert target.dtype == torch.int64
