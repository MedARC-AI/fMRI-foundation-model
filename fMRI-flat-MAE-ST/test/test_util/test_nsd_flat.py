from typing import List, Optional

import pytest
import torch
from torch.utils.data import default_collate

from util.nsd_flat import create_nsd_flat


@pytest.mark.parametrize(
    "sub_list,clip_mode,gsr,shuffle",
    [
        (None, "seq", True, False),
        ([1, 3], "seq", True, False),
        (None, "seq", True, False),
        (None, "event", True, False),
        (None, "seq", False, False),
        (None, "seq", True, True),
    ]
)
def test_create_nsd_flat(
    sub_list: Optional[List[str]], clip_mode: str, gsr: bool, shuffle: bool
):
    dataset = create_nsd_flat(
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
    expected_keys = {"key", "sub", "ses", "run", "start"}
    if clip_mode == "event":
        expected_keys.add("nsd_id")
    assert set(meta.keys()) == expected_keys


@pytest.mark.parametrize("target", ["cluster"])
def test_create_nsd_flat_target(target: str):
    dataset = create_nsd_flat(shuffle=True, clip_mode="event", target=target)
    dataset = dataset.batched(2, collation_fn=default_collate)
    batch = next(iter(dataset))
    img = batch["image"]
    target = batch["target"]
    assert img.shape == (2, 1, 16, 144, 320)
    assert target.shape == (2,)
    assert target.dtype == torch.int64
