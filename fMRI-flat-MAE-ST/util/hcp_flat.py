import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

HCP_FLAT_ROOT = "https://huggingface.co/datasets/bold-ai/HCP-Flat/resolve/main"
NUM_SHARDS = {"train": 1629, "test": 174}
FRAME_SIZE_BYTES = 29859


def create_hcp_flat(
    root: Optional[str] = None,
    split: Literal["train", "test"] = "train",
    shards: Optional[Union[int, Iterable[int]]] = None,
    frames: int = 16,
    shuffle: Optional[bool] = None,
    buffer_size_mb: int = 3840,
) -> wds.WebDataset:
    """
    Create HCP-Flat dataset. Yields samples of (key, images) where key is the webdataset
    sample key and images is shape (C, T, H, W).

    References:
        https://github.com/webdataset/webdataset/issues/250#issuecomment-1454094496
        https://github.com/tmbdev-archive/webdataset-imagenet-2/blob/main/imagenet.py
        https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/readers/reader_wds.py
    """
    urls = get_hcp_flat_urls(root, split, shards)

    if shuffle:
        buffer_size = int(buffer_size_mb * 1024 * 1024 / (frames * FRAME_SIZE_BYTES))
        print(f"Shuffle buffer size: {buffer_size}")
    else:
        buffer_size = 0

    # In training, we resample shards with replacement independently in every worker and
    # yield batches up to the target number of samples. In test, we iterate over the
    # shards in order, with workers getting interleaving shards, and yield batches up to
    # the target samples. In a distributed setting with variable size shards, setting a
    # fixed number of samples is the easiest way to get balanced batches per worker. In
    # training we will still see all data eventually. But in test, it means we cut off
    # some data.

    # Note that in training this does not do deterministic shuffling, which we would
    # need for exact reproducibility. They get determistic shuffling in timm, but it's
    # more complicated.

    # Nb, in initial pretraining runs we shuffled before generating clips, which
    # resulted in less random batches. Tbd whether this makes a difference.
    dataset = (
        wds.WebDataset(
            urls,
            resampled=shuffle,
            shardshuffle=1000 if shuffle else False,
            nodesplitter=wds.split_by_node,
            select_files=select_extensions(("bold.npy", "meta.json")),
        )
        .decode()
        .map(extract_sample)
        .compose(to_clips(frames))
        .shuffle(buffer_size)
        .map_tuple(partial(to_tensor, mask=load_hcp_flat_mask()))
    )
    return dataset


def get_hcp_flat_urls(
    root: Optional[str] = None,
    split: Literal["train", "test"] = "train",
    shards: Optional[Union[int, Iterable[int]]] = None,
):
    root = root or os.environ.get("HCP_FLAT_ROOT") or HCP_FLAT_ROOT

    shards = shards or NUM_SHARDS[split]
    if isinstance(shards, int):
        shards = range(shards)
    assert (
        min(shards) >= 0 and max(shards) < NUM_SHARDS[split]
    ), f"Invalid shards {shards}; expected in [0, {NUM_SHARDS[split]})"

    urls = [f"{root}/{split}/hcp-flat_{split}_{shard:06d}.tar" for shard in shards]
    return urls


def select_extensions(extensions: Tuple[str, ...]):
    extensions_set = set(extensions)
    def select_files(fname: str):
        suffix = ".".join(fname.split(".")[1:])
        return suffix in extensions_set
    return select_files


def extract_sample(sample: Dict[str, Any]):
    key = sample["__key__"]
    img = sample["bold.npy"]
    meta = sample["meta.json"]
    meta = {"key": key, **meta}
    return img, meta


def to_tensor(img: np.ndarray, mask: torch.Tensor):
    img = torch.from_numpy(img) / 255.0
    img = (img - 0.5) / 0.2
    img = unmask(img, mask)
    img = img.unsqueeze(0)  # (C, T, H, W)
    return img


def unmask(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    unmasked = torch.zeros(
        (img.shape[0], *mask.shape), dtype=img.dtype, device=img.device
    )
    unmasked[:, mask] = img
    return unmasked


def to_clips(frames: int = 16):
    def _filter(src: IterableDataset[Tuple[np.ndarray, Dict[str, Any]]]):
        for img, meta in src:
            offset = random.randint(0, frames)
            for start in range(offset, len(img) - frames, frames):
                # copy to avoid a memory leak due to storing the entire underlying array
                # https://github.com/webdataset/webdataset/issues/354
                clip = img[start : start + frames].copy()
                meta = {**meta, "start": start}
                yield clip, meta
    return _filter


def load_hcp_flat_mask() -> torch.Tensor:
    mask = np.load(Path(__file__).parents[1] / "hcp-flat_mask.npy")
    mask = torch.as_tensor(mask)
    return mask
