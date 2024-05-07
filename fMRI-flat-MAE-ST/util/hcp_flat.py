import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

HCP_FLAT_ROOT = "https://huggingface.co/datasets/bold-ai/HCP-Flat/resolve/main"
NUM_SHARDS = {"train": 611, "test": 66}
NUM_SAMPLES = {"train": 100000, "test": 5000}


def create_hcp_flat_loader(
    root: Optional[str] = None,
    training: bool = True,
    shards: Optional[Union[int, Iterable[int]]] = None,
    samples: Optional[int] = None,
    frames: int = 16,
    batch_size: int = 64,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
):
    """
    Create HCP-Flat data loader. Yields batches of (keys, images) where keys is a list
    of sample keys and images is a batch of clips, shape (N, C, T, H, W).

    References:
        https://github.com/webdataset/webdataset/issues/250#issuecomment-1454094496
        https://github.com/tmbdev-archive/webdataset-imagenet-2/blob/main/imagenet.py
        https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/readers/reader_wds.py
    """
    root = root or os.environ.get("HCP_FLAT_ROOT") or HCP_FLAT_ROOT
    split = "train" if training else "test"

    shards = shards or NUM_SHARDS[split]
    if isinstance(shards, int):
        shards = range(shards)
    assert (
        min(shards) >= 0 and max(shards) < NUM_SHARDS[split]
    ), f"Invalid shards {shards}; expected in [0, {NUM_SHARDS[split]})"

    samples = samples or NUM_SAMPLES[split]
    _, world_size, _, _ = wds.utils.pytorch_worker_info()
    batches = samples // (world_size * batch_size)
    assert batches > 1, f"Too few samples: {samples} for world and batch size"

    urls = [f"{root}/{split}/hcp-flat_{split}_{shard:06d}.tar" for shard in shards]

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

    # Note that we are splitting the long timeseries into clips after shuffling, which
    # means clips from the same series will appear consecutively in the batch(es).
    # I think this is not too bad. It is basically equivalent to training on longer
    # sequences. Clipping before shuffling might be preferred, but it results in a bad
    # system memory leak (https://github.com/webdataset/webdataset/issues/354).
    dataset = (
        wds.WebDataset(
            urls,
            resampled=training,
            shardshuffle=1000 if training else False,
            nodesplitter=wds.split_by_node,
            select_files=select_extensions(("bold.npy",)),
            cache_dir=cache_dir,
        )
        .shuffle(1000 if training else 0)
        .decode()
        .map(partial(extract_images, mask=load_hcp_flat_mask()))
        .compose(partial(to_clips, frames=frames))
        .batched(batch_size, partial=False)
    )

    loader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=num_workers
    )
    loader = loader.with_epoch(batches)
    return loader


def select_extensions(extensions: Tuple[str, ...]):
    extensions_set = set(extensions)
    def select_files(fname: str):
        suffix = ".".join(fname.split(".")[1:])
        return suffix in extensions_set
    return select_files


def extract_images(sample: Dict[str, Any], mask: torch.Tensor):
    key = sample["__key__"]
    images = sample["bold.npy"]
    images = torch.from_numpy(images) / 255.0
    images = (images - 0.5) / 0.2
    images = unmask(images, mask)
    # (C, T, H, W,)
    images = images.unsqueeze(0)
    return key, images


def unmask(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    images_unmasked = torch.zeros(
        (images.shape[0], *mask.shape), dtype=images.dtype, device=images.device
    )
    images_unmasked[:, mask] = images
    return images_unmasked


def to_clips(frames: int = 16):
    def _filter(src: IterableDataset[Tuple[str, torch.Tensor]]):
        for key, images in src:
            offset = random.randint(0, frames)
            for start in range(offset, images.shape[1] - frames, frames):
                yield key, images[:, start : start + frames]
    return _filter


def load_hcp_flat_mask() -> torch.Tensor:
    mask = np.load(Path(__file__).parents[1] / "hcp-flat_mask.npy")
    mask = torch.as_tensor(mask)
    return mask
