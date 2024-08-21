import gzip
import json
import os
import random
from fnmatch import fnmatch
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

HCP_FLAT_ROOT = "https://huggingface.co/datasets/bold-ai/HCP-Flat/resolve/main"
NUM_SHARDS = {"train": 1629, "test": 174}
NSD_NUM_SHARDS = 43
FRAME_SIZE_BYTES = 29859

# Tasks and conditions used in prior works (Zhang, 2021; Rastegarnia, 2023)
INCLUDE_TASKS = {
    "EMOTION", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"
}
INCLUDE_CONDS = {
    "fear",
    "neut",
    "math",
    "story",
    "lf",
    "lh",
    "rf",
    "rh",
    "t",
    "match",
    "relation",
    "mental",
    "rnd",
    "0bk_body",
    "2bk_body",
    "0bk_faces",
    "2bk_faces",
    "0bk_places",
    "2bk_places",
    "0bk_tools",
    "2bk_tools",
}

HCP_TR = {"3T": 0.72, "7T": 1.0}
DEFAULT_DELAY = 4 * 0.72

##### NSD #####

def create_nsd_flat(
    root: Optional[str] = "/weka/proj-medarc/shared/NSD-Flat",
    shards: Optional[Union[int, Iterable[int]]] = 43,
    frames: int = 16,
    shuffle: Optional[bool] = True,
    buffer_size_mb: int = 3840,
) -> wds.WebDataset:
    """
    Create NSD-Flat dataset. Yields samples of (key, images) where key is the webdataset
    sample key and images is shape (C, T, H, W).
    """
    urls = get_nsd_flat_urls(root, shards)

    clipping = nsd_seq_clips(frames)

    if shuffle:
        buffer_size = int(buffer_size_mb * 1024 * 1024 / (frames * FRAME_SIZE_BYTES))
        print(f"Shuffle buffer size: {buffer_size}")
    else:
        buffer_size = 0

    dataset = (
        wds.WebDataset(
            urls,
            resampled=shuffle,
            shardshuffle=1000 if shuffle else False,
            nodesplitter=wds.split_by_node,
            select_files=partial(select_files),
        )
        .decode()
        .map(extract_sample)
        .compose(clipping)
        .shuffle(buffer_size)
        .map_tuple(partial(to_tensor, mask=load_nsd_flat_mask()))
    )
    return dataset

def get_nsd_flat_urls(
    root: Optional[str] = None,
    shards: Optional[Union[int, Iterable[int]]] = None,
):
    if isinstance(shards, int):
        shards = range(shards)
    assert (
        min(shards) >= 0 and max(shards) < NSD_NUM_SHARDS
    ), f"Invalid shards {shards}; expected in [0, {NSD_NUM_SHARDS})"

    urls = [f"{root}/tars/nsd-flat_{shard:06d}.tar" for shard in shards]
    return urls

def load_nsd_flat_mask() -> torch.Tensor:
    mask = np.load(Path(__file__).parents[1] / "nsd-flat_mask.npy")
    mask = torch.as_tensor(mask)
    return mask


# ##### HCP #####

def create_hcp_flat(
    root: Optional[str] = None,
    split: Literal["train", "test"] = "train",
    shards: Optional[Union[int, Iterable[int]]] = None,
    clip_mode: Literal["seq", "event"] = "seq",
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
    root = root or os.environ.get("HCP_FLAT_ROOT") or HCP_FLAT_ROOT
    urls = get_hcp_flat_urls(root, split, shards)

    if shuffle:
        buffer_size = int(buffer_size_mb * 1024 * 1024 / (frames * FRAME_SIZE_BYTES))
        print(f"Shuffle buffer size: {buffer_size}")
    else:
        buffer_size = 0

    if clip_mode == "seq":
        clipping = seq_clips(frames)
    elif clip_mode == "event":
        all_events_path = Path(root) / "all_events.json.gz"
        with gzip.open(all_events_path) as f:
            all_events = json.load(f)
        clipping = event_clips(all_events, frames)

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
            select_files=partial(select_files, task_only=clip_mode=="event"),
        )
        .decode()
        .map(extract_sample)
        .compose(clipping)
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

def event_clips(
    all_events: Dict[str, List[Dict[str, Any]]],
    frames: int = 16,
    delay: float = DEFAULT_DELAY,
):
    def _filter(src: IterableDataset[Tuple[np.ndarray, Dict[str, Any]]]):
        for img, meta in src:
            tr = HCP_TR[meta["mag"]]
            events = all_events[meta["key"]]
            if not events or meta["task"] not in INCLUDE_TASKS:
                continue

            for event in events:
                cond = event["trial_type"]
                if cond not in INCLUDE_CONDS:
                    continue

                onset = event["onset"]
                duration = event["duration"]
                onset_idx = int((onset + delay) / tr)
                # sometimes the end of the trial is cut off
                offset_idx = min(int((onset + delay + duration) / tr), len(img))
                count = (offset_idx - onset_idx) // frames
                for ii in range(count):
                    start = onset_idx + ii * frames
                    clip = img[start : start + frames].copy()
                    meta = {**meta, "start": start, "trial_type": cond}
                    yield clip, meta
    return _filter


def load_hcp_flat_mask() -> torch.Tensor:
    mask = np.load(Path(__file__).parents[1] / "hcp-flat_mask.npy")
    mask = torch.as_tensor(mask)
    return mask


# ALL #

def select_files(fname: str, *, task_only: bool = False):
    suffix = ".".join(fname.split(".")[1:])
    keep = suffix in {"bold.npy", "meta.json", "events.json"}
    if task_only:
        keep = keep and fnmatch(fname, "*mod-tfMRI*mag-3T*")
    return keep


def extract_sample(sample: Dict[str, Any]):
    key = sample["__key__"]
    img = sample["bold.npy"]
    meta = sample["meta.json"]
    meta = {"key": key, **meta}
    
    try:
        events = sample["events.json"]
        return img, meta, events
    except:
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


def hcp_seq_clips(frames: int = 16):
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

def nsd_seq_clips(frames: int = 16):
    def _filter(src: IterableDataset[Tuple[np.ndarray, Dict[str, Any]]]):
        for img, meta, events in src:
            offsets = np.arange(frames)
            for offset in offsets:
                for start in range(offset, len(img) - frames, frames):
                    # copy to avoid a memory leak due to storing the entire underlying array
                    # https://github.com/webdataset/webdataset/issues/354
                    clip = img[start : start + frames].copy()
                    meta = {**meta, "start": start}
                    yield clip, meta, events
    return _filter
