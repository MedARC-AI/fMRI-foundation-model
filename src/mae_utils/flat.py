import gzip
import json
import os
import sys
import random
from fnmatch import fnmatch
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils import zscore

shared1000 = np.where(np.load("/weka/proj-medarc/shared/mindeyev2_dataset/shared1000.npy"))[0]

HCP_FLAT_ROOT = "https://huggingface.co/datasets/bold-ai/HCP-Flat/resolve/main"
HCP_NUM_SHARDS = 1803
NSD_NUM_SHARDS = 300
FRAME_SIZE_BYTES = 29859
HCP_MASK_SIZE = 29859

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
DEFAULT_DELAY_SECS = 4 * 0.72

##### NSD #####

def create_nsd_flat(
    root: Optional[str] = "/weka/proj-medarc/shared/NSD-Flat",
    shards: Optional[Union[int, Iterable[int]]] = 300,
    frames: int = 16,
    shuffle: Optional[bool] = True,
    buffer_size_mb: int = 3840,
    gsr: Optional[bool] = True,
    sub: Optional[str] = None,
    ses: Optional[str] = None,
    num_sessions: Optional[int] = 40,
    run: Optional[str] = None,
    mindeye_only: Optional[bool] = False,
    only_shared1000: Optional[bool] = False,
    mindeye_TR_delay: int = 3,
) -> wds.WebDataset:
    """
    Create NSD-Flat dataset. Yields samples of (key, images) where key is the webdataset
    sample key and images is shape (C, T, H, W).
    """
    urls = get_nsd_flat_urls(root, shards)
    # 300 tar paths aka shards for NSD
    # each tar has .npy files like sub-08_ses-24_run-14.bold.npy already shuffled subj & sess

    clipping = seq_clips(frames, mindeye_only=mindeye_only, 
                         only_shared1000=only_shared1000, mindeye_TR_delay=mindeye_TR_delay,
                         mask=load_nsd_flat_mask(), gsr=gsr)
    
    if shuffle:
        buffer_size = int(buffer_size_mb * 1024 * 1024 / (frames * FRAME_SIZE_BYTES))
        print(f"Shuffle buffer size: {buffer_size}")
    else:
        buffer_size = 0

    dataset = (
        wds.WebDataset(
            urls,
            resampled=True,
            shardshuffle=300 if shuffle else False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            select_files=partial(select_files, sub=sub, ses=ses, run=run, num_sessions=num_sessions),
        )
        .shuffle(buffer_size) # batch takes samples randomly across all the shards
        .decode()
        .map(partial(extract_sample,gsr=gsr))
        .compose(clipping)
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

def load_nsd_flat_mask(folder="/weka/proj-medarc/shared/NSD-Flat/") -> torch.Tensor:
    mask = np.load(os.path.join(folder, "nsd-flat_mask.npy"))
    mask = torch.as_tensor(mask)
    return mask

def load_nsd_flat_mask_visual(folder="/weka/proj-medarc/shared/NSD-Flat/") -> torch.Tensor:
    mask = np.load(os.path.join(folder, "nsd-flat_mask_visual.npy"))
    mask = torch.as_tensor(mask)
    return mask


##### HCP #####

def create_hcp_flat(
    root: Optional[str] = None,
    shards: Optional[Union[int, Iterable[int]]] = None,
    clip_mode: Literal["seq", "event"] = "seq",
    frames: int = 16,
    shuffle: Optional[bool] = None,
    buffer_size_mb: int = 3840,
    gsr: Optional[bool] = True,
    sub_min: Optional[int] = 0,
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
    urls = get_hcp_flat_urls(root, shards)

    if shuffle:
        buffer_size = int(buffer_size_mb * 1024 * 1024 / (frames * FRAME_SIZE_BYTES))
        print(f"Shuffle buffer size: {buffer_size}")
    else:
        buffer_size = 0

    if clip_mode == "seq":
        clipping = seq_clips(frames, mask=load_hcp_flat_mask(), gsr=gsr)
    elif clip_mode == "event":
        err # need to debug this
        # all_events_path = "/weka/proj-medarc/shared/HCP-Flat/all_events.json.gz"
        # with gzip.open(all_events_path) as f:
        #     all_events = json.load(f)
        # clipping = hcp_event_clips(all_events, frames)

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
            resampled=True,
            shardshuffle=1000 if shuffle else False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            select_files=partial(select_files_hcp, task_only=clip_mode=="event", sub_min=sub_min),
        )
        .decode()
        .map(partial(extract_sample,gsr=gsr))
        .compose(clipping)
        .shuffle(buffer_size)
    )
    return dataset


def get_hcp_flat_urls(
    root: Optional[str] = None,
    shards: Optional[Union[int, Iterable[int]]] = None,
):
    root = root or os.environ.get("HCP_FLAT_ROOT") or HCP_FLAT_ROOT

    shards = shards or HCP_NUM_SHARDS
    if isinstance(shards, int):
        shards = range(shards)
    assert (
        min(shards) >= 0 and max(shards) < HCP_NUM_SHARDS
    ), f"Invalid shards {shards}; expected in [0, {HCP_NUM_SHARDS})"

    urls = [f"{root}/tars/hcp-flat_{shard:06d}.tar" for shard in shards]
    return urls

def load_hcp_flat_mask(folder="/weka/proj-medarc/shared/HCP-Flat/") -> torch.Tensor:
    mask = np.load(os.path.join(folder, "hcp-flat_mask.npy"))
    mask = torch.as_tensor(mask)
    return mask

def hcp_event_clips(
    all_events: Dict[str, List[Dict[str, Any]]],
    frames: int = 16,
    delay: float = DEFAULT_DELAY_SECS,
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

def select_files_hcp(fname: str, *, 
                 task_only: bool = False,
                 sub_min: int = 0):
    # Define the file suffixes to keep
    suffix = ".".join(fname.split(".")[1:])
    keep = suffix in {"bold.npy", "meta.json", "events.json", "misc.npz"}

    # Additional filtering based on task_only
    if task_only:
        keep = keep and fnmatch(fname, "*mod-tfMRI*mag-3T*")

    if sub_min > 0:
        match = re.search(r"sub-(\d+)", fname)
        subject_id = int(match.group(1))
        keep = keep and (sub_min <= subject_id)

    return keep

# ALL #
import re
def select_files(fname: str, *, sub=None, ses=None, run=None, num_sessions=40):
    # Define the file suffixes to keep
    suffix = ".".join(fname.split(".")[1:])
    keep = suffix in {"bold.npy", "meta.json", "events.json", "misc.npz"}

    if ses is not None:
        keep = keep and (ses in fname)

    if num_sessions < 10:
        session_pattern = rf"ses-0[1-{num_sessions}]"
    elif num_sessions < 20:
        session_pattern = rf"ses-0[1-9]|ses-1[0-{num_sessions - 10}]"
    elif num_sessions < 30:
        session_pattern = rf"ses-0[1-9]|ses-1[0-9]|ses-2[0-{num_sessions - 20}]"
    else:
        session_pattern = rf"ses-0[1-9]|ses-1[0-9]|ses-2[0-9]|ses-3[0-{num_sessions - 30}]"

    if num_sessions < 40:
        keep = keep and re.search(session_pattern, fname) is not None

    # If `run` is specified, handle special cases for "task_only" or other `run` patterns
    if run is not None:
        if run == "task_only":
            # Exclude run-14 (resting-state); allow only specific task runs
            match = re.search(r"run-(0[1-9]|1[0-3])", fname)
            keep = keep and match is not None
        else:
            keep = keep and (run in fname)

    if sub is not None:
        if isinstance(sub, list):
            # If `sub` is a list of numbers denoting subject IDs
            sub_pattern = rf"sub-({'|'.join(f'{s:02d}' for s in sub)})"
            keep = keep and re.search(sub_pattern, fname) is not None
        else:
            # if sub is simply something like 'sub-01'
            keep = keep and (sub in fname)

    return keep


def extract_sample(sample: Dict[str, Any], gsr=False):
    key = sample["__key__"]
    img = sample["bold.npy"]
    meta = sample["meta.json"]
    meta = {"key": key, **meta}
    events = sample["events.json"]
    misc = sample["misc.npz"]
    
    if not gsr:
        mean = misc["mean"]
        std = misc["std"]
        beta = misc["beta"]
        global_signal = misc["global_signal"]
        offset = misc["offset"]
        
        img = img / 255.0
        img = (img - 0.5) / 0.2

        img = mean + std * img
        img = img + global_signal[:, None] * beta + offset

        session_mean = img.mean(axis=0)
        session_std = img.std(axis=0)

        img = (img - session_mean[None]) / session_std[None]

    return img, meta, events, misc


def to_tensor(img, mask, gsr=False):
    img = torch.from_numpy(img)
    if gsr:
        img = (img - 0.5) / 0.2
    if mask is not None:
        img = unmask(img, mask)
    img = img.unsqueeze(0).float()  # (C, T, H, W)
    return img

def unmask(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    unmasked = torch.zeros(
        (img.shape[0], *mask.shape), dtype=img.dtype, device=img.device
    )
    unmasked[:, mask] = img
    return unmasked


def batch_unmask(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # img: shape [B, C, D] -> [32, 16, 30191]
    # mask: shape [M, N] -> [144, 320]

    B, C, D = img.shape  # Batch size, channels, last dimension size
    M, N = mask.shape  # Mask dimensions

    # Ensure the mask is flattened to apply along the last dimension (D) of img
    flat_mask = mask.view(-1)  # shape: [M * N]
    num_unmasked_elements = flat_mask.sum()  # The number of true elements in the mask

    # Initialize an empty tensor for the unmasked output
    unmasked = torch.zeros((B, C, M * N), dtype=img.dtype, device=img.device)

    # Use broadcasting and advanced indexing to unmask
    idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # Indices where mask is True
    unmasked[:, :, idx] = img[:, :, :num_unmasked_elements]

    # Reshape the unmasked tensor to the original shape
    unmasked = unmasked.view(B, C, M, N)

    return unmasked

def seq_clips(frames: int = 16, mindeye_only=False, mindeye_TR_delay=3, only_shared1000=False, mask=None, gsr=False):
    def _filter(src: IterableDataset[Tuple[np.ndarray, Dict[str, Any]]]):
        offset = np.random.choice(np.arange(frames-1))
        for ii, (img, meta, events, meanstd) in enumerate(src):
            if mindeye_only:
                group = [(s['index'], s['nsd_id']) for s in events]
                mindeye_info = np.array(group)
                if len(mindeye_info)==0:
                    continue
                image_onsets, image_nsd_id = mindeye_info[:,0], mindeye_info[:,1]
                for istart, start in enumerate(image_onsets + mindeye_TR_delay):
                    nsd_id = image_nsd_id[istart].item() - 1 # because nsd_id is 1-indexed
                    if only_shared1000 is None:
                        if not (nsd_id in shared1000):
                            clip = img[start : start + frames].copy()
                            clip = to_tensor(clip, mask=mask, gsr=gsr)
                            yield clip, f"{int(meta['ses']):02}{int(meta['run']):02}{start:03}", nsd_id, meta['sub'], meanstd['mean'], meanstd['std']
                    elif only_shared1000:
                        if nsd_id in shared1000:
                            clip = img[start : start + frames].copy()
                            clip = to_tensor(clip, mask=mask, gsr=gsr)
                            yield clip, meta, nsd_id, meta['sub'], meanstd['mean'], meanstd['std']
                    else:
                        # if not (nsd_id in shared1000):
                        clip = img[start : start + frames].copy()
                        clip = to_tensor(clip, mask=mask, gsr=gsr)
                        yield clip, f"{int(meta['ses']):02}{int(meta['run']):02}{start:03}", nsd_id, meta['sub'], meanstd['mean'], meanstd['std']
            else:
                # you dont want len(starts)>batch_size or else all samples come from exact same npy file!
                # but if len(starts)=1 then dataloading is slower and samples never come from same npy
                starts = np.random.choice(np.arange(offset, len(img) - frames, frames), size=8, replace=False)
                for start in starts:
                    # copy to avoid a memory leak due to storing the entire underlying array
                    # https://github.com/webdataset/webdataset/issues/354
                    clip = img[start : start + frames].copy()
                    meta = {**meta, "start": start}
                    clip = to_tensor(clip, mask=mask, gsr=gsr)
                    yield clip, meta, meanstd['mean'], meanstd['std'] 
    return _filter