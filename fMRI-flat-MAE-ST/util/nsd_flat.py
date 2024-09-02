import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

from util.misc import zscore

NSD_FLAT_ROOT = "https://huggingface.co/datasets/bold-ai/NSD-Flat/resolve/main"
NUM_SHARDS = 300
MASK_SIZE = 30191

NSD_TR = 1.0
DEFAULT_DELAY_TRS = 0  # TODO: default delay?


def create_nsd_flat(
    root: Optional[str] = None,
    shards: Optional[Union[int, Iterable[int]]] = None,
    sub_list: Optional[List[int]] = None,
    clip_mode: Literal["seq", "event"] = "seq",
    target: Optional[Literal["cluster"]] = None,
    frames: int = 16,
    stride: Optional[int] = None,
    gsr: bool = True,
    shuffle: bool = False,
    buffer_size: int = 1024,
) -> wds.WebDataset:
    """
    Create NSD-Flat dataset. Yields dict samples with keys "image", "meta", and
    optionally "target". The images have shape (C, T, H, W).

    References:
        https://github.com/webdataset/webdataset/issues/250#issuecomment-1454094496
        https://github.com/tmbdev-archive/webdataset-imagenet-2/blob/main/imagenet.py
        https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/readers/reader_wds.py
    """
    assert (
        target is None or clip_mode == "event"
    ), "event clipping required for targets"

    root = root or os.environ.get("NSD_FLAT_ROOT") or NSD_FLAT_ROOT
    urls = get_nsd_flat_urls(root, shards)

    if shuffle:
        # Nb, after undoing gsr the data are float32 rather than uint8, to avoid more
        # precision loss
        dtype_size_bytes = 32 if not gsr else 8
        buffer_size_bytes = buffer_size * frames * MASK_SIZE * dtype_size_bytes
        print(f"Shuffle buffer size (MB): {buffer_size_bytes / 1024 / 1024:.0f}")

    if clip_mode == "seq":
        clipping = seq_clips(frames, stride=stride, is_training=shuffle)
    elif clip_mode == "event":
        clipping = event_clips(frames)

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
            select_files=select_files(sub_list=sub_list),
        )
        .decode()
        .map(extract_sample)
    )

    if not gsr:
        dataset = dataset.map(ungsr)

    dataset = dataset.compose(clipping)

    # add an integer "target" key to the sample
    if target is not None:
        assert target == "cluster", "only cluster targets implemented"
        class_map_path = Path(root) / "nsd_coco_73k_semantic_cluster_ids.npy"
        class_map = np.load(class_map_path)
        dataset = dataset.compose(with_targets(class_map))

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    # late conversion to tensor, after buffering to save memory
    dataset = dataset.map_dict(
        image=partial(to_tensor, mask=load_nsd_flat_mask(root))
    )
    return dataset


def get_nsd_flat_urls(
    root: Optional[str] = None,
    shards: Optional[Union[int, Iterable[int]]] = None,
):
    root = root or os.environ.get("NSD_FLAT_ROOT") or NSD_FLAT_ROOT

    shards = shards or NUM_SHARDS
    if isinstance(shards, int):
        shards = range(shards)
    assert (
        min(shards) >= 0 and max(shards) < NUM_SHARDS
    ), f"Invalid shards {shards}; expected in [0, {NUM_SHARDS})"

    urls = [f"{root}/tars/nsd-flat_{shard:06d}.tar" for shard in shards]
    return urls


def select_files(
    sub_list: Optional[List[int]] = None,
    exclude_exts: Optional[Tuple[str, ...]] = None,
):
    if sub_list is not None:
        assert all(
            isinstance(sub, int) and 1 <= sub <= 8 for sub in sub_list
        ), "invalid sub_list, expected a list of int in [1, 8]"

        sub_list = set(sub_list)

    if exclude_exts is not None:
        exclude_exts = set(exclude_exts)

    def _filter(fname: str):
        key, ext = fname.split(".", maxsplit=1)

        if exclude_exts and ext in exclude_exts:
            return False

        ents = dict(kv.split("-") for kv in key.split("_"))

        if sub_list and int(ents["sub"]) not in sub_list:
            return False

        return True

    return _filter


def extract_sample(sample: Dict[str, Any]):
    key = sample["__key__"]
    bold = sample["bold.npy"]
    meta = sample["meta.json"]
    meta = {"key": key, **meta}
    events = sample["events.json"]
    misc = sample.get("misc.npz")
    return {"bold": bold, "meta": meta, "events": events, "misc": misc}


def ungsr(sample: Dict[str, Any]):
    bold = sample["bold"]
    misc = sample["misc"]

    mean = misc["mean"]
    std = misc["std"]
    offset = misc["offset"]
    global_signal = misc["global_signal"]
    beta = misc["beta"]

    # uint8 to float32 with normal range
    bold = bold.astype("float32") / 255.0
    bold = (bold - 0.5) / 0.2

    # recover timeseries
    bold = std * bold + mean
    bold = bold + global_signal[:, None] * beta + offset

    # re-zscore
    bold, _, _ = zscore(bold)
    return {**sample, "bold": bold}


def to_tensor(img: np.ndarray, mask: torch.Tensor):
    img = torch.from_numpy(img)
    if img.dtype == torch.uint8:
        img = img / 255.0
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


def seq_clips(frames: int = 16, stride: Optional[int] = None, is_training: bool = True):
    stride = stride or frames

    def _filter(src: IterableDataset[Dict[str, Any]]):
        for sample in src:
            bold = sample["bold"]
            meta = sample["meta"]

            first_idx = random.randint(0, frames) if is_training else 0
            count = len(bold) // frames
            for ii in range(count):
                start = first_idx + ii * frames
                stop = start + frames
                if stop > len(bold):
                    break

                # copy to avoid a memory leak due to storing the entire underlying array
                # https://github.com/webdataset/webdataset/issues/354
                clip = bold[start:stop].copy()
                meta = {**meta, "start": start}
                yield {"image": clip, "meta": meta}
    return _filter


def event_clips(
    frames: int = 16,
    delay_trs: int = DEFAULT_DELAY_TRS
):
    def _filter(src: IterableDataset[Dict[str, Any]]):
        for sample in src:
            bold = sample["bold"]
            meta = sample["meta"]
            events = sample["events"]
            for event in events:
                nsd_id = event["nsd_id"]
                onset_idx = event["index"]

                start = onset_idx + delay_trs
                stop = start + frames
                if stop <= len(bold):
                    clip = bold[start:stop].copy()
                    meta = {**meta, "start": start, "nsd_id": nsd_id}
                    yield {"image": clip, "meta": meta}
    return _filter


def with_targets(class_id_map: np.ndarray):
    def _filter(src: IterableDataset[Dict[str, Any]]):
        for sample in src:
            nsd_id = sample["meta"]["nsd_id"]
            target = class_id_map[nsd_id]
            yield {**sample, "target": target}
    return _filter


def load_nsd_flat_mask(root: Path) -> torch.Tensor:
    mask = np.load(Path(root) / "nsd-flat_mask.npy")
    mask = torch.as_tensor(mask)
    return mask
