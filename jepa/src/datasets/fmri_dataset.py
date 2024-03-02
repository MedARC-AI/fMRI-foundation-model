# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import pathlib
import warnings
import webdataset as wds

sys.path.append('/mnt/c/Users/Moham/Desktop/fMRI-foundation-model')

from MAE.utils import *
from datasets import *
import datasets.utils as dutils

from logging import getLogger

import numpy as np
import pandas as pd

import torch

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_fmridataset(
    data_paths,
    batch_size,
    num_samples_per_epoch=1024,
    cache_dir="./cache",
    frames_per_clip=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
):  
    data_paths = data_paths[0]
    data_paths_tars = [data_paths + os.sep + folder for folder in os.listdir(data_paths) if ".tar" in folder]
    num_samples_per_epoch = dutils.count_num_samples(data_paths)

    aug_transform = utils.DataPrepper(
        masking_strategy="conservative",
        patch_depth=8,
        patch_height=8,
        patch_width=8,
        frame_patch_size=1,
        num_timepoints=20
    )

    train_data = wds.WebDataset(data_paths_tars, resampled=False, cache_dir=cache_dir, handler=log_and_continue).select(filter_corrupted_images).rename(key="__key__",
    func="func.png",
    header="header.npy",
    dataset="dataset.txt",
    minmax="minmax.npy",
    meansd="meansd.png").map_dict(func=utils.grayscale_decoder,
    meansd=utils.grayscale_decoder,
    minmax=utils.numpy_decoder).to_tuple(*("func", "minmax", "meansd")).map(aug_transform).with_epoch(num_samples_per_epoch)

    train_dl = wds.WebLoader(
        train_data.batched(batch_size), 
        pin_memory=True,
        shuffle=False,
        batch_size=None,
        num_workers=num_workers, 
        collate_fn=collator,
        persistent_workers=num_workers>0,
    ).with_epoch(num_samples_per_epoch//batch_size).with_length(num_samples_per_epoch)

    return train_data, train_dl, None
    