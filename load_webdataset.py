import time
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn
import random
from random import randrange
import os
from einops import rearrange
os.environ["WDS_VERBOSE_CACHE"] = "1"
import numpy as np

import webdataset as wds

# tar_urls = "s3://proj-fmri/openneuro-wds/openneuro-0-100-ps13-f8-r1-bspline-shuffled/func-{000000..000577}.tar"
# urls = f"pipe:aws s3 cp {tar_urls} -"
# urls = "file:/scratch/openneuro-0-100-ps13-f8-r1-bspline-shuffled-old/func-{000000..000577}.tar"
urls = "aws s3 cp s3://proj-fmri/openneuro-wds/openneuro-0-100-ps13-f8-r1-bspline-shuffled/func-{000000..000577}.tar -"

def real_sample(src):
    for sample in src:
        for k, v in sample.items():
            if k.endswith(".func.npy"):
                yield {
                    "__key__": sample["__key__"] + '.' + k[:-9],
                    "__url__": sample["__url__"],
                    "func.npy": v,
                }

class Patchify:
    def __init__(self, patch_size, tubelet_size, max_num_patches):
        self.max_num_patches = max_num_patches
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
    
    def __call__(self, sample):
        key, func = sample
        func = torch.from_numpy(func)
        T, X, Y, Z = func.shape
        # print(func.shape)
        rearranged = rearrange(func, '(t t0) (x x0) (y y0) (z z0) -> (t x y z) (t0 x0 y0 z0)', t0=self.tubelet_size, x0=self.patch_size[0], y0=self.patch_size[1], z0=self.patch_size[2])
        L, C = rearranged.shape
        # print(L, C)
        paded = torch.cat([rearranged, torch.zeros((self.max_num_patches - L, C))], dim=0)
        mask = torch.cat([torch.ones((L,)), torch.zeros((self.max_num_patches - L,))], dim=0)
        token_shape = torch.tensor([T // self.tubelet_size, X // self.patch_size[0], Y // self.patch_size[1], Z // self.patch_size[2]])
        return key, paded, mask, token_shape

max_voxels = 8 * 13 * 13 * 13 * 196

def simple_process(sample):
    key = sample[0]
    sample = torch.from_numpy(sample[1]).flatten()
    if sample.shape[0] < max_voxels:
        return key, torch.cat([sample, torch.zeros((max_voxels - sample.shape[0],))])
    else:
        return key, sample

seed = 0
num_worker_batches = 5

process = Patchify((13, 13, 13), 2, 196 * 4)

# dataset = wds.WebDataset(urls, resampled=True).decode("torch").compose(real_sample)\
#     .to_tuple("__key__", "func.npy")\
#     .shuffle(1000, initial=1000, rng=random.Random(seed))\
#     .map(process)\
#     .batched(256, partial=False)\
#     .with_epoch(num_worker_batches)


dataset = wds.WebDataset(urls, resampled=True).decode("torch").compose(real_sample)\
    .to_tuple("__key__", "func.npy")\
    .map(process)\
    .batched(256, partial=False)\
    .with_epoch(num_worker_batches)

# dataset = wds.WebDataset(urls, resampled=True).decode("torch").compose(real_sample)\
#     .to_tuple("__key__", "func.npy")\
#     .map(simple_process)\
#     .batched(256, partial=False)\
#     .with_epoch(num_worker_batches)

# loader = wds.WebLoader(dataset, num_workers=4)

t = time.time()
count = 0
for epoch in range(5):
    for data in dataset:
        print(data[0][0])
        # print(data[1].shape)
        # print(data[2].sum())
        # print(data[2])
        # print(data[3])

        count += 1
        print((time.time() - t) / count)

        # break
