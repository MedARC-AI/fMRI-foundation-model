import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator

import torchio as tio
import datasets
import torch

from einops import rearrange

os.environ["WDS_VERBOSE_CACHE"] = "1"
import webdataset as wds

class Patchify:
    def __init__(self, patch_size, tubelet_size, max_num_patches):
        self.max_num_patches = max_num_patches
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
    
    def __call__(self, func):
        func = torch.from_numpy(func)
        T, X, Y, Z = func.shape
        rearranged = rearrange(func, '(t t0) (x x0) (y y0) (z z0) -> (t x y z) (t0 x0 y0 z0)', t0=self.tubelet_size, x0=self.patch_size[0], y0=self.patch_size[1], z0=self.patch_size[2])
        L, C = rearranged.shape
        paded = torch.cat([rearranged, torch.zeros((self.max_num_patches - L, C))], dim=0)
        mask = torch.cat([torch.ones((L,)), torch.zeros((self.max_num_patches - L,))], dim=0)
        token_shape = torch.tensor([T // self.tubelet_size, X // self.patch_size[0], Y // self.patch_size[1], Z // self.patch_size[2]])
        return paded, mask, token_shape

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        # TODO: add augmentation for fMRI
        self.transform = transforms.Compose([
            tio.CropOrPad((65, 78, 65)),
            # tio.RandomFlip(axes=('LR',)),
            # tio.RandomAffine(scales=(0.9, 1.1), degrees=10, isotropic=False, default_pad_value='otsu'),
            # tio.RandomAffine(scales=(0.9, 1.1)),
            # tio.RandomNoise(std=(0, 0.1)),
            # tio.RandomBlur(std=(0, 0.1)),
            # tio.RandomBiasField(coefficients=(0, 0.1)),
            ])
        self.patchify = Patchify(args.patch_size, args.tubelet_size, args.max_num_patches)
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio, args.max_num_patches
            )

    def __call__(self, sample):
        key, func = sample
        process_data = self.transform(func)
        paded, mask, token_shape = self.patchify(process_data)
        # print(process_data.shape)
        return paded, token_shape, mask, self.masked_position_generator(token_shape)

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

# for now, we need this because there are dots in __key__ when creating the dataset
# don't need this after we fix the dataset creation
# def real_sample(src):
#     for sample in src:
#         for k, v in sample.items():
#             if k.endswith(".func.npy"):
#                 yield {
#                     "__key__": sample["__key__"] + '.' + k[:-9],
#                     "__url__": sample["__url__"],
#                     "func.npy": v,
#                 }

def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    tar_urls = args.data_urls
    if args.data_location == 'file':
        urls = f"file:{tar_urls}"
    elif args.data_location == 's3':
        urls = f"pipe:aws s3 cp {tar_urls} -"
    dataset = wds.WebDataset(urls, resampled=args.data_resample).decode("torch")\
        .to_tuple("__key__", "func.npy")\
        .shuffle(args.data_buffer_size, initial=args.data_buffer_size, rng=random.Random(args.data_seed))\
        .map(transform)\
        .batched(args.batch_size, partial=False)\
        .with_epoch(args.data_batch_per_epoch)
    print("Data Aug = %s" % str(transform))
    return dataset

# old version with huggingface datasets
# def build_pretraining_dataset(args):
#     transform = DataAugmentationForVideoMAE(args)
#     dataset = datasets.load_dataset('fMRI-openneuro', 'test1', 
#                                     num_datasets=args.data_ndatasets, 
#                                     num_frames=args.num_frames,
#                                     sampling_rate=args.sampling_rate,
#                                     streaming=True)
#     dataset = dataset.map(transform)
#     print("Data Aug = %s" % str(transform))
#     return dataset['train']
