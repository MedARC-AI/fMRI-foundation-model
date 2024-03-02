import os
import tarfile

import torch
import numpy as np
from skimage import filters
from einops import rearrange
from PIL import Image
from io import BytesIO

def count_num_samples(root, is_tar=True):
    unique_samples = set()  # To hold unique base names of samples
    count = 0
    
    for dirpath, dirnames, filenames in os.walk(root):
        if is_tar:
            for filename in filenames:
                if filename.endswith('.tar'):
                    try:
                        with tarfile.open(os.path.join(dirpath, filename), 'r') as tar:
                            # Extract base names for each file in the tar.
                            tar_samples = {name.split('.')[0] for name in tar.getnames()}
                            count += len(tar_samples)
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
        else:
            for filename in filenames:
                # Split filename by "." and take the first part as the base name
                base_name = filename.split('.')[0]
                unique_samples.add(base_name)
    
    if not is_tar:
        count = len(unique_samples)
    
    return count


def grayscale_decoder(image_data):
    return np.array(Image.open(BytesIO(image_data))).astype(np.float32) / 65535

def numpy_decoder(npy_data):
    return np.load(BytesIO(npy_data))

def get_brain_pos_patches(
    brain_segmentation,
    patch_depth=8,
    patch_height=8,
    patch_width=8,
    frame_patch_size=1,
    masking_strategy="conservative",
):
    reshaped_mask = reshape_to_original(brain_segmentation)
    frames, _, _, depth = reshaped_mask.shape
    if masking_strategy == "conservative":
        # plt.imshow(reshaped_mask.sum(axis=(0, -1))) # [64, 64]
        reshaped_mask = reshaped_mask.sum(axis=(0, -1), keepdim=True).repeat(
            frames, 1, 1, depth
        )  # [4, 64, 64, 48]

    patched_mask = rearrange(
        reshaped_mask,
        "(f pf) (d pd) (h ph) (w pw) -> f d h w (pd ph pw pf)",
        pd=patch_depth,
        ph=patch_height,
        pw=patch_width,
        pf=frame_patch_size,
    )
    return (patched_mask.sum(-1) > 0).int().flatten()

def threshold_based_masking(org_images):
    org_images[org_images == 1] = 0  # ignore the padding
    thresholds = filters.threshold_multiotsu(org_images.numpy(), classes=3)
    brain_segmentation = org_images > thresholds.min()
    return brain_segmentation

def reshape_to_2d(tensor):
    if tensor.ndim == 5:
        tensor = tensor[0]
    assert tensor.ndim == 4
    return rearrange(tensor, "b h w c -> (b h) (c w)")

def reshape_to_original(tensor_2d, h=64, w=64, c=48):
    # print(tensor_2d.shape) # torch.Size([1, 256, 3072])
    return rearrange(tensor_2d, "(tr h) (c w) -> tr h w c", h=h, w=w, c=c)

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    print(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def filter_corrupted_images(sample):
    """If all the required files are not present don't use them."""
    correct_data = ("func.png" in sample and "dataset.txt" in sample and "header.npy" in sample and "meansd.png" in sample and "minmax.npy" in sample)
    return correct_data

class DataPrepper:
    def __init__(
        self,
        masking_strategy="conservative",
        patch_depth=8,
        patch_height=8,
        patch_width=8,
        frame_patch_size=1,
        num_timepoints=4,
    ):
        self.masking_strategy = masking_strategy
        self.patch_depth = patch_depth
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.frame_patch_size = frame_patch_size
        self.num_timepoints = num_timepoints

    def __call__(self, sample):
        func, minmax, meansd = sample
        min_, max_, min_meansd, max_meansd = minmax
        reshaped_func = reshape_to_original(func)

        if len(reshaped_func) == 4:
            timepoints = np.arange(4)
        else:
            start_timepoint = np.random.choice(np.arange(len(reshaped_func) - self.num_timepoints))
            timepoints = np.arange(start_timepoint, start_timepoint + self.num_timepoints)

        func = torch.Tensor(reshaped_func[timepoints])
        meansd = torch.Tensor(reshape_to_original(meansd))

        # Keep track of the empty patches
        mean, sd = meansd
        org_images = reshape_to_2d(func * mean + sd)
        brain_segmentation = threshold_based_masking(org_images)
        pos_patches = get_brain_pos_patches(
            brain_segmentation,
            patch_depth=self.patch_depth,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            frame_patch_size=self.frame_patch_size,
            masking_strategy=self.masking_strategy,
        )
        func = func.permute(0, -1, 1, 2).contiguous()
        meansd = meansd.permute(0, -1, 1, 2).contiguous()
        # return func, meansd, pos_patches
        return func