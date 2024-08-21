from io import BytesIO
import os
import random
import numpy as np
import torch
from einops import rearrange
from nilearn import plotting
from PIL import Image
from skimage import filters
from torchvision import transforms
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
import re

def my_split_by_node(urls): return urls

def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")

def my_split_by_node(urls): return urls

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def grayscale_decoder(image_data):
    return np.array(Image.open(BytesIO(image_data))).astype(np.float32) / 65535


def numpy_decoder(npy_data):
    return np.load(BytesIO(npy_data))


def reshape_to_2d(tensor):
    if tensor.ndim == 5:
        tensor = tensor[0]
    assert tensor.ndim == 4
    return rearrange(tensor, "b h w c -> (b h) (c w)")


def reshape_to_original(tensor_2d, h=64, w=64, c=48):
    # print(tensor_2d.shape) # torch.Size([1, 256, 3072])
    return rearrange(tensor_2d, "(tr h) (c w) -> tr h w c", h=h, w=w, c=c)


def plot_numpy_nii(image):
    while image.ndim > 3:
        image = image[0]
    nii = nib.Nifti1Image(image.astype(np.float32), np.eye(4))  # noqa
    plotting.plot_epi(nii, cmap="gray")


def threshold_based_masking(org_images):
    thresholds = filters.threshold_multiotsu(org_images.numpy(), classes=3)
    brain_segmentation = org_images > thresholds.min()
    return brain_segmentation


def get_brain_pos_patches(
    func,
    patch_depth=8,
    patch_height=8,
    patch_width=8,
    frame_patch_size=1,
    masking_strategy="conservative",
):
    _, _, depth = func.shape
    if masking_strategy == "conservative":
        func = func.sum(axis=(-1), keepdim=True).repeat(1, 1, depth)
    else:
        raise Exception("Not implemented other masking strategies than conservative.")

    return func


class DataPrepper:
    def __init__(
        self,
        num_frames=4,
        masking_strategy="MNI",
        patch_depth=8,
        patch_height=8,
        patch_width=8,
        frame_patch_size=1,
    ):
        self.num_frames = num_frames
        self.masking_strategy = masking_strategy
        self.patch_depth = 8
        self.patch_height = 8
        self.patch_width = 8
        self.frame_patch_size = 1

    def __call__(self, func):
        start_timepoint = np.random.choice(np.arange(func.shape[1] - self.num_frames))
        timepoints = np.arange(start_timepoint, start_timepoint + self.num_frames)

        func = func[:,timepoints]

        if self.masking_strategy=="MNI":
            return func, None
        
        brain_segmentation = threshold_based_masking(func.mean(1))
        pos_patches = None
        for brain in brain_segmentation:
            output = get_brain_pos_patches(
                brain,
                patch_depth=self.patch_depth,
                patch_height=self.patch_height,
                patch_width=self.patch_width,
                frame_patch_size=self.frame_patch_size,
                masking_strategy=self.masking_strategy,
            )
            if pos_patches is None:
                pos_patches = output[None]
            else:
                pos_patches = torch.vstack((pos_patches, output[None]))
        return func, pos_patches


def plot_slices(unpatches):
    if unpatches.ndim == 5:
        unpatches = unpatches[0]
    return transforms.ToPILImage()(reshape_to_2d(unpatches))


def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')
        

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("param counts:\n{:,} total\n{:,} trainable".format(total, trainable))
    return trainable


def contrastive_loss(
    cls_token1: torch.Tensor, cls_token2: torch.Tensor, temperature: torch.Tensor
):
    feat1 = cls_token1 / cls_token1.norm(dim=1, keepdim=True)
    feat2 = cls_token2 / cls_token2.norm(dim=1, keepdim=True)

    cosine_sim = feat1 @ feat2.T
    logit_scale = temperature.exp()  # log scale, learned during training
    feat1 = cosine_sim * logit_scale
    feat2 = feat1.T

    labels = torch.arange(feat1.shape[0]).to(feat1.device)
    loss = (
        torch.nn.functional.cross_entropy(feat1, labels)
        + torch.nn.functional.cross_entropy(feat2, labels)
    ) / 2
    return loss

### MindEye functions ###

def soft_clip_loss(preds, targs, temp=0.006):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def mixco(voxels, beta=0.15, s_thresh=0.5, perm=None, betas=None, select=None):
    if perm is None:
        perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    if betas is None:
        betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    if select is None:
        select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss
    
def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def batchwise_cosine_similarity(Z,B):
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def prenormed_batchwise_cosine_similarity(Z,B):
    return (Z @ B.T).T

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def get_masking_ratio(current_epoch, total_epochs, start_masking_ratio, end_masking_ratio):
    """Returns the masking ratio for the current epochs. Linearly increase the masking ratio over the span of the training"""
    return start_masking_ratio + (end_masking_ratio-start_masking_ratio) * ((current_epoch+1)/total_epochs)

def view_brain(data):
    if torch.is_tensor(data):
        data = data.numpy()
    if data.ndim==5:
        new_nii = nib.Nifti1Image((data[0,0].astype(np.float32)-.5)*2, np.eye(4))
    elif data.ndim==4:
        new_nii = nib.Nifti1Image((data[0].astype(np.float32)-.5)*2, np.eye(4))
    elif data.ndim==3:
        new_nii = nib.Nifti1Image((data.astype(np.float32)-.5)*2, np.eye(4))
    else:
        raise Exception("Check dimensionality of your brain data")
    return plotting.view_img(new_nii, bg_img=None, vmax=1, cmap=plt.cm.gray, threshold=None)

def get_first_tar(train_urls):
    if isinstance(train_urls, list):
        # If train_urls is a list, get the first element
        url = train_urls[0]
    else:
        # If train_urls is a string, treat it as the only element
        url = train_urls

    # Extract the first tar file using regular expression
    match = re.search(r'\{(\d+)\.\.', url)
    if match:
        first_tar = match.group(1)
        return f"/scratch/fmri_foundation_datasets/NSD_MNI_wds/{first_tar}.tar"
    else:
        return None