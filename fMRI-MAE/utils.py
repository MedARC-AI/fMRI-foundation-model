from io import BytesIO
import os
import random
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from skimage import filters
from torchvision import transforms
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
import re
import torch.nn.functional as F
import torch.nn as nn

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

def crop_or_pad(tensor, new_shape):
    # Ensure the tensor has at least three dimensions
    if tensor.dim() < 3:
        raise ValueError("Tensor must have at least 3 dimensions")

    # Current dimensions of the last three axes
    current_shape = tensor.shape[-3:]

    # Compute padding and cropping needed for each dimension
    padding_crop = [(ns - cs) for ns, cs in zip(new_shape, current_shape)]
    if sum(padding_crop)==0:
        return tensor

    # Apply cropping if necessary
    if any(pc < 0 for pc in padding_crop):
        crop_slices = [slice(-pc//2, ns-pc//2) if pc < 0 else slice(None) for pc, ns in zip(padding_crop, new_shape)]
        tensor = tensor[..., crop_slices[0], crop_slices[1], crop_slices[2]]

    # Calculate padding to apply after cropping if necessary
    pad_values = [(max(0, pc)//2, max(0, pc) - max(0, pc)//2) for pc in padding_crop]

    # Apply padding
    tensor = F.pad(tensor, pad_values[2] + pad_values[1] + pad_values[0])

    return tensor


class DataPrepper:
    def __init__(
        self,
        num_frames=4,
        masking_strategy="MNI",
        patch_depth=8,
        patch_height=8,
        patch_width=8,
        frame_patch_size=1,
        image_size=[88, 104, 72]
    ):
        self.num_frames = num_frames
        self.masking_strategy = masking_strategy
        self.patch_depth = 8
        self.patch_height = 8
        self.patch_width = 8
        self.frame_patch_size = 1
        self.image_size=image_size

    def __call__(self, func):
        start_timepoint = np.random.choice(np.arange(func.shape[1] - self.num_frames))
        timepoints = np.arange(start_timepoint, start_timepoint + self.num_frames)

        func = func[:,timepoints]

        # crop image_size acc to config
        func = crop_or_pad(func, self.image_size)

        if self.masking_strategy=="MNI" or self.masking_strategy=="None":
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

def patchwise_cosine_similarity(latents1,latents2=None):
    if latents2 is None:
        latents_norm = latents1/latents1.norm(dim=-1, keepdim=True)
        cos_sim = torch.bmm(latents_norm, latents_norm.permute(0,2,1))
    else:
        latents_norm1 = latents1/latents1.norm(dim=-1, keepdim=True)
        latents_norm2 = latents2/latents2.norm(dim=-1, keepdim=True)
        cos_sim = latents_norm1 @ latents_norm2.T
    return cos_sim
    
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

def view_brain(data,cut_coords=None):
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
    return plotting.view_img(new_nii, bg_img=None, cut_coords=cut_coords, vmax=1, cmap=plt.cm.gray, threshold=None)

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


class VICRegHandler(nn.Module):
    def __init__(self, in_dim, num_layers=3, act=nn.GELU, h=1024, out_dim=4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            act(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            act(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x):
        return self.projector(x)

    @staticmethod
    def filter_global_to_local(l, enc_mask, dec_mask):
        '''Get the subset of global tokens that correspond to encoder mask only'''
        comb_mask = enc_mask | dec_mask
        comb_indices = torch.where(comb_mask)[0]
        enc_indices = torch.where(enc_mask)[0]
        # enc_set = set(enc_indices.cpu().tolist()) 
        
        # new_mask = torch.zeros_like(comb_indices, dtype=bool)
        # for i, idx in enumerate(comb_indices):
        #     if idx in enc_set:
        #         new_mask[i] = True
        
        new_mask = torch.isin(comb_indices, enc_indices)    
        return l[:, new_mask]

    @staticmethod
    def vicreg_loss(l1, l2, gamma=1.0, lamda=25, mu=25, nu=1, eps=1e-4):
        # rand_indices = torch.randperm(l1.shape[0])[:int(0.25*l1.shape[0])]
        # l1 = l1[rand_indices]
        # l2 = l2[rand_indices]

        std_l1 = torch.sqrt(l1.flatten(1).var(dim=0)+eps)  # nxd
        std_l2 = torch.sqrt(l2.flatten(1).var(dim=0)+eps)  # nxd
        var_loss = F.relu(gamma - std_l1).mean() + F.relu(gamma - std_l2).mean()
        del std_l1, std_l2

        sim_loss = F.mse_loss(l1, l2)

        l1 = l1 - l1.mean(0, keepdim=True)  # b,n,d
        l2 = l2 - l2.mean(0, keepdim=True)
        # always keep cls and pick a random set of tokens
        rand_indices = torch.cat([torch.tensor([0]), 1+torch.randperm(l1.shape[1]-1)])[:int(0.2*l1.shape[1])]
        # off_diag = ~torch.eye(l1.shape[2], dtype=bool)[None].to(l1.device).expand(l1.shape[1], -1, -1)
        
        l1_sub = l1[:, rand_indices]
        del l1
        cov_l1 = torch.bmm(l1_sub.permute(1,2,0), l1_sub.permute(1,0,2))/(l1_sub.shape[0]-1)  # 0.1*n,d,d
        # cov_loss = (cov_l1[off_diag]**2).sum()/(l1_sub.shape[1]*l1_sub.shape[2])  # too much mem needed
        cov_loss = ((cov_l1**2).sum() - (torch.diagonal(cov_l1, dim1=1,dim2=2)**2).sum())/(l1_sub.shape[1]*l1_sub.shape[2])
        del cov_l1, l1_sub

        l2_sub = l2[:, rand_indices]
        del l2
        cov_l2 = torch.bmm(l2_sub.permute(1,2,0), l2_sub.permute(1,0,2))/(l2_sub.shape[0]-1)
        cov_loss = cov_loss + ((cov_l2**2).sum() - (torch.diagonal(cov_l2, dim1=1,dim2=2)**2).sum())/(l2_sub.shape[1]*l2_sub.shape[2])  # div by nxd
        del cov_l2, l2_sub

        vic_loss = lamda * sim_loss + mu * var_loss + nu * cov_loss

        return vic_loss


class SimCLRHandler(nn.Module):
    def __init__(self, in_dim, num_layers=2, act=nn.GELU, out_dim=1024):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            act(),
            nn.Linear(in_dim, max(in_dim,out_dim)),
        )

    def forward(self, x):
        return self.projector(x)

    @staticmethod
    def simclr_loss(lats, temp=0.006):
        logits = (nn.functional.normalize(lats.flatten(1),dim=-1) @
                    nn.functional.normalize(lats.flatten(1),dim=-1).T) / temp

        labels = torch.diag_embed(
            torch.ones(logits.shape[0] // 2), offset=logits.shape[0] // 2
        ) + torch.diag_embed(torch.ones(logits.shape[0] // 2), offset=-logits.shape[0] // 2)
        labels = labels.to(lats.device)
        
        mask = torch.ones_like(logits).bool()
        torch.diagonal(mask).fill_(False)
        
        labels = labels[mask].reshape(logits.shape[0], logits.shape[0]-1)
        logits = logits[mask].reshape(*labels.shape)

        contr_loss = -(logits.log_softmax(-1) * labels).sum(-1).mean()

        return contr_loss
