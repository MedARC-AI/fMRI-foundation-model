#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages and setup gpu configuration.
# This code block shouldnt need to be adjusted!
import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import copy
import math
from einops import rearrange
from einops.layers.torch import Rearrange
import time
import random
import h5py
import webdataset as wds
import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import utils
from models import get_vit
import nibabel as nib
from nilearn import plotting

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

### Multi-GPU config ###
device_count = torch.cuda.device_count()
print(f"Number of available CUDA devices: {device_count}")

local_rank = os.getenv('LOCAL_RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print(f"LOCAL RANK={local_rank}")

num_devices = os.getenv('NUM_GPUS')
if num_devices is None: 
    num_devices = 1
else:
    num_devices = int(num_devices)
print(f"NUM GPUS={num_devices}")
distributed = True if num_devices>1 else False
if distributed: assert device_count==num_devices

node = os.getenv('SLURM_NODEID')
if node is None:
    node = 0
else:
    node = int(node)
print(f"NODE={node}")

global_rank = os.getenv('RANK')
if global_rank is None:
    global_rank = 0
else:
    global_rank = int(global_rank)
print(f"GLOBAL RANK={global_rank}")

world_size = os.getenv('WORLD_SIZE')
if world_size is None: 
    world_size = 1
else:
    world_size = int(world_size)
print(f"WORLD_SIZE={world_size}")

if utils.is_interactive():
    # Following allows you to change functions in models.py or utils.py and 
    # have this notebook automatically update with your revisions
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Load parameters from yaml config
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

# create global variables from the config
for attribute_name in config.keys():
    globals()[attribute_name] = config[f'{attribute_name}']

data_type = torch.float16 # change depending on your mixed_precision
batch_size = global_batch_size // num_devices

# FSDP Setup
if distributed:
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy
    import functools
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    print("starting init_process_group...")
    dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
    print(f"setting device to cuda:{local_rank}")
    try:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda',local_rank)
        print(f"\nSuccessfully set cuda:{local_rank} | global_rank{global_rank} | node{node}")
    except Exception as error:        
        print(f"\nFAILED TO SET DEVICE cuda:{local_rank} | global_rank{global_rank} | node{node}")
        print("An exception occurred:", error)
        
else:
    device = torch.device('cuda')

print("PID of this process =",os.getpid())
print("device =", device, "distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)


# # Configuration

# In[2]:


print(config)

# seed all random functions
utils.seed_everything(seed)

outdir = os.path.abspath(f'../ckpts/{model_name}')
os.makedirs(outdir,exist_ok=True)
print("outdir", outdir)

if use_contrastive_loss:
    global_batch_size = global_batch_size // 2 # contrastive loss doubles the batch size with the same samples and different masks
print("global_batch_size", global_batch_size)

use_cls_token = True if use_contrastive_loss else use_cls_token
print("use_cls_token", use_cls_token)

num_patches = int(
    (img_size[0] / patch_size)
    * (img_size[1] / patch_size)
    * (img_size[2] / patch_size)
    * num_frames
)
num_patches_per_timepoint = num_patches // num_frames
num_encoder_patches = int(num_patches_per_timepoint * (1 - tube_start_masking_ratio) * num_frames)
num_decoder_patches = int(num_patches_per_timepoint * (1 - decoder_mask_ratio) * num_frames)
print("num_patches", num_patches)
print("num_encoder_patches", num_encoder_patches)
print("num_decoder_patches", num_decoder_patches)


# # Prep models

# In[3]:


# Initialize list to keep track of session_ids to use with nn.Embedding
session_ids = []
def get_or_append_session_id(new_ids,device=device):
    indices = []
    for id in new_ids:
        if id not in session_ids:
            session_ids.append(id)
        indices.append(session_ids.index(id))
    return torch.tensor(indices).to(device)
# print(get_or_append_session_id(["0001", "33", "42"]))
# print(get_or_append_session_id(["2", "33"]))


# In[4]:


vit_size = {
    "encoder": encoder_model,
    "decoder": decoder_model
}
    
model = get_vit(
    size=vit_size,
    image_size=img_size,  # depth, height, width
    image_patch_size=(patch_size,patch_size,patch_size),  # depth, height, width patch size
    frames=num_frames,
    frame_patch_size=frame_patch_size,
    channels=1,
    use_rope_emb=use_rope_emb,
    use_cls_token=use_cls_token,
)
utils.count_params(model)

# function to select random num_frames from sample and obtain brain-positive patches
aug_transform = utils.DataPrepper(
    num_frames=num_frames,
    masking_strategy=masking_strategy,
    patch_depth=patch_size,
    patch_height=patch_size,
    patch_width=patch_size,
    frame_patch_size=frame_patch_size,
)

# test that the model works without error
if not distributed:
    model = model.to(device)
    encoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
    encoder_mask[:num_encoder_patches] = True
    decoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
    decoder_mask[-num_decoder_patches:] = True
    with torch.no_grad():
        print("\nencoder")
        encoder_out = model(
                    torch.randn(6, 1, num_frames, img_size[0], img_size[1], img_size[2]).to(device),
                    encoder_mask=encoder_mask,
                    verbose=True)
        print("\ndecoder")
        decoder_out = model(
                    encoder_out, 
                    encoder_mask=encoder_mask, 
                    decoder_mask=decoder_mask, 
                    verbose=True)
        if use_cls_token:
            enc_cls_token = encoder_out[:, :1, :]
            encoder_patches = encoder_out[:, 1:, :]
            dec_cls_token = decoder_out[:, :1, :]
            decoder_patches = decoder_out[:, 1:, :]
            print("\nenc_cls_token", enc_cls_token.shape)
            print("encoder_patches", encoder_patches.shape)
            print("dec_cls_token", dec_cls_token.shape)
            print("decoder_patches", decoder_patches.shape)


# ## Create dataset and data loaders

# In[5]:


# from dataloader import create_dataset, create_loader
# print(train_urls)

# train_dp = create_dataset(train_urls, 
#                           is_s3=is_s3, 
#                           sample_shuffle=100, shard_shuffle=100)
# train_dl = create_loader(train_dp, batch_size=batch_size, num_workers=num_workers)


# In[3]:


from litdata import StreamingDataset
from torch.utils.data import DataLoader

# Remote path where full dataset is stored
input_dir = "s3://proj-fmri/fmri_foundation_datasets/NSD_MNI_litdata"
# "/weka/proj-fmri/paulscotti/fMRI-foundation-model/dataset_creation/wds_creation/nsd_litdata"
#'s3://my-bucket/my_optimized_dataset'

# Create the Streaming Dataset
dataset = StreamingDataset(input_dir, shuffle=True)

# Access any elements of the dataset
sample = dataset[50]
func = sample['func']
print(func.shape, func.dtype)

# Create PyTorch DataLoader and iterate over it to train your AI models.
train_dl = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)


# ### Check data loaders work

# In[ ]:


if not distributed:
    num_it = 2
    print(f"Yielding {num_it} batches")
    
    for i, batch in enumerate(train_dl):
        print("iter",i)
        input_func = batch['func']
        if i >= (num_it-1):
            break
    
    print("Done!")
    print("input_func", input_func.shape)


# # Playing with the data, visualization of patching + masking

# In[10]:


# func, brain_pos_pats = aug_transform(input_func)
# print(func.shape)
# utils.view_brain(func.clamp(0,1))


# In[20]:


if masking_strategy=="MNI":
    MNI_brain = nib.load("/weka/proj-fmri/paulscotti/old_fMRI-foundation-model/dataset_creation/afni_conversion/tpl-MNI152NLin2009cAsym_res-02_T1w_brain.nii.gz").get_fdata()
    brain_pos_voxels = MNI_brain[6:94,8:112,10:82]
    brain_pos_pats = model.patchify(torch.Tensor(brain_pos_voxels)[None,None,None])
    brain_pos_pats_vit = rearrange(brain_pos_pats, "b ... d -> b (...) d").mean(-1)[0]


# In[21]:


if utils.is_interactive():
    # extract func volumes and their reference mean and standard deviation volumes
    if masking_strategy=="MNI":
        func, _ = aug_transform(input_func)
    else:
        func, brain_pos_voxels = aug_transform(input_func)
        brain_pos_pats = model.patchify(torch.Tensor(brain_pos_voxels)[None,None,None])
        brain_pos_pats_vit = rearrange(brain_pos_pats, "b ... d -> b (...) d").mean(-1)[0]
    func = func.unsqueeze(1)  # add empty first dimension to serve as 1d channel dimension

    # patchify func samples
    print("func", func.shape)
    patches = model.patchify(func)
    print("patches", patches.shape)
    patches_vit = rearrange(patches, "b ... d -> b (...) d")
    print("patches_vit", patches_vit.shape)
    print("num patches in one timepoint", patches_vit.shape[1] // num_frames)

    # start by masking everything (aka include nothing)
    tube_mask = torch.zeros(num_patches // num_frames).to(torch.bool)
    # approximate brain positive patches for the whole batch
    batch_positive_approx = (brain_pos_pats_vit > 0)
    mask_idx_candidates = torch.where(batch_positive_approx)[0]
    mask_idx_candidates = mask_idx_candidates[torch.randperm(len(mask_idx_candidates))]
    print("Percentage of brain positive patches", len(mask_idx_candidates) / len(batch_positive_approx))
    tube_idx = mask_idx_candidates[: int(num_patches / num_frames * (1 - tube_start_masking_ratio))]
    print("num tube patches =", len(tube_idx))
    tube_mask[tube_idx] = True  # Trues mean to include the patch, False means to remove the patch
    tube_mask = tube_mask.tile(num_frames)  # repeat masking for the other timepoints
    print("tube mask percent", tube_mask.sum().item() / len(tube_mask))

    # create decoder mask similar to tube mask, but ensure no overlap
    decoder_mask = torch.zeros(num_patches // num_frames).to(torch.bool)  # start by masking everything (aka include nothing)
    remaining_mask_idx = mask_idx_candidates[int(num_patches / num_frames * (1 - tube_start_masking_ratio)) :]  # brain positive tokens not selected for the encoder tokens
    decoder_mask_idx = remaining_mask_idx[:int(num_patches / num_frames * (1 - decoder_mask_ratio))]
    print("num decoder patches =", len(decoder_mask_idx))
    decoder_mask[decoder_mask_idx] = True
    decoder_mask = decoder_mask.tile(num_frames)  # repeat masking for the other timepoints
    print("decoder_mask percent", decoder_mask.sum().item() / len(decoder_mask))

    # apply masks to patches_vit
    tube_patches_vit = copy.deepcopy(patches_vit.detach())
    decoder_patches_vit = copy.deepcopy(patches_vit.detach())
    # tube_patches_vit[:, tube_mask] = 1
    # decoder_patches_vit[:, decoder_mask] = 1
    tube_patches_vit[:, ~tube_mask] = 0
    decoder_patches_vit[:, ~decoder_mask] = 0

    # undo patchification so we can visualize
    tube_unpatches = rearrange(
        tube_patches_vit,
        "b (f d h w) c -> b f d h w c",
        d=img_size[0]//patch_size,
        h=img_size[1]//patch_size,
        w=img_size[2]//patch_size,
    )
    decoder_unpatches = rearrange(
        decoder_patches_vit,
        "b (f d h w) c -> b f d h w c",
        d=img_size[0]//patch_size,
        h=img_size[1]//patch_size,
        w=img_size[2]//patch_size,
    )
    print("tube_unpatches", tube_unpatches.shape)
    print("decoder_unpatches", decoder_unpatches.shape)
    
    encoder_func = rearrange(
        tube_unpatches,
        "b f d h w (pd ph pw pf c) -> b c (f pf) (d pd) (h ph) (w pw)",
        b=len(func),
        f=num_frames,
        d=img_size[0] // patch_size,
        h=img_size[1] // patch_size,
        w=img_size[2] // patch_size,
        pd=patch_size,
        ph=patch_size,
        pw=patch_size,
        pf=frame_patch_size,
    )
    decoder_func = rearrange(
        decoder_unpatches,
        "b f d h w (pd ph pw pf c) -> b c (f pf) (d pd) (h ph) (w pw)",
        b=len(func),
        f=num_frames,
        d=img_size[0] // patch_size,
        h=img_size[1] // patch_size,
        w=img_size[2] // patch_size,
        pd=patch_size,
        ph=patch_size,
        pw=patch_size,
        pf=frame_patch_size,
    )
    print("encoder_func", encoder_func.shape)
    print("decoder_func", decoder_func.shape)
    
    brain_pos_vit = copy.deepcopy(patches_vit.detach())
    brain_pos_vit[:,batch_positive_approx.repeat(num_frames)] = 1
    brain_pos_vit[:,~batch_positive_approx.repeat(num_frames)] = 0
    brain_pos_unpatches = rearrange(
        brain_pos_vit,
        "b (f d h w) c -> b f d h w c",
        d=img_size[0]//patch_size,
        h=img_size[1]//patch_size,
        w=img_size[2]//patch_size,
    )
    brain_pos_func = rearrange(
        brain_pos_unpatches,
        "b f d h w (pd ph pw pf c) -> b c (f pf) (d pd) (h ph) (w pw)",
        b=len(func),
        f=num_frames,
        d=img_size[0] // patch_size,
        h=img_size[1] // patch_size,
        w=img_size[2] // patch_size,
        pd=patch_size,
        ph=patch_size,
        pw=patch_size,
        pf=frame_patch_size,
    )

    # Visualize
    idx = 0
    print("original func")
    display(transforms.ToPILImage()(utils.reshape_to_2d(func[idx].clamp(0,1))))
    
    print("\nbrain-positive patches")
    display(transforms.ToPILImage()(utils.reshape_to_2d(brain_pos_func[idx].clamp(0,1))))

    print("\nencoder func")
    display(transforms.ToPILImage()(utils.reshape_to_2d(encoder_func[idx].clamp(0,1))))

    print("\ndecoder func")
    display(transforms.ToPILImage()(utils.reshape_to_2d(decoder_func[idx].clamp(0,1))))


# # FSDP / optimizer / saving functions

# In[22]:


if distributed:    
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=500
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        auto_wrap_policy=my_auto_wrap_policy,
        use_orig_params=False,
        cpu_offload=None, #CPUOffload(offload_params=True)
        sync_module_states=True,
        limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
        device_id=device,
    )
    print(f"\nSuccessfully loaded FSDP model to device on global_rank {global_rank}\n")


# In[23]:


no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
opt_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)
num_iterations_per_epoch = num_samples_per_epoch // global_batch_size

total_steps = num_epochs * num_iterations_per_epoch * num_devices
print("total_steps", total_steps)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=total_steps,
)

print("\nDone with model preparations!")
num_params = utils.count_params(model)


# In[24]:


default_ckpt_path = outdir+f'/last.pth'

def save_ckpt(model,tag="last"):
    if distributed: dist.barrier()
    model_states = model.state_dict()
    if global_rank == 0:
        ckpt_path = outdir+f'/{tag}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_states,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }, ckpt_path)
        print(f"\n---saved {ckpt_path}!---\n")

def resume_ckpt(model, optimizer, device, ckpt_path=default_ckpt_path):
    if global_rank == 0:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    if distributed: dist.barrier()
    torch.cuda.empty_cache()
    return model, optimizer, lr_scheduler, epoch


# # Start wandb (if enabled)

# In[25]:


if utils.is_interactive():
    wandb_log = False
if global_rank==0 and wandb_log: # only use main process for wandb logging
    import wandb
    wandb_project = 'found'
    print(f"wandb {wandb_project} run {model_name}")
    # need to configure wandb beforehand in terminal with "wandb init"!
    wandb_config = {
      "model_name": model_name,
      "global_batch_size": global_batch_size,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "num_samples_per_epoch": num_samples_per_epoch,
      "num_iterations_per_epoch": num_iterations_per_epoch,
      "encoder_model": encoder_model,
      "decoder_model": decoder_model,
      "tube_start_masking_ratio": tube_start_masking_ratio,
      "tube_end_masking_ratio": tube_end_masking_ratio,
      "decoder_mask_ratio": decoder_mask_ratio,
      "num_frames": num_frames,
      "patch_size": patch_size,
      "frame_patch_size": frame_patch_size,
      "use_contrastive_loss": use_contrastive_loss,
      "use_cls_token": use_cls_token,
      "constrastive_loss_weight": constrastive_loss_weight,
      "num_params": num_params,
      "max_lr": max_lr,
      "ckpt_interval": ckpt_interval,
      "ckpt_saving": ckpt_saving,
      "seed": seed,
      "distributed": distributed,
      "num_devices": num_devices,
      "world_size": world_size,
      "train_urls": train_urls,
    }
    print("wandb_config:\n",wandb_config)
    print("wandb_id:",model_name)
    wandb.init(
        id=model_name,
        project=wandb_project,
        name=model_name,
        config=wandb_config,
        resume="allow",
    )
else:
    wandb_log = False


# # Start training

# In[26]:


epoch = 0
lrs, recon_losses, contrastive_losses, means_losses = [], [], [], []


# In[ ]:


if resume_from_ckpt is True:
    if os.path.exists(default_ckpt_path):
        print(f"Resuming from {default_ckpt_path}...")
        model, optimizer, lr_scheduler, epoch = resume_ckpt()


# In[27]:


if distributed: dist.barrier()
mse = nn.MSELoss()
if use_contrastive_loss:
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # learned logit scale
model.train()
progress_bar = tqdm(range(epoch, num_epochs), disable=global_rank!=0, desc="Overall")
for epoch in progress_bar:
    # get the masking ratio for the current epoch
    tube_mask_ratio = utils.get_masking_ratio(
        current_epoch=epoch, 
        total_epochs=num_epochs, 
        start_masking_ratio=tube_start_masking_ratio, 
        end_masking_ratio=tube_end_masking_ratio
    )
    with torch.cuda.amp.autocast(dtype=data_type):
        for train_i, batch in enumerate(train_dl):
            optimizer.zero_grad()

            input_func = batch['func']
            if train_i==0 and epoch==0:
                print("min", input_func.min())
                print("max", input_func.max())
            input_func = input_func.clamp(0,1)

            # input_session_ids = list(map(lambda x: x[0] + "_" + x[1] + "_" + x[2], zip(batch['dataset_id.txt'], batch['subject_id.txt'], batch['session_id.txt'])))
            # func_ids = get_or_append_session_id(input_session_ids)

            if masking_strategy=="MNI":
                func, _ = aug_transform(input_func)
            else:
                func, brain_pos_voxels = aug_transform(input_func)
                brain_pos_pats = model.patchify(torch.Tensor(brain_pos_voxels)[None,None,None])
                brain_pos_pats_vit = rearrange(brain_pos_pats, "b ... d -> b (...) d").mean(-1)[0]
                
            if use_contrastive_loss:  # create positive pairs by duplicating the batch
                func = torch.cat([func, func], dim=0)
                meansd = torch.cat([meansd, meansd], dim=0)
                brain_pos_pats = torch.cat([brain_pos_pats, brain_pos_pats], dim=0)

            func = func.unsqueeze(1)

            # create tube mask (i.e., a mask that is the same for all frames/timepoints)
            tube_mask = torch.zeros(num_patches // num_frames).to(torch.bool)
            batch_positive_approx = (brain_pos_pats_vit > 0)
            mask_idx_candidates = torch.where(batch_positive_approx)[0]
            mask_idx_candidates = mask_idx_candidates[torch.randperm(len(mask_idx_candidates))]
            tube_idx = mask_idx_candidates[:int(num_patches / num_frames * (1 - tube_mask_ratio))]
            tube_mask[tube_idx] = True
            tube_mask = tube_mask.tile(num_frames)

            # create decoder mask
            decoder_mask = torch.zeros(num_patches // num_frames).to(torch.bool)
            remaining_mask_idx = mask_idx_candidates[int(num_patches / num_frames * (1 - tube_mask_ratio)):]
            decoder_mask_idx = remaining_mask_idx[:int(num_patches / num_frames * (1 - decoder_mask_ratio))]
            decoder_mask[decoder_mask_idx] = True
            decoder_mask = decoder_mask.tile(num_frames)

            # encode the tube patches
            encoder_out = model(func, encoder_mask=tube_mask, device=device) #id=func_ids, 
            if use_cls_token:
                enc_cls_token = encoder_out[:,:1,:]

            # decode both the encoder_out patches and masked decoder patches
            decoder_out = model(encoder_out, encoder_mask=tube_mask, decoder_mask=decoder_mask, device=device)
            # # subset only the reconstructed decoder patches
            output = decoder_out[:, -decoder_mask.sum():]

            # compare to ground truth and calculate loss
            target_patches = model.patchify(func)
            target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")
            target = target_patches_vit[:, decoder_mask].to(device)

            # add ground truth means to decoder outputs
            means_patches = target_patches.mean(1).unsqueeze(1)
            means_patches_vit = rearrange(means_patches, "b ... d -> b (...) d")
            means = means_patches_vit.tile(num_frames,1)[:, decoder_mask].to(device)

            loss = mse(output, target)

            # contrastive loss
            if use_contrastive_loss:
                n_b = len(func) // 2
                cls_token1 = enc_cls_token[:n_b, 0, :]  # first half of batch, cls_token shape B, 1, d_model
                cls_token2 = enc_cls_token[n_b:, 0, :]
                contrastive_loss = utils.contrastive_loss(cls_token1, cls_token2, temperature=logit_scale)
                loss += constrastive_loss_weight * contrastive_loss
                contrastive_losses.append(contrastive_loss.item())

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            recon_losses.append(loss.item())
            means_losses.append(mse(means, target).item())
            lrs.append(optimizer.param_groups[0]["lr"])

            if train_i >= (num_iterations_per_epoch-1):
                print("train_i", train_i, "local_rank", local_rank, "global_rank", global_rank)
                break

        logs = {
            "train/loss": np.mean(recon_losses[-(train_i + 1) :]),
            "train/means_loss": np.mean(means_losses[-(train_i + 1) :]),
            "train/num_steps": len(recon_losses),
            "lr": np.mean(lrs[-(train_i + 1) :]),
            "epoch": epoch,
            "tube_mask_ratio": tube_mask_ratio,
            "decoder_mask_ratio": decoder_mask_ratio,
        }
        progress_bar.set_postfix(**logs)
        if distributed: print(logs)

        if global_rank==0:
            # Plot progress (first sample in batch)
            with torch.no_grad():
                if utils.is_interactive() or wandb_log:
                    idx = 0
                    if epoch % 5 == 0:
                        decode_vis = torch.zeros_like(target_patches_vit)
                        decode_vis[:, decoder_mask] = output.to(decode_vis.device).to(decode_vis.dtype)
                        decoder_unpatches = rearrange(
                            decode_vis,
                            "b (f d h w) c -> b f d h w c",
                            d=img_size[0]//patch_size,
                            h=img_size[1]//patch_size,
                            w=img_size[2]//patch_size,
                        )
                        decoder_func = rearrange(
                            decoder_unpatches,
                            "b f d h w (pd ph pw pf c) -> b c (f pf) (d pd) (h ph) (w pw)",
                            b=batch_size,
                            f=num_frames,
                            d=img_size[0]//patch_size,
                            h=img_size[1]//patch_size,
                            w=img_size[2]//patch_size,
                            pd=patch_size,
                            ph=patch_size,
                            pw=patch_size,
                            pf=frame_patch_size,
                        )
                        orig_image = utils.reshape_to_2d(func[idx].clamp(0,1))
                        recon_image = utils.reshape_to_2d(decoder_func[idx].clamp(0,1))
    
                        combined_image = orig_image.clone()
                        combined_image[recon_image!=0] = recon_image[recon_image!=0]
                        
                        random_start = np.random.randint(recon_image.shape[1]-400)
                        orig_image = transforms.ToPILImage()(orig_image[:,random_start:random_start+400])
                        recon_image = transforms.ToPILImage()(recon_image[:,random_start:random_start+400])
                        combined_image = transforms.ToPILImage()(combined_image[:,random_start:random_start+400])
    
                        if wandb_log:
                            logs[f"train/orig"] = wandb.Image(orig_image, caption=f"epoch{epoch:03d}")
                            logs[f"train/recon"] = wandb.Image(recon_image, caption=f"epoch{epoch:03d}")
                            logs[f"train/combined"] = wandb.Image(combined_image, caption=f"epoch{epoch:03d}")
                        else:
                            # display(orig_image)
                            # display(recon_image)
                            display(combined_image)
    
            if wandb_log: wandb.log(logs)
            
        # wait for other GPUs to catch up if needed
        if distributed: dist.barrier()

        # Save model checkpoint
        if (ckpt_saving) and ((epoch % ckpt_interval == 0) or (epoch==num_epochs-1)):
            save_ckpt(model,"last")
        
        torch.cuda.empty_cache()

if distributed:
    dist.destroy_process_group()


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(recon_losses)
plt.title("Training re-construction losses")
plt.show()
if use_contrastive_loss:
    plt.figure(figsize=(8, 3))
    plt.plot(contrastive_losses)
    plt.title("Training contrastive losses")
    plt.show()


# In[ ]:




