#!/usr/bin/env python
# coding: utf-8

# # Configuration

# In[68]:


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
import gc
from einops import rearrange
from einops.layers.torch import Rearrange
import time
import random
import h5py
import webdataset as wds
import matplotlib.pyplot as plt
print("reached")
import torch
print("reached1")
import torch.nn as nn
print("reached2")
import torch.nn.functional as F
print("reached3")
from torchvision import transforms
print("reached4")
import utils
from models import *
import nibabel as nib
from nilearn import plotting
import schedulers

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
print("reached5")
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
# batch_size = global_batch_size // num_devices
global_batch_size = batch_size * num_devices

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
        device = torch.cuda.current_device() #torch.device('cuda',local_rank)
        print(f"\nSuccessfully set cuda:{local_rank} | global_rank{global_rank} | node{node}")
    except Exception as error:        
        print(f"\nFAILED TO SET DEVICE cuda:{local_rank} | global_rank{global_rank} | node{node}")
        print("An exception occurred:", error)
    dist.barrier()
    print("passed barrier\n")
else:
    device = torch.device('cuda')

print("PID of this process =",os.getpid())
print("device =", device, "distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)


# In[69]:


print(config)

# seed all random functions
utils.seed_everything(seed)

outdir = os.path.abspath(f'../ckpts/{model_name}')
print("outdir", outdir)

if use_contrastive_loss:
    global_batch_size = global_batch_size // 2 # contrastive loss doubles the batch size with the same samples and different masks
print("global_batch_size", global_batch_size)

use_cls_token = True if use_contrastive_loss else use_cls_token
print("use_cls_token", use_cls_token)

num_patches = int(
    (image_size[0] / patch_size)
    * (image_size[1] / patch_size)
    * (image_size[2] / patch_size)
    * num_frames
)
num_patches_per_timepoint = num_patches // num_frames
num_encoder_patches = int(num_patches_per_timepoint * (1 - x_encoder_start_masking_ratio) * num_frames)
num_decoder_patches = int(num_patches_per_timepoint * (1 - y_encoder_mask_ratio) * num_frames)
print("num_patches", num_patches)
print("num_encoder_patches", num_encoder_patches)
print("num_decoder_patches", num_decoder_patches)


# # Test to Check Model Architecture

# In[70]:


image_depth, image_height, image_width = image_size
image_patch_size=(patch_size,patch_size,patch_size)
patch_depth, patch_height, patch_width = image_patch_size
x_encoder = Transformer(
    embed_dim,
    depth,
    num_heads,
    dim_head,
    mlp_dim,
    use_rope=use_rope_emb,
    grid_time=num_frames // frame_patch_size,
    grid_depth=image_depth // patch_depth,
    grid_height=image_height // patch_height,
    grid_width=image_width // patch_width,
    cls_token=use_cls_token,
)
print("x_encoder")
print(utils.count_params(x_encoder))
y_encoder = Transformer(
    embed_dim,
    depth,
    num_heads,
    dim_head,
    mlp_dim,
    use_rope=use_rope_emb,
    grid_time=num_frames // frame_patch_size,
    grid_depth=image_depth // patch_depth,
    grid_height=image_height // patch_height,
    grid_width=image_width // patch_width,
    cls_token=use_cls_token,
)
print("y_encoder")
print(utils.count_params(y_encoder))
predictor = Transformer(
    embed_dim,
    depth,
    num_heads,
    dim_head,
    mlp_dim,
    use_rope=use_rope_emb,
    grid_time=num_frames // frame_patch_size,
    grid_depth=image_depth // patch_depth,
    grid_height=image_height // patch_height,
    grid_width=image_width // patch_width,
    cls_token=use_cls_token,
)
print("predictor")
print(utils.count_params(predictor))


# In[71]:


model = SimpleViT(
    x_encoder=x_encoder,
    y_encoder=y_encoder,
    predictor=predictor,
    image_size=image_size, 
    image_patch_size=image_patch_size, 
    num_frames=num_frames,
    frame_patch_size=frame_patch_size,
    channels=1,
    use_rope_emb=use_rope_emb,
    use_cls_token=use_cls_token,
)
if not distributed:
    model = model.to(device)
utils.count_params(model)


# In[72]:


aug_transform = utils.DataPrepper(
    num_frames=num_frames,
    masking_strategy=masking_strategy,
    patch_depth=patch_size,
    patch_height=patch_size,
    patch_width=patch_size,
    frame_patch_size=frame_patch_size,
)


# # Model Preparation

# In[73]:


from dataloader import create_dataset, create_loader

if not is_s3:
    print(train_urls)
    if distributed: dist.barrier()
    train_dp = create_dataset(train_urls, 
                              is_s3=is_s3, 
                              sample_shuffle=1, shard_shuffle=100)
    train_dl = create_loader(train_dp, batch_size=batch_size, num_workers=num_workers)
else:
    print("Dataloading from s3!")
    train_urls = s3_train_urls
    print(train_urls)
    train_dp = create_dataset(train_urls, 
                              is_s3=is_s3, 
                              sample_shuffle=1, shard_shuffle=100)
    train_dl = create_loader(train_dp, batch_size=batch_size, num_workers=num_workers)


# In[74]:


if not distributed:
    num_it = 2
    print(f"Yielding {num_it} batches")
    
    for i, batch in enumerate(train_dl):
        print("iter",i)
        input_func = batch['func.npy']
        if i >= (num_it-1):
            break
    
    print("Done!")
    print("input_func", input_func.shape)


# In[75]:


if distributed:    
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=200000
    )
    print(f"\nPrepping FSDP on {global_rank} {node}...\n")
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
    dist.barrier()
    print(f"\nSuccessfully passed barrier! {global_rank}\n")


# In[77]:


no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
opt_grouped_parameters = [
    {'params': [p for n, p in model.x_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.predictor.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
] #we only update the xencoder and predictor, the yencoder is updated through a moving average of the xencoder
model.y_encoder.requires_grad_(False)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        utils.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)

for m in model.x_encoder.modules():
    init_weights(m)
for m in  model.predictor.modules():
    init_weights(m)

model.y_encoder = copy.deepcopy(model.x_encoder)

for m in model.x_encoder.modules():
    if isinstance(m, torch.nn.Linear):
        print(m.weight, "h")
        break

for m in model.y_encoder.modules():
    if isinstance(m, torch.nn.Linear):
        print(m.weight)
        break

for m in model.predictor.modules():
    if isinstance(m, torch.nn.Linear):
        print(m.weight)
        break
    

    
optimizer = torch.optim.AdamW(opt_grouped_parameters)
num_iterations_per_epoch = num_samples_per_epoch // global_batch_size
print("num_iterations_per_epoch", num_iterations_per_epoch)
total_steps = num_epochs * num_iterations_per_epoch * num_devices
print("total_steps", total_steps)

print("\nDone with model preparations!")
num_params = utils.count_params(model)

#momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(num_iterations_per_epoch*num_epochs*ipe_scale)
                          #for i in range(int(num_iterations_per_epoch*num_epochs*ipe_scale)+1))
momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(total_steps*ipe_scale)
                          for i in range(int(total_steps*ipe_scale)+1))
count=0
print("\nmomentum_scheduler set")

#lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #optimizer,
    #max_lr=max_lr,
    #total_steps=total_steps,
#)

lr_scheduler = schedulers.WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * num_iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * num_iterations_per_epoch),
    )
print("\nlr_scheduler set")


# In[66]:


if masking_strategy=="MNI":
    MNI_brain = nib.load("/scratch/gpfs/ks9249/fMRI-foundation-model/dataset_creation/afni_conversion/tpl-MNI152NLin2009cAsym_res-02_T1w_brain.nii.gz").get_fdata()
    brain_pos_voxels = MNI_brain[6:94,8:112,10:82]
    brain_pos_pats = model.patchify(torch.Tensor(brain_pos_voxels)[None,None,None])
    brain_pos_pats_vit = rearrange(brain_pos_pats, "b ... d -> b (...) d").mean(-1)[0]


# In[52]:


default_ckpt_path = outdir+f'/last.pth'

def save_ckpt(model,tag="last"):
    if distributed: dist.barrier()
    model_states = model.state_dict()
    if global_rank == 0:
        ckpt_path = outdir+f'/{tag}.pth'
        os.makedirs(outdir,exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_states,
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
        print(f"\n---saved {ckpt_path}!---\n")
        # save config.yaml copy
        with open(f'{outdir}/config.yaml', 'w') as file:
            yaml.dump(config, file)

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


# In[11]:


import os
os.environ["WANDB_MODE"]="offline"

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
      "depth": depth,
      "mlp_dim": mlp_dim,
      "x_encoder_start_masking_ratio": x_encoder_start_masking_ratio,
      "x_encoder_end_masking_ratio": x_encoder_end_masking_ratio,
      "y_encoder_mask_ratio": y_encoder_mask_ratio,
      "num_frames": num_frames,
      "patch_size": patch_size,
      "frame_patch_size": frame_patch_size,
      "use_contrastive_loss": use_contrastive_loss,
      "use_cls_token": use_cls_token,
      "constrastive_loss_weight": constrastive_loss_weight,
      "num_params": num_params,
      "ckpt_interval": ckpt_interval,
      "ckpt_saving": ckpt_saving,
      "seed": seed,
      "distributed": distributed,
      "num_devices": num_devices,
      "world_size": world_size,
      "train_urls": train_urls,
      "is_s3": is_s3,
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

# In[ ]:


epoch = 0
lrs, recon_losses, contrastive_losses, test_losses,train_reg, test_reg = [], [], [], [], [], []
best_test_loss = 1e9
torch.cuda.empty_cache()


# In[ ]:


if resume_from_ckpt is True:
    if os.path.exists(default_ckpt_path):
        print(f"Resuming from {default_ckpt_path}...")
        model, optimizer, lr_scheduler, epoch = resume_ckpt()


# In[ ]:


def loss_fn(z, h):
    loss = 0.
    # Compute loss and accumulate for each mask-enc/mask-pred pair
    for zi, hi in zip(z, h):
        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
    loss /= len(z)
    return loss

def reg_fn(z):
    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

def cos_fn(z):
    # Flatten z from [batch, channels, height, width] to [batch, -1]
    z_flat = z.flatten(1)
    
    # Compute normalized vectors to prevent division by zero
    z_norm = torch.nn.functional.normalize(z_flat, p=2, dim=1)
    
    # Compute cosine similarity matrix
    cossim_matrix = torch.mm(z_norm, z_norm.transpose(0, 1))
    
    # Compute the sum of cosine similarities of distinct pairs
    cossim_sum = (cossim_matrix.sum() - len(z))/2
    
    # Compute the total number of distinct pairs
    cossim_num = len(z) * (len(z) - 1) / 2
    
    # Return the average cosine similarity for distinct pairs
    return cossim_sum / cossim_num

l1 = nn.L1Loss() #Following VJEPA architecture, which uses L1 loss not L2 loss

if use_contrastive_loss:
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # learned logit scale
    
if distributed: dist.barrier()
progress_bar = tqdm(range(epoch, num_epochs), disable=global_rank!=0)
for epoch in progress_bar:
    # get the masking ratio for the current epoch
    tube_mask_ratio = utils.get_masking_ratio(
        current_epoch=epoch, 
        total_epochs=num_epochs, 
        start_masking_ratio=x_encoder_start_masking_ratio, 
        end_masking_ratio=x_encoder_end_masking_ratio
    )
    with torch.cuda.amp.autocast(dtype=data_type):
        model.train()
        for train_i, batch in enumerate(train_dl):
            optimizer.zero_grad()

            input_func = batch['func.npy']
            
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
                
            func = func.unsqueeze(1).to(device)

            # create tube mask (i.e., a mask that is the same for all frames/timepoints)
            tube_mask = torch.zeros(num_patches // num_frames).to(torch.bool)
            batch_positive_approx = (brain_pos_pats_vit > 0)
            mask_idx_candidates = torch.where(batch_positive_approx)[0]
            mask_idx_candidates = mask_idx_candidates[torch.randperm(len(mask_idx_candidates))]
            tube_idx = mask_idx_candidates[:int(num_patches / num_frames * (1 - tube_mask_ratio))]
            tube_mask[tube_idx] = True
            tube_mask = tube_mask.tile(num_frames)

            # print("before encoder");utils.print_cuda_memory_usage()
            
            # feed into x-encoder
            xencoder_out = model(func, encoder_mask=tube_mask, encoder_type = "x")
            # print("x_encoder");utils.print_cuda_memory_usage()
            
            # feed entire func into y-encoder
            yencoder_out = model(func, encoder_mask=tube_mask, encoder_type = "y")
            # print("y_encoder");utils.print_cuda_memory_usage()
            
            # feed output of x-encoder into predictor
            predictor_out = model(xencoder_out, encoder_mask=tube_mask, encoder_type="p")
            # print("predictor");utils.print_cuda_memory_usage()


            # compare output of predictor to output of y-encoder and calculate L1 Loss
            loss_jepa = l1(predictor_out,yencoder_out)  # jepa prediction loss
            loss_reg = cos_fn(yencoder_out)
            loss = loss_jepa + reg_coeff * loss_reg
            #loss = l1(predictor_out,yencoder_out)

            
            # contrastive loss
            if use_contrastive_loss:
                n_b = len(func) // 2
                cls_token1 = enc_cls_token[:n_b, 0, :]  # first half of batch, cls_token shape B, 1, d_model
                cls_token2 = enc_cls_token[n_b:, 0, :]
                contrastive_loss = utils.contrastive_loss(cls_token1, cls_token2, temperature=logit_scale)
                loss += constrastive_loss_weight * contrastive_loss
                contrastive_losses.append(contrastive_loss.item())


            

            if train_i==0 and epoch==0:
                print("calculated first loss")
            if train_i==1 and epoch==0:
                print("reached train_i=1")
            if train_i==0 and epoch==1:
                print("reached epoch1")
            
            # backwards + step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            recon_losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]["lr"])
            train_reg.append(loss_reg.item())
            
            # update y-encoder using exponential-moving average of x-encoder params to prevent collapse
            m = next(momentum_scheduler)
            with torch.no_grad():
                for param_q, param_k in zip(model.x_encoder.parameters(), model.y_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

            if train_i==0 and epoch==0:
                print("finished first epoch!")

            if train_i >= (num_iterations_per_epoch-1):
                print("train_i", train_i, "local_rank", local_rank, "global_rank", global_rank)
                break

        logs = {
            "train/loss": np.mean(recon_losses[-(train_i + 1) :]),
            "train/num_steps": len(recon_losses),
            "lr": np.mean(lrs[-(train_i + 1) :]),
            "epoch": epoch,
            "tube_mask_ratio": tube_mask_ratio,
            "train/loss_reg": np.mean(train_reg[-(train_i + 1) :]),
        }
        progress_bar.set_postfix(**logs)
        if distributed: print(logs)
            
        if global_rank==0:
            if wandb_log: wandb.log(logs)
                    
        # Save model checkpoint
        if (ckpt_saving) and ((epoch % ckpt_interval == 0) or (epoch==num_epochs-1)):
            save_ckpt(model,"last")
            
        # wait for other GPUs to catch up if needed
        if distributed: dist.barrier()
        torch.cuda.empty_cache()
        gc.collect()

if distributed:
    dist.barrier()
    dist.destroy_process_group()

