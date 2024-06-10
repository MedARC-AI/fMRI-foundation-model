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
import shutil
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import utils
from models import *
import nibabel as nib
from nilearn import plotting

import schedulefree

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
# if utils.is_interactive():
#     raise ValueError()

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

from tqdm import tqdm

# Load parameters from yaml config
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

# create global variables from the config
for attribute_name in config.keys():
    globals()[attribute_name] = config[f'{attribute_name}']

data_type = torch.float32 # change depending on your mixed_precision
# batch_size = global_batch_size // num_devices
global_batch_size = batch_size * world_size

# FSDP Setup
if distributed:
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy
    import functools
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
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


print(config)

# seed all random functions
utils.seed_everything(seed)

outdir = os.path.abspath(f'../ckpts/{model_name}')
os.makedirs(outdir,exist_ok=True)
print("outdir", outdir)
print("global_batch_size", global_batch_size)
print("use_cls_token", use_cls_token)

if type(patch_size) == int:
    patch_size = [patch_size,patch_size,patch_size]
patch_depth = patch_size[0]
patch_height = patch_size[1]
patch_width = patch_size[2]

num_patches = int(
    (img_size[0] / patch_depth)
    * (img_size[1] / patch_height)
    * (img_size[2] / patch_width)
    * num_frames
)
num_patches_per_timepoint = num_patches // frame_patch_size
num_encoder_patches = int(np.floor((num_patches_per_timepoint * num_frames // frame_patch_size) * (1 - tube_start_masking_ratio)))
num_decoder_patches = int(np.floor((num_patches_per_timepoint * num_frames  // frame_patch_size) * (1 - decoder_mask_ratio)))
print("num_patches", num_patches)
print("num_patches_per_timepoint", num_patches_per_timepoint)
print("num_encoder_patches", num_encoder_patches)
print("num_decoder_patches", num_decoder_patches)


vit_size = {
    "encoder": encoder_model,
    "decoder": decoder_model
}
    
model = get_vit(
    size=vit_size,
    image_size=img_size,  # depth, height, width
    image_patch_size=(patch_depth,patch_height,patch_width),  # depth, height, width patch size
    frames=num_frames,
    frame_patch_size=frame_patch_size,
    channels=1,
    use_rope_emb=use_rope_emb,
    use_cls_token=use_cls_token,
    use_decoder_same_emb_dim=use_decoder_same_emb_dim,
    decoder_depth=decoder_depth
)
utils.count_params(model)

# function to select random num_frames from sample and obtain brain-positive patches
aug_transform = utils.DataPrepper(
    num_frames=num_frames*2,
    masking_strategy=masking_strategy,
    patch_depth=patch_depth,
    patch_height=patch_height,
    patch_width=patch_width,
    frame_patch_size=frame_patch_size,
    image_size=img_size
)

# test that the model works without error
model = model.to(device)
encoder_mask = torch.zeros(num_patches_per_timepoint).to(torch.bool)
encoder_mask[:num_encoder_patches] = True
decoder_mask = torch.zeros(num_patches_per_timepoint).to(torch.bool)
decoder_mask[-num_decoder_patches:] = True
decoder_mask[encoder_mask] = False
with torch.no_grad():
    print("\nencoder")
    encoder_out = model(
                torch.randn(batch_size, 1, num_frames, img_size[0], img_size[1], img_size[2]).to(device),
                encoder_mask=encoder_mask,
                verbose=True)
    if use_decoder:
        print("\ndecoder")
        decoder_out = model(
                    encoder_out, 
                    encoder_mask=encoder_mask, 
                    decoder_mask=decoder_mask, 
                    verbose=True)
    if use_cls_token:
        enc_cls_token = encoder_out[:, :1, :]
        encoder_patches = encoder_out[:, 1:, :]
        print("\nenc_cls_token", enc_cls_token.shape)
        print("encoder_patches", encoder_patches.shape)
        if use_decoder:
            dec_cls_token = decoder_out[:, :1, :]
            decoder_patches = decoder_out[:, 1:, :]
            print("dec_cls_token", dec_cls_token.shape)
            print("decoder_patches", decoder_patches.shape)

class LinearProbe(nn.Module):
    def __init__(self, input_dim, h=256, num_classes=8):
        super(LinearProbe, self).__init__()
        # self.classifier = nn.Linear(input_dim, num_classes)
        self.classifier = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        x = self.classifier(x)
        return x

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    print(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def filter_corrupted_images(sample):
    """If all the required files are not present don't use them."""
    correct_data = ("func.npy" in sample)
    return correct_data

### ================      Train Dataset and DataLoader    ====================
from braceexpand import braceexpand
print(train_urls)
if is_s3:
    expanded_urls = [f"pipe:aws s3 cp {url} -" for pattern in train_urls for url in braceexpand(pattern)]
else:
    expanded_urls = [str(url) for pattern in train_urls for url in braceexpand(pattern)]

train_data = (
    wds.WebDataset(expanded_urls, resampled=True, nodesplitter=wds.split_by_node, handler=log_and_continue)
    .shuffle(100, initial=100, rng=random.Random(seed))
    .select(filter_corrupted_images)
    .decode("torch")
)
train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)

### ================      Test Dataset and DataLoader    ====================
print(test_urls)
if is_s3:
    expanded_urls = [f"pipe:aws s3 cp {url} -" for pattern in test_urls for url in braceexpand(pattern)]
else:
    expanded_urls = [str(url) for pattern in train_urls for url in braceexpand(pattern)]

test_data = (
    wds.WebDataset(expanded_urls, resampled=True, nodesplitter=wds.split_by_node, handler=log_and_continue)
    .shuffle(100, initial=100, rng=random.Random(seed))
    .select(filter_corrupted_images)
    .decode("torch")
)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)


if distributed:    
    # my_auto_wrap_policy = functools.partial(
    #     size_based_auto_wrap_policy, min_num_params=200000
    # )
    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, 
        transformer_layer_cls={
            Attention, # <--- Your Transformer layer class
        },
    )
    print(f"\nPrepping FSDP on {global_rank} {node}...\n")
    model = model.to(device)
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
    print(f"\nSuccessfully loaded FSDP model to device on global_rank {global_rank}\n")

if use_contrastive_loss:
    model.simclr_handler = utils.SimCLRHandler(model.encoder_embed_dim).to(device)
if use_vic_loss:
    model.vicreg_handler = utils.VICRegHandler(model.encoder_embed_dim).to(device)

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
opt_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]

if distributed:
    max_lr = max_lr * global_batch_size
    print(f"multiply lr {max_lr} by global batch size: max_lr={max_lr}")

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)
# optimizer = schedulefree.AdamWScheduleFree(opt_grouped_parameters, lr=max_lr)

num_iterations_per_epoch = num_samples_per_epoch // global_batch_size
print("num_iterations_per_epoch", num_iterations_per_epoch)

probe_num_iterations_per_epoch = test_num_samples_per_epoch // global_batch_size
print("probe_num_iterations_per_epoch", probe_num_iterations_per_epoch)

total_steps = num_epochs * num_iterations_per_epoch * num_devices
print("total_steps", total_steps)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=total_steps,
    pct_start=10*num_iterations_per_epoch/total_steps,
    div_factor=25,
    final_div_factor=1000
)

print("\nDone with model preparations!")
num_params = utils.count_params(model)


def save_ckpt(model,tag="last"):
    if distributed: dist.barrier()
    model_states = model.state_dict()
    if global_rank == 0:
        os.makedirs(outdir,exist_ok=True)
        ckpt_path = outdir+f'/{tag}.pth'
        
        if tag == "last" and os.path.exists(ckpt_path):
            shutil.copyfile(os.path.join(outdir, f'{tag}.pth'), os.path.join(outdir, f'{tag}_old.pth'))
        # print(f'saving {ckpt_path}',flush=True)
        if tag=='last':
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_states,
                'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_states,
                }, ckpt_path)
            
        if tag == "last" and os.path.exists(os.path.join(outdir, f'{tag}_old.pth')):
            os.remove(os.path.join(outdir, f'{tag}_old.pth'))
    print(f"\n---saved {ckpt_path}!---\n")

if utils.is_interactive():
#     wandb_log = False
    ckpt_saving = False
if local_rank==0 and wandb_log: # only use main process for wandb logging
    import wandb
    wandb_project = 'fmri_foundation'
    print(f"wandb {wandb_project} run {model_name}")
    # need to configure wandb beforehand in terminal with "wandb init"!
    wandb_config = {
      "model_name": model_name,
      "global_batch_size": global_batch_size,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "num_samples_per_epoch": num_samples_per_epoch,
      "test_num_samples_per_epoch": test_num_samples_per_epoch,
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
      "contrastive_loss_weight": contrastive_loss_weight,
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


epoch = 0
lrs, train_losses, recon_losses, contrastive_losses, vic_losses = [], [], [], [], []
cos_sim_encoder_output, cos_sim_decoder_output, cos_sim_encoder_output_patchwise = [], [], []
probe_losses, probe_accs, test_losses, test_accs = [], [], [], []
cos_sim_encoder_output_patchwise_test, cos_sim_encoder_output_test = [], []


if masking_strategy=="MNI":
    from einops.layers.torch import Rearrange
    MNI_brain = nib.load("/weka/proj-fmri/paulscotti/fMRI-foundation-model/dataset_creation/afni_conversion/tpl-MNI152NLin2009cAsym_res-02_T1w_brain.nii.gz").get_fdata()
    brain_pos_voxels = MNI_brain[6:94,8:112,10:82]
    brain_pos_pats = Rearrange(
            "b c (f pf) (d pd) (h ph) (w pw) -> b f d h w (pd ph pw pf c)",
            pd=patch_depth,
            ph=patch_height,
            pw=patch_width,
            pf=1,
        )(torch.Tensor(brain_pos_voxels)[None,None,None])
    brain_pos_pats_vit = rearrange(brain_pos_pats, "b ... d -> b (...) d").mean(-1)[0]

# auto resume
if os.path.exists(os.path.join(outdir, 'last.pth')) or os.path.exists(os.path.join(outdir, 'last_old.pth')):
    if os.path.exists(os.path.join(outdir, 'last_old.pth')):
        if os.path.exists(os.path.join(outdir, 'last.pth')):
            # this is corrupted
            os.remove(os.path.join(outdir, f'last.pth'))
        # set last_old as last
        shutil.move(os.path.join(outdir, f'last_old.pth'), os.path.join(outdir, f'last.pth'))
    
    ckpt_path = os.path.join(outdir, 'last.pth')
    resume_from_ckpt = True


if resume_from_ckpt:
    print("\n---resuming from ckpt_path---\n", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    epoch = checkpoint['epoch']+1
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
    model.load_state_dict(checkpoint['model_state_dict'])
    total_steps_done = epoch*num_iterations_per_epoch
    for _ in range(total_steps_done):
        lr_scheduler.step()
    del checkpoint
    torch.cuda.empty_cache()

mse = nn.MSELoss()
l1 = nn.L1Loss()
crossentropy = nn.CrossEntropyLoss()
if use_contrastive_loss:
    contrastive_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs)
progress_bar = tqdm(range(epoch, num_epochs), disable=local_rank!=0, desc="Overall")
for epoch in progress_bar:
    # get the masking ratio for the current epoch
    tube_mask_ratio = utils.get_masking_ratio(
        current_epoch=epoch,
        total_epochs=num_epochs,
        start_masking_ratio=tube_start_masking_ratio,
        end_masking_ratio=tube_end_masking_ratio
    )
    with torch.cuda.amp.autocast(dtype=data_type):
        model.train()
        # optimizer.train()
        for train_i, batch in enumerate(train_dl):
            optimizer.zero_grad()

            input_func = batch['func.npy']

            subject_id = batch['subject_id.txt']
            subject_id = torch.Tensor([int(subject[-2:]) for subject in subject_id]).long()
            subject_id = torch.repeat_interleave(subject_id.long(), 2).to(device)
            # session_id = batch['session_id.txt']
            # session_id = torch.Tensor([int(session[-2:]) for session in session_id]).long().repeat(2).to(device)
            # session_id = torch.repeat_interleave(session_id.long(), 2)

            if masking_strategy=="None":
                func, _ = aug_transform(input_func)
                brain_pos_pats_vit = torch.ones(num_patches_per_timepoint)
            elif masking_strategy=="MNI":
                func, _ = aug_transform(input_func)
            else:
                func, brain_pos_voxels = aug_transform(input_func)
                brain_pos_pats = model.patchify(torch.Tensor(brain_pos_voxels)[None,None,None])
                brain_pos_pats_vit = rearrange(brain_pos_pats, "b ... d -> b (...) d").mean(-1)[0]

            func = func.reshape(-1, num_frames, 
                                func.shape[-3], func.shape[-2], func.shape[-1])
            func = func.unsqueeze(1).float().to(device).clamp(0,1)
            
            # create encoder and decoder masks
            rand_patches = torch.randperm(num_patches_per_timepoint)
            
            # encoder_mask1 = torch.zeros(num_patches_per_timepoint).to(torch.bool)
            # encoder_mask1[rand_patches[:num_encoder_patches]] = True
            # encoder_mask1 = encoder_mask1.tile(num_frames//frame_patch_size)

            # encoder_mask2 = torch.zeros(num_patches_per_timepoint).to(torch.bool)
            # encoder_mask2[rand_patches[num_encoder_patches:2*num_encoder_patches]] = True
            # encoder_mask2 = encoder_mask2.tile(num_frames//frame_patch_size)

            encoder_mask = torch.zeros(num_patches_per_timepoint).to(torch.bool)
            encoder_mask[rand_patches[:num_encoder_patches]] = True
            encoder_mask = encoder_mask.tile(num_frames//frame_patch_size)
            
            decoder_mask = torch.zeros(num_patches_per_timepoint).to(torch.bool)
            decoder_mask[rand_patches[2*num_encoder_patches:2*num_encoder_patches+num_decoder_patches]] = True
            decoder_mask = decoder_mask.tile(num_frames//frame_patch_size)

            # encode the tube patches
            # encoder_out1 = model(func, encoder_mask=encoder_mask1, device=device)
            # encoder_out2 = model(func, encoder_mask=encoder_mask2, device=device)
            # if use_cls_token:
            #     enc_cls_token1 = encoder_out1[:,:1,:]
            #     enc_cls_token2 = encoder_out2[:,:1,:]
            
            encoder_out = model(func, encoder_mask=encoder_mask, device=device)
            if use_cls_token:
                enc_cls_token = encoder_out[:,:1,:]

            if use_decoder:
                # decode both the encoder_out patches and masked decoder patches
                decoder_out = model(encoder_out, encoder_mask=encoder_mask, decoder_mask=decoder_mask, device=device)
                # subset only the reconstructed decoder patches
                output = decoder_out[:, -decoder_mask.sum():]

                # compare to ground truth and calculate loss
                target_patches = model.patchify(func)
                target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")
                target = target_patches_vit.to(device)[:, decoder_mask]

                target_mean = target.mean(0)
                target_std = target.std(0)
                target_normed = (target - target_mean) / (target_std + 1e-6)

                recon_loss = mse(output, target_normed)
                recon_losses.append(recon_loss.item())
                loss = recon_loss
            else:
                recon_loss = torch.nan
                recon_losses.append(recon_loss)
                loss = 0

            # old contrastive loss (simclr)
            if use_contrastive_loss and not use_vic_loss:
                # encode the decoder patches
                encoder_out2 = model(func, encoder_mask=decoder_mask, device=device)
                enc_cls_token2 = encoder_out2[:,:1,:]
                
                temp = contrastive_temps[epoch]

                all_cls = torch.cat([enc_cls_token, enc_cls_token2], dim=0)
                all_cls_proj = model.simclr_handler(all_cls)

                contr_loss = utils.SimCLRHandler.simclr_loss(all_cls_proj, temp)
                
                contrastive_losses.append(contr_loss.item())
                loss += (contr_loss * contrastive_loss_weight)
            # new loss
            elif use_vic_loss:
                full_mask = encoder_mask+decoder_mask
                encoder_out2 = model(func, encoder_mask=full_mask, device=device)
                enc_cls_token2 = encoder_out2[:,:1,:]

                # l1 = encoder_out[:, 1:, :]
                l1 = encoder_out
                l2 = encoder_out2[:, 1:, :]
                l2_sub = utils.VICRegHandler.filter_global_to_local(l2, encoder_mask, decoder_mask)
                l2_sub = torch.cat([l2[:, :1, :], l2_sub], dim=1)

                l1_proj = model.vicreg_handler(l1)
                l2_proj = model.vicreg_handler(l2_sub)

                vic_loss = utils.VICRegHandler.vicreg_loss(l1_proj, l2_proj, gamma=gamma) # gamma=0.5)

                if use_contrastive_loss:
                    temp = contrastive_temps[epoch]
                    all_cls = torch.cat([enc_cls_token, enc_cls_token2], dim=0)
                    all_cls_proj = model.simclr_handler(all_cls)

                    contr_loss = utils.SimCLRHandler.simclr_loss(all_cls_proj, temp)
                    contrastive_losses.append(contr_loss.item())
                else:
                    contr_loss = 0
                    contrastive_losses.append(0)
                
                vic_losses.append(vic_loss.item())
                loss += (contr_loss * contrastive_loss_weight + vic_loss * vic_loss_weight)
            else:
                vic_losses.append(0)
                contrastive_losses.append(0)


            cos_sim_encoder_output_patchwise.append(utils.patchwise_cosine_similarity(encoder_out)[~torch.eye(encoder_out.shape[1], dtype=bool)[None].expand(encoder_out.shape[0],-1,-1)].mean().item())
            cos_sim_encoder_output.append(utils.batchwise_cosine_similarity(encoder_out.flatten(1)/1e3,encoder_out.flatten(1)/1e3)[~torch.eye(len(encoder_out),dtype=torch.bool)].mean().item())
            if use_decoder:
                cos_sim_decoder_output.append(utils.batchwise_cosine_similarity(output,output)[~torch.eye(len(output),dtype=torch.bool)].mean().item())

            loss.backward()
            print(f'Cont loss: {loss.item():.4f}', end='\r')
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            if lr_scheduler is not None:
                lr_scheduler.step()
            train_losses.append(loss.item())

            if train_i >= (num_iterations_per_epoch-1):
                break
        
        logs = {
            "train/loss": np.mean(train_losses[-(train_i + 1) :]),
            "train/recon_losses": np.mean(recon_losses[-(train_i + 1) :]),
            "train/contrastive_losses": np.mean(contrastive_losses[-(train_i + 1) :]),
            "train/vic_losses": np.mean(vic_losses[-(train_i + 1) :]),
            "train/num_steps": len(recon_losses),
            "train/cos_sim_encoder_output": np.mean(cos_sim_encoder_output[-(train_i + 1) :]),
            "train/cos_sim_decoder_output": np.mean(cos_sim_decoder_output[-(train_i + 1) :]) if use_decoder else np.nan,
            "train/cos_sim_encoder_output_patchwise": np.mean(cos_sim_encoder_output_patchwise[-(train_i + 1) :]),
            "lr": np.mean(lrs[-(train_i + 1) :]),
            "epoch": epoch,
            "tube_mask_ratio": tube_mask_ratio,
            "decoder_mask_ratio": decoder_mask_ratio,
        }

        if epoch % probe_freq == probe_freq-1 or epoch == num_epochs-1:
            # reset linear_probe
            # if use_cls_token:
            #     linear_probe = LinearProbe((num_patches_per_timepoint+1)*model.encoder_embed_dim)
            # else:
            #     linear_probe = LinearProbe(num_patches_per_timepoint*model.encoder_embed_dim)
            linear_probe = LinearProbe(model.encoder_embed_dim)
            linear_probe = linear_probe.to(device)
            probe_opt_grouped_parameters = [
                {'params': [p for n, p in linear_probe.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
                {'params': [p for n, p in linear_probe.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]
            probe_optimizer = torch.optim.AdamW(probe_opt_grouped_parameters, lr=1e-3)
            probe_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                probe_optimizer,
                max_lr=1e-3,  #3e-5
                total_steps=probe_num_iterations_per_epoch*8
            )

            model.eval()
            # optimizer.eval()
            linear_probe.train()
            for probe_i, batch in enumerate(train_dl):
                probe_optimizer.zero_grad()

                input_func = batch['func.npy']

                subject_id = batch['subject_id.txt']
                subject_id = torch.Tensor([int(subject[-2:]) for subject in subject_id]).long()
                subject_id = torch.repeat_interleave(subject_id.long(), 2).to(device)

                func, _ = aug_transform(input_func)
                func = func.reshape(-1, num_frames, 
                                    func.shape[-3], func.shape[-2], func.shape[-1])
                func = func.unsqueeze(1).float().to(device).clamp(0,1)

                encoder_mask = torch.ones(num_patches_per_timepoint).to(torch.bool)
                encoder_mask = encoder_mask.tile(num_frames//frame_patch_size)

                for i in range(0,func.shape[0],8):
                    sub_func = func[i:i+8]
                    sub_subject_id = subject_id[i:i+8]

                    probe_i_inner = probe_i*8 + i//8

                    # encode the tube patches
                    with torch.no_grad():
                        encoder_out = model(sub_func, encoder_mask=encoder_mask, device=device)
                        encoder_cls = encoder_out[:,0,:]
                        # encoder_out = nn.functional.normalize(encoder_out,dim=-1)

                    # linear probe
                    subject_pred = linear_probe(encoder_cls)
                    probe_loss = crossentropy(subject_pred, sub_subject_id-1) # minus 1 because subject_id is 1-indexed

                    probe_accuracy = (torch.max(subject_pred,1).indices == (sub_subject_id-1)).sum() / len(sub_subject_id)
                    probe_accs.append(probe_accuracy.item())
                    probe_losses.append(probe_loss.item())

                    print(f'Probe: {probe_i_inner}, {probe_accuracy.item():.4f}, {probe_loss.item():.4f}')

                    probe_loss.backward()
                    probe_optimizer.step()
                    probe_scheduler.step()

                if probe_i >= (probe_num_iterations_per_epoch-1):
                    break

            for test_i, batch in enumerate(test_dl):
                input_func = batch['func.npy']

                subject_id = batch['subject_id.txt']
                subject_id = torch.Tensor([int(subject[-2:]) for subject in subject_id]).long()
                subject_id = torch.repeat_interleave(subject_id.long(), 2).to(device)

                func, _ = aug_transform(input_func)
                func = func.reshape(-1, num_frames, 
                                    func.shape[-3], func.shape[-2], func.shape[-1])
                func = func.unsqueeze(1).float().to(device).clamp(0,1)

                encoder_mask = torch.ones(num_patches_per_timepoint).to(torch.bool)
                encoder_mask = encoder_mask.tile(num_frames//frame_patch_size)

                # encode the tube patches
                with torch.no_grad():
                    encoder_out = model(func, encoder_mask=encoder_mask, device=device)
                    encoder_cls = encoder_out[:,0,:]
                    # encoder_out = nn.functional.normalize(encoder_out,dim=-1)

                # linear probe
                subject_pred = linear_probe(encoder_cls)
                test_loss = crossentropy(subject_pred, subject_id-1) # minus 1 because subject_id is 1-indexed

                test_accuracy = (torch.max(subject_pred,1).indices == (subject_id-1)).sum() / len(subject_id)
                test_accs.append(test_accuracy.item())
                test_losses.append(test_loss.item())

                cos_sim_encoder_output_patchwise_test.append(utils.patchwise_cosine_similarity(encoder_out)[~torch.eye(encoder_out.shape[1], dtype=bool)[None].expand(encoder_out.shape[0],-1,-1)].mean().item())
                cos_sim_encoder_output_test.append(utils.batchwise_cosine_similarity(encoder_out.flatten(1)/1e3,encoder_out.flatten(1)/1e3)[~torch.eye(len(encoder_out),dtype=torch.bool)].mean().item())

                print("test", test_i, test_accuracy.item(), test_loss.item())

                if test_i >= 1:
                    break

            logs.update({
                "train/probe_losses": np.mean(probe_losses[-(probe_i + 1) :]),
                "train/probe_accs": np.mean(probe_accs[-(probe_i + 1) :]),
                "test/probe_losses": np.mean(test_losses[-(test_i + 1) :]),
                "test/probe_accs": np.mean(test_accs[-(test_i + 1) :]),
                "test/cos_sim_encoder_output": np.mean(cos_sim_encoder_output_test[-(test_i + 1) :]),
                "test/cos_sim_encoder_output_patchwise": np.mean(cos_sim_encoder_output_patchwise_test[-(test_i + 1) :]),
            })
        
        progress_bar.set_postfix(**logs)
        if utils.is_interactive(): print(logs)

        # Plot progress (first sample in batch)
        with torch.no_grad():
            if use_decoder and (utils.is_interactive() or wandb_log):
                if epoch % viz_freq == viz_freq-1 or epoch == num_epochs-1:
                    output = (output * target_std) + target_mean
                    idx = 0
                    
                    decode_vis = torch.zeros_like(target_patches_vit)
                    decode_vis[:, decoder_mask] = output.to(decode_vis.device).to(decode_vis.dtype)
                    decoder_unpatches = rearrange(
                        decode_vis,
                        "b (f d h w) c -> b f d h w c",
                        d=img_size[0]//patch_depth,
                        h=img_size[1]//patch_height,
                        w=img_size[2]//patch_width,
                    )
                    decoder_func = rearrange(
                        decoder_unpatches,
                        "b f d h w (pd ph pw pf c) -> b c (f pf) (d pd) (h ph) (w pw)",
                        b=batch_size*2,
                        f=num_frames//frame_patch_size,
                        d=img_size[0]//patch_depth,
                        h=img_size[1]//patch_height,
                        w=img_size[2]//patch_width,
                        pd=patch_depth,
                        ph=patch_height,
                        pw=patch_width,
                        pf=frame_patch_size,
                    )
                    orig_image = utils.reshape_to_2d(func[idx])
                    recon_image = utils.reshape_to_2d(decoder_func[idx])

                    combined_image = orig_image.clone()
                    combined_image[recon_image!=0] = recon_image[recon_image!=0]

                    random_start = np.arange(3100,3450)
                    orig_image = transforms.ToPILImage()(orig_image[:,random_start])
                    recon_image = transforms.ToPILImage()(recon_image[:,random_start])
                    combined_image = transforms.ToPILImage()(combined_image[:,random_start])

                    if wandb_log:
                        logs[f"train/orig"] = wandb.Image(orig_image, caption=f"epoch{epoch:03d}")
                        logs[f"train/recon"] = wandb.Image(recon_image, caption=f"epoch{epoch:03d}")
                        logs[f"train/combined"] = wandb.Image(combined_image, caption=f"epoch{epoch:03d}")
                    else:
                        if epoch==0:
                            print("orig_image")
                            display(orig_image)
                            print("recon_image")
                            display(recon_image)
                            print("combined_image")
                        display(combined_image)

    if wandb_log: wandb.log(logs)

    # Save model checkpoint
    if (ckpt_saving) and ((epoch % ckpt_interval == ckpt_interval-1) or (epoch==num_epochs-1)):
        save_ckpt(model,"last")

    # wait for other GPUs to catch up if needed
    if distributed: dist.barrier()
    torch.cuda.empty_cache()
        
if distributed:
    dist.destroy_process_group()