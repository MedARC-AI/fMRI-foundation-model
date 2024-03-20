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
import gc
from einops import rearrange
from einops.layers.torch import Rearrange
import time
import random
import h5py
import webdataset as wds
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import utils
from models import *
from mindeye_models import *
import nibabel as nib
from nilearn import plotting
# from accelerate import Accelerator

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
    print("NOT distributed")
    device = torch.device('cuda')

print("PID of this process =",os.getpid())
print("device =", device, "distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)


# In[2]:


# Load VJEPA parameters from yaml config
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

# create global variables from the config
for attribute_name in config.keys():
    globals()[attribute_name] = config[f'{attribute_name}']

# Load MindEye parameters from yaml config (will override any params with same name)
mindeye_config = yaml.load(open('mindeye_config.yaml', 'r'), Loader=yaml.FullLoader)

# create global variables from the config
for attribute_name in mindeye_config.keys():
    globals()[attribute_name] = mindeye_config[f'{attribute_name}']


# In[3]:


# First use "accelerate config" in terminal for setup
global_batch_size = batch_size * num_devices
data_type = torch.float16 # change depending on your mixed_precision
# accelerator = Accelerator(mixed_precision="fp16")
global_batch_size = batch_size * num_devices
print("batch_size: ", batch_size)
print("global_batch_size: ", global_batch_size)


# # Configuration

# In[4]:


print("vjepa config\n\n",config)
print("mindeye_config\n",mindeye_config)

if utils.is_interactive():
    ckpt_saving = False
    wandb_log = False

# seed all random functions
utils.seed_everything(seed)

outdir = os.path.abspath(f'../ckpts/{config["model_name"]}')
print("outdir", outdir)

num_patches = int(
    (image_size[0] / patch_size)
    * (image_size[1] / patch_size)
    * (image_size[2] / patch_size)
    * num_frames
)
print("num_patches", num_patches)


# # Load VJEPA model

# In[5]:


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


# In[6]:


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
utils.count_params(model)


# In[7]:


# load from ckpt
mae_ckpt_pth = os.path.abspath(f'../ckpts/{mae_model_name}/last.pth')
print("mae_ckpt_pth", mae_ckpt_pth)

checkpoint = torch.load(mae_ckpt_pth)
state_dict = checkpoint['model_state_dict']

model.load_state_dict(state_dict, strict=True)


# In[8]:


# set foundation model to evaluation
model.eval()
model.requires_grad_(False)
model.to(device)
pass


# # Setup MindEye Model

# In[9]:


nsddata_raw_stimuli = pd.read_csv(f"{nsd_raw_path}/nsddata_rawdata.csv")
TR_delay = 3 # to account for bold hrf
train_TRs = np.round(nsddata_raw_stimuli[nsddata_raw_stimuli['shared1000'] == False]['global_TR_onsets'].values + TR_delay).astype(np.int32)
test_TRs = np.round(nsddata_raw_stimuli[nsddata_raw_stimuli['shared1000'] == True]['global_TR_onsets'].values + TR_delay).astype(np.int32)


# In[10]:


# Load 73k NSD images
f = h5py.File(f'{nsd_image_path}/coco_images_224_float16.hdf5', 'r')
images = f['images'][:] 
images = torch.Tensor(images).to("cpu").to(data_type)
print("Loaded all 73k possible NSD images!", images.shape)

# Load MindEye hdf5
f = h5py.File(f'{nsd_raw_path}/subj01_mnidata.h5', 'r') #subj01_rawdata_old.h5
mindeye_global_trs = f['global_trs'][:]
mindeye_funcs = f['funcs']


# In[11]:


clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)
clip_seq_dim, clip_emb_dim = 256, 1664


# In[34]:


subj = s = 1
subj_list = [subj]

num_samples_per_epoch = (750*num_sessions) // num_devices
num_iterations_per_epoch = num_samples_per_epoch // batch_size
print("num_iterations_per_epoch: ", num_iterations_per_epoch)
train_data = {}
train_dl = {}

print(f"Training with {num_sessions} sessions")
train_url = f"{nsd_wds_path}/subj0{s}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
print(train_url)
    
train_data[f'subj0{s}'] = wds.WebDataset(train_url,resampled=True,nodesplitter=utils.my_split_by_node)\
                    .shuffle(750, initial=1500, rng=random.Random(42))\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
# train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
train_dl[f'subj0{s}'] = wds.WebLoader(
    train_data[f'subj0{s}'].batched(batch_size), 
    pin_memory=True,
    shuffle=False,
    batch_size=None,
    num_workers=num_workers, 
    persistent_workers=num_workers>0,
).with_epoch(num_iterations_per_epoch)

if global_rank==0:
    print("Loaded all subj train dls and betas!\n")
    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{nsd_wds_path}/subj0{subj}/new_test/" + "0.tar"
    print(test_url)
    test_data = wds.WebDataset(test_url,resampled=True,nodesplitter=utils.my_split_by_node)\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    # test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
    test_dl = wds.WebLoader(
        test_data.batched(num_test),
        pin_memory=True,
        shuffle=False,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=num_workers>0,
    ).with_epoch(num_iterations_per_epoch)
    print(f"Loaded test dl for subj{subj}!\n")


# In[13]:


# from accelerate.state import AcceleratorState
# try:
#     AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = global_batch_size
#     print("deepspeed reconfigured, train_micro_batch_size_per_gpu = ", global_batch_size)
# except:
#     print("skipping deepspeed reconfiguration...")


# In[14]:


mindeye = MindEyeModule()
mindeye.ridge = RidgeRegression(np.array([in_dim]), out_features=hidden_dim)
mindeye.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=4, drop=drop,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, clip_scale=1)
utils.count_params(mindeye.ridge)
utils.count_params(mindeye.backbone)
utils.count_params(mindeye)


# In[15]:


if distributed: 
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=200000
    )
    print(f"\nPrepping FSDP on {global_rank} {node}...\n")
    mindeye = FSDP(
        mindeye,
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
else:
    mindeye=mindeye.to(device)


# In[16]:


no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
opt_grouped_parameters = [
    {'params': [p for n, p in mindeye.ridge.named_parameters()], 'weight_decay': 1e-2},
    {'params': [p for n, p in mindeye.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in mindeye.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

total_steps = num_epochs * num_iterations_per_epoch
print("total_steps", total_steps)
pct_start = 2/num_epochs if num_epochs>1 else 1.
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=total_steps,
)

print("\nDone with model preparations!")
num_params = utils.count_params(model)


# In[17]:


def save_ckpt(tag="last"):
    ckpt_path = outdir+f'/{tag}/downstream'
    os.makedirs(ckpt_path,exist_ok=True)
    accelerator.save_model(model, ckpt_path, max_shard_size="2GB", safe_serialization=True)
    print(f"\n---saved {ckpt_path}!---\n")
        
def save_progress(tag="last"):
    if accelerator.is_main_process:
        ckpt_path = outdir+f'/{tag}/downstream'
        torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "losses": losses,
                    "test_losses": test_losses,
                    "lrs": lrs,
                },
                os.path.join(ckpt_path, f"params.pt"),
            )


# In[18]:


if global_rank==0 and wandb_log: # only use main process for wandb logging
    import wandb
    wandb_project = 'found_downstream'
    print(f"wandb {wandb_project} run {model_name}")
    # need to configure wandb beforehand in terminal with "wandb init"!
    wandb_config = {
      "model_name": model_name,
      "mae_model_name": mae_model_name,
      "global_batch_size": global_batch_size,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "num_sessions": num_sessions,
      "num_samples_per_epoch": num_samples_per_epoch,
      "in_dim": in_dim,
      "hidden_dim": hidden_dim,
      "mixup_pct": mixup_pct,
      "num_params": num_params,
      "max_lr": max_lr,
      "ckpt_interval": ckpt_interval,
      "ckpt_saving": ckpt_saving,
      "seed": seed,
      "distributed": distributed,
      "num_devices": num_devices,
      "world_size": world_size,
      "train_url": train_url,
      "test_url": test_url,
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


# # Train MindEye with Foundation Model

# In[19]:


epoch = 0
losses, test_losses, lrs = [], [], []
best_test_loss = 1e9
torch.cuda.empty_cache()


# In[20]:


import nibabel as nib
MNI_brain = nib.load("/weka/proj-fmri/paulscotti/fMRI-foundation-model/dataset_creation/afni_conversion/tpl-MNI152NLin2009cAsym_res-02_T1w_brain.nii.gz").get_fdata()
brain_pos_voxels = MNI_brain[6:94,8:112,10:82]
brain_pos_voxels[6:94,:(112-60),10:62] = 0
brain_pos_pats = model.patchify(torch.Tensor(brain_pos_voxels)[None,None,None])
brain_pos_pats_vit = rearrange(brain_pos_pats, "b ... d -> b (...) d").mean(-1)[0]
tube_mask = (brain_pos_pats_vit > 0).tile(num_frames)
print(tube_mask.sum(), tube_mask.sum() / len(tube_mask))


# In[21]:


# resume from ckpt (e.g., if you are resuming from a run that got pre-empted)
load_progress = False
if wandb_log:
    if wandb.run.resumed:
        load_checkpoint_in_model(model, outdir+"/last")
        load_progress = True
elif resume_from_ckpt: # if resuming without using wandb
    load_checkpoint_in_model(model, outdir+"/last")
    load_progress = True
    
if load_progress:
    ckpt_path = outdir+'/last'
    prev_params = torch.load(ckpt_path+"/params.pt")
    optimizer.load_state_dict(prev_params["optimizer"])
    lr_scheduler.load_state_dict(prev_params["scheduler"])
    epoch = prev_params["epoch"]
    losses = prev_params["losses"]
    test_losses = prev_params["test_losses"]
    lrs = prev_params["lrs"]
    print("Loaded model params from", ckpt_path, "at epoch", epoch)


# In[22]:


train_dls = [train_dl[f'subj0{s}'] for s in subj_list]
# mindeye, optimizer, *train_dls, lr_scheduler = accelerator.prepare(
#     mindeye, optimizer, *train_dls, lr_scheduler
# )
# # skipping test_dl because we just use local_rank=0 for validation


# In[ ]:


print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch, num_epochs), disable=global_rank!=0)
mse = nn.MSELoss()
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

test_image=None
num_test_eval=batch_size # should instead be 300 to mimic MindEye2 retrieval evaluation, but this leads to OOM

for epoch in progress_bar:
    print(f"epoch {epoch}")
    mindeye.train()

    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    test_fwd_percent_correct = 0.
    test_bwd_percent_correct = 0.
    loss_clip_total = 0.
    test_loss_clip_total = 0.

    # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
    voxel_iters = {} # empty dict because diff subjects have differing # of voxels
    image_iters = torch.zeros(num_iterations_per_epoch, batch_size*len(subj_list), 3, 224, 224).float()
    annot_iters = {}
    perm_iters, betas_iters, select_iters = {}, {}, {}
    for s, train_dl in enumerate(train_dls):
        with torch.cuda.amp.autocast(dtype=data_type):
            for iter, (behav0, past_behav0, future_behav0, old_behav0) in enumerate(train_dl):
                image0 = images[behav0[:,0,0].cpu().long()].float()
                image_iters[iter,s*batch_size:s*batch_size+batch_size] = image0

                if epoch==0 and iter==0: print("\nreached start of train!\n")

                # if images are not fully preloaded, then can do this inefficient but more memory friendly approach
                # for ib,b in enumerate(behav0[:,0,0].cpu().long()):
                #     if ib==0:
                #         image0 = torch.Tensor(images[[b]])
                #     else:
                #         image0 = torch.vstack((image0, torch.Tensor(images[[b]])))
                # image_iters[iter,s*batch_size:s*batch_size+batch_size] = image0
                
                # get the corresponding raw voxel time series
                for ib,b in enumerate(behav0[:,0,5].cpu().long().numpy()):
                    tr = (nsddata_raw_stimuli[nsddata_raw_stimuli['global_trial'].isin([b.item()])]['global_TR_onsets'].values + TR_delay).astype(np.int32).item()
                    if ib==0:
                        voxels_raw = mindeye_funcs[tr-2:tr+2][None][None]
                    else:
                        voxels_raw = np.vstack((voxels_raw, mindeye_funcs[tr-2:tr+2][None][None]))
                voxels_raw = torch.Tensor(voxels_raw).to(device)
                
                ## Process it through pretrained VJEPA Y-Encoder (encodes the entire voxels_raw) ##
                encoder_out = model(voxels_raw, encoder_mask=tube_mask, encoder_type = "y")
                voxel0 = encoder_out.flatten(1).unsqueeze(1).cpu()
                
                assert len(voxel0) == batch_size

                if epoch < int(mixup_pct * num_epochs):
                    voxel0, perm, betas, select = utils.mixco(voxel0)
                    perm_iters[f"subj0{subj_list[s]}_iter{iter}"] = perm
                    betas_iters[f"subj0{subj_list[s]}_iter{iter}"] = betas
                    select_iters[f"subj0{subj_list[s]}_iter{iter}"] = select

                voxel_iters[f"subj0{subj_list[s]}_iter{iter}"] = voxel0

                if iter >= num_iterations_per_epoch:
                    break

    # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
    for train_i in range(num_iterations_per_epoch):
        with torch.cuda.amp.autocast(dtype=data_type):
            optimizer.zero_grad()
            loss=0.
            if train_i==0 and epoch==0: print("\nreached start of 2nd train loop!\n")

            voxel_list = [voxel_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
            image = image_iters[train_i].detach()
            image = image.to(device)

            clip_target = clip_img_embedder(image)
            assert not torch.any(torch.isnan(clip_target))

            if epoch < int(mixup_pct * num_epochs):
                perm_list = [perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                perm = torch.cat(perm_list, dim=0)
                betas_list = [betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                betas = torch.cat(betas_list, dim=0)
                select_list = [select_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                select = torch.cat(select_list, dim=0)

            voxel_ridge_list = [mindeye.ridge(voxel_list[si],si) for si,s in enumerate(subj_list)]
            voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

            backbone, clip_voxels = mindeye.backbone(voxel_ridge)

            clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
            clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

            if epoch < int(mixup_pct * num_epochs):                
                loss_clip = utils.mixco_nce(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006,
                    perm=perm, betas=betas, select=select)
            else:
                epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                loss_clip = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=epoch_temp)

            loss_clip_total += loss_clip.item()
            loss += loss_clip

            # forward and backward top 1 accuracy        
            labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
            fwd_percent_correct += utils.topk(utils.prenormed_batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
            bwd_percent_correct += utils.topk(utils.prenormed_batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

            utils.check_loss(loss)
            loss.backward()
            # accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            lr_scheduler.step()

    mindeye.eval()
    if global_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
            for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):  
                loss=0.     
                if test_i==0 and epoch==0: print("\nreached test!\n")

                coco_idx = behav[:,0,0].cpu().long()
                _,test_indices = np.unique(coco_idx, return_index=True)
                test_indices = np.random.permutation(test_indices)[:num_test_eval]
                image = images[coco_idx[test_indices]].float().to(device)
                
                # get the corresponding raw voxel time series
                for ib,b in enumerate(behav[test_indices,0,5].cpu().long().numpy()):
                    tr = (nsddata_raw_stimuli[nsddata_raw_stimuli['global_trial'].isin([b.item()])]['global_TR_onsets'].values + TR_delay).astype(np.int32).item()
                    if ib==0:
                        voxels_raw = mindeye_funcs[tr-2:tr+2][None][None]
                    else:
                        voxels_raw = np.vstack((voxels_raw, mindeye_funcs[tr-2:tr+2][None][None]))
                voxels_raw = torch.Tensor(voxels_raw).to(device)
                
                ## Process it through pretrained MAE ##
                encoder_out = model(voxels_raw, encoder_mask=tube_mask, encoder_type = "y")
                voxel = encoder_out.flatten(1).unsqueeze(1)

                assert len(image) == num_test_eval

                clip_target = clip_img_embedder(image.float())

                voxel_ridge = mindeye.ridge(voxel,0) # 0th index of subj_list
                backbone, clip_voxels = mindeye.backbone(voxel_ridge)

                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                
                loss_clip = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006)

                test_loss_clip_total += loss_clip.item()
                loss += loss_clip

                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                test_fwd_percent_correct += utils.topk(utils.prenormed_batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                test_bwd_percent_correct += utils.topk(utils.prenormed_batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
                
                utils.check_loss(loss)                
                test_losses.append(loss.item())

        logs = {"train/loss": np.mean(losses[-(train_i+1):]),
            "test/loss": np.mean(test_losses[-(test_i+1):]),
            "train/lr": lrs[-1],
            "train/num_steps": len(losses),
            "test/num_steps": len(test_losses),
            "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
            "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
            "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
            "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
            "train/loss_clip_total": loss_clip_total / (train_i + 1),
            "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
        }
        progress_bar.set_postfix(**logs)
        if not distributed: print(logs)
        if wandb_log: wandb.log(logs)
            
    # Save model checkpoint
    if (ckpt_saving) and (epoch % ckpt_interval == 0):
        save_ckpt()

    # wait for other GPUs to catch up if needed
    # accelerator.wait_for_everyone()
    dist.barrier()
    torch.cuda.empty_cache()
    gc.collect()


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(losses)
plt.title("Training losses")
plt.show()

plt.figure(figsize=(8, 3))
plt.plot(test_losses)
plt.title("Test losses")
plt.show()

