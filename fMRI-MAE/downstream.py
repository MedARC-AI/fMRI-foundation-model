# Import packages and setup gpu configuration.
# This code block shouldnt need to be adjusted!
import os
import shutil
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
from models import *
from mindeye_models import *
import nibabel as nib
from nilearn import plotting
from functools import partial

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

### Multi-GPU config ###
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

global_rank = os.getenv('RANK')
if global_rank is None:
    global_rank = 0
else:
    global_rank = int(global_rank)
print(f"GLOBAL RANK={global_rank}")

from tqdm import tqdm

# Load parameters from yaml config
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

# create global variables from the config
for attribute_name in config.keys():
    globals()[attribute_name] = config[f'{attribute_name}']
    
# Load MindEye parameters from yaml config (will override any params with same name)
mindeye_config = yaml.load(open('mindeye_config.yaml', 'r'), Loader=yaml.FullLoader)

# create global variables from the config
for attribute_name in mindeye_config.keys():
    globals()[attribute_name] = mindeye_config[f'{attribute_name}']

data_type = torch.float32 # change depending on your mixed_precision

batch_size = global_batch_size // num_devices
print("batch_size", batch_size)
    
# First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading
from accelerate import Accelerator
accelerator = Accelerator(split_batches=False, mixed_precision="fp16")

from accelerate.state import AcceleratorState
try:
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = batch_size
    print("deepspeed reconfigured, train_micro_batch_size_per_gpu = ", batch_size)
except:
    print("skipping deepspeed reconfiguration...")

print("PID of this process =",os.getpid())
device = accelerator.device
print("device:",device)
world_size = accelerator.state.num_processes
num_workers = num_devices
print(accelerator.state)

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
print = accelerator.print # only print if local_rank=0

print("mae config\n\n",config)
print("mindeye_config\n",mindeye_config)

# if utils.is_interactive():
#     ckpt_saving = False
#     wandb_log = False

# seed all random functions
utils.seed_everything(seed)

mae_ckpt_pth = os.path.abspath(f'../ckpts/{mae_model_name}/last.pth')
print("mae_ckpt_pth", mae_ckpt_pth)

outdir = os.path.abspath(f'../ckpts/{model_name}')
os.makedirs(outdir,exist_ok=True)
print("outdir", outdir)

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
print("num_patches", num_patches)

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


def save_ckpt(tag):
    ckpt_path = outdir+f'/{tag}.pth'
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        if tag == "last" and os.path.exists(ckpt_path):
            shutil.copyfile(ckpt_path, os.path.join(outdir, f'{tag}_old.pth'))
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
        if tag == "last" and os.path.exists(os.path.join(outdir, f'{tag}_old.pth')):
            os.remove(os.path.join(outdir, f'{tag}_old.pth'))
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")

def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,outdir=outdir,multisubj_loading=False): 
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir+'/last.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading: # remove incompatible ridge layer that will otherwise error
        state_dict.pop('ridge.linears.0.weight',None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint['epoch']
        print("Epoch",epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint


checkpoint = torch.load(mae_ckpt_pth, map_location=device)
try:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
except:
    pass

# set foundation model to evaluation
model.eval()
model.requires_grad_(False)
model.to(device)
pass


nsddata_raw_stimuli = pd.read_csv(f"{nsd_raw_path}/nsddata_rawdata.csv")
TR_delay = 3 # to account for bold hrf
train_TRs = np.round(nsddata_raw_stimuli[nsddata_raw_stimuli['shared1000'] == False]['global_TR_onsets'].values + TR_delay).astype(np.int32)
test_TRs = np.round(nsddata_raw_stimuli[nsddata_raw_stimuli['shared1000'] == True]['global_TR_onsets'].values + TR_delay).astype(np.int32)


# Load 73k NSD images
f = h5py.File(f'{nsd_image_path}/coco_images_224_float16.hdf5', 'r')
images = f['images'][:] 
images = torch.Tensor(images).to("cpu").to(data_type)
print("Loaded all 73k possible NSD images!", images.shape)

# Load MindEye hdf5
f = h5py.File(f'{nsd_raw_path}/subj01_mnidata.h5', 'r') #subj01_rawdata_old.h5
mindeye_global_trs = f['global_trs'][:]
mindeye_funcs = f['funcs']


# clip_img_embedder = FrozenOpenCLIPImageEmbedder(
#     arch="ViT-bigG-14",
#     version="laion2b_s39b_b160k",
#     output_tokens=True,
#     only_tokens=True,
# )
# clip_seq_dim = 256
# clip_emb_dim = 1664
# clip_img_embedder.to(device)

clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=False,
    only_tokens=False,
    init_device=device,
    device=device,
    cache_dir="/weka/proj-fmri/shared/cache"
)
clip_seq_dim = 1
clip_emb_dim = 1280
clip_img_embedder.to(device)


subj = s = 1
subj_list = [subj]

# if multi_subject:
#     nsessions_allsubj=np.array([40, 40, 32, 30, 40, 32, 40, 30])
#     num_samples_per_epoch = (750*40) // num_devices 
# else:
#     num_samples_per_epoch = (750*num_sessions) // num_devices 

# num_samples_per_epoch = 1024 #(750*num_sessions) // num_devices

print("dividing batch size by subj_list, which will then be concatenated across subj during training...") 
batch_size = batch_size // len(subj_list)
num_iterations_per_epoch = num_samples_per_epoch // (batch_size*len(subj_list))
print("batch_size =", batch_size, "num_iterations_per_epoch =",num_iterations_per_epoch, "num_samples_per_epoch =",num_samples_per_epoch)

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
    num_test=300  # 3000
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
).with_epoch(10)
print(f"Loaded test dl for subj{subj}!\n")


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x

# class RidgeRegression(torch.nn.Module):
#     # make sure to add weight_decay when initializing optimizer
#     def __init__(self, input_sizes, out_features, seq_len=1): 
#         super(RidgeRegression, self).__init__()
#         self.seq_len = seq_len
#         self.out_features = out_features
#         self.linears = torch.nn.ModuleList([
#                 torch.nn.Linear(input_size, out_features) for input_size in input_sizes
#             ])
#     def forward(self, x, subj_idx):
#         out = torch.cat([self.linears[subj_idx](x[:,seq]).unsqueeze(1) for seq in range(self.seq_len)], dim=1)
#         return out
    
class MLP(torch.nn.Module):
    def __init__(self, input_sizes, out_features, seq_len=0): 
        super(MLP, self).__init__()
        self.input_sizes = input_sizes[0]
        self.out_features = out_features
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.input_sizes),
            nn.GELU(),
            nn.Linear(self.input_sizes, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, out_features)
        )
    def forward(self, x, z=None):
        out = self.mlp(x[:,0])
        return out

class ReversibleBrainNetwork(nn.Module):
    def __init__(self, out_dim=768, in_dim=15724, h=4096, n_blocks=4, norm_type='bn', act_first=True, 
                 encoder_tokens=257, reverse=True, **kwargs):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        # self.temp = nn.Parameter(torch.tensor(.006))
        start_dim, end_dim = (out_dim*encoder_tokens, in_dim) if reverse else (in_dim, out_dim*encoder_tokens)
        self.lin0 = nn.Sequential(
            nn.Linear(start_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(drop)
            ) for _ in range(n_blocks)
        ])
        
        self.lin1 = nn.Linear(h, end_dim, bias=True)
        self.n_blocks = n_blocks
        
    def forward(self, x, *args, **kwargs):
        x = self.lin0(x.flatten(1))  # bs, h
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)
        return x


mindeye = MindEyeModule()
# mindeye.ridge = MLP(np.array([in_dim]), out_features=clip_emb_dim*clip_seq_dim)
mindeye.ridge = ReversibleBrainNetwork(in_dim=in_dim, h=hidden_dim, out_dim=clip_emb_dim, 
                                       encoder_tokens=1, norm_type='ln', act_first=False, reverse=False)

# mindeye.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, n_blocks=4, drop=drop,
#                           clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, clip_scale=1)
utils.count_params(mindeye.ridge)
# utils.count_params(mindeye.backbone)
utils.count_params(mindeye)


no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
opt_grouped_parameters = [
    {'params': [p for n, p in mindeye.ridge.named_parameters()], 'weight_decay': 1e-2},
    # {'params': [p for n, p in mindeye.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    # {'params': [p for n, p in mindeye.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

total_steps = iters_scale_factor * num_epochs * num_iterations_per_epoch
print("total_steps", total_steps)
pct_start = 2/num_epochs if num_epochs>1 else 1.
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=total_steps,
    pct_start=0.1
)

print("\nDone with model preparations!")
num_params = utils.count_params(mindeye)


if accelerator.is_main_process and wandb_log: # only use main process for wandb logging
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


epoch = 0
losses, test_losses, lrs = [], [], []
best_test_loss = 1e9
torch.cuda.empty_cache()

# if masking_strategy=="MNI":
from einops.layers.torch import Rearrange

MNI_brain = nib.load("/weka/proj-fmri/paulscotti/fMRI-foundation-model/dataset_creation/afni_conversion/tpl-MNI152NLin2009cAsym_res-02_T1w_brain.nii.gz").get_fdata()
brain_pos_voxels = utils.crop_or_pad(torch.from_numpy(MNI_brain[6:94,8:112,10:82]), img_size)
# brain_pos_voxels = MNI_brain[10:90,12:108,14:78]

# brain_pos_voxels = brain_pos_voxels[:,30:31,:]

brain_pos_pats = Rearrange(
        "b c (f pf) (d pd) (h ph) (w pw) -> b f d h w (pd ph pw pf c)",
        pd=patch_depth,
        ph=patch_height,
        pw=patch_width,
        pf=1,
    )(brain_pos_voxels[None,None,None])

brain_pos_pats_vit = rearrange(brain_pos_pats, "b ... d -> b (...) d").mean(-1)[0]
    
tube_mask = torch.zeros(num_patches // num_frames).to(torch.bool)
batch_positive_approx = (brain_pos_pats_vit > 0)
mask_idx_candidates = torch.where(batch_positive_approx)[0]
mask_idx_candidates = mask_idx_candidates[torch.randperm(len(mask_idx_candidates))]
tube_idx = mask_idx_candidates[:int(num_patches / num_frames * (1 - tube_end_masking_ratio))]
tube_mask[tube_idx] = True
tube_mask = tube_mask.tile(num_frames//frame_patch_size)

# load multisubject stage1 ckpt if set
if multisubject_ckpt!="None" and not resume_from_ckpt:
    load_ckpt("last",outdir=multisubject_ckpt,load_lr=False,load_optimizer=False,load_epoch=False,strict=False,multisubj_loading=True)
    
# load saved ckpt model weights into current model
if resume_from_ckpt:
    load_ckpt("last",load_lr=True,load_optimizer=True,load_epoch=True)
elif wandb_log:
    if wandb.run.resumed:
        if os.path.exists(os.path.join(outdir, 'last.pth')) or os.path.exists(os.path.join(outdir, 'last_old.pth')):
            if os.path.exists(os.path.join(outdir, 'last_old.pth')):
                if os.path.exists(os.path.join(outdir, 'last.pth')):
                    # this is corrupted
                    os.remove(os.path.join(outdir, f'last.pth'))
                # set last_old as last
                shutil.move(os.path.join(outdir, f'last_old.pth'), os.path.join(outdir, f'last.pth'))
            
            # ckpt_path = os.path.join(outdir, 'last.pth')
            # resume_from_ckpt = True
            load_ckpt("last",load_lr=True,load_optimizer=True,load_epoch=True)

train_dls = [train_dl[f'subj0{s}'] for s in subj_list]

mindeye, optimizer, *train_dls, lr_scheduler = accelerator.prepare(mindeye, optimizer, *train_dls, lr_scheduler)
# leaving out test_dl since we will only have local_rank 0 device do evals

print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch, num_epochs), disable=not accelerator.is_main_process)
mse = nn.MSELoss()
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

bn = nn.BatchNorm1d(512,affine=False).to(device)

test_image=None
num_test_eval=batch_size # should instead be average same-image 300 to mimic MindEye2 retrieval evaluation

for epoch in progress_bar:
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
            for iter, (behav0, past_behav0, future_behav0, old_behav0) in enumerate(tqdm(train_dl,total=num_iterations_per_epoch)):
                image0 = images[behav0[:,0,0].cpu().long()].float()
                image_iters[iter,s*batch_size:s*batch_size+batch_size] = image0

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
                voxels_raw = torch.Tensor(voxels_raw).clamp(0,1).to(device)
                voxels_raw = utils.crop_or_pad(voxels_raw, img_size)
                
                ## Process it through pretrained MAE ##
                encoder_out = model(voxels_raw, encoder_mask=tube_mask)
                # encoder_out = bn(encoder_out)
                
                voxel0 = encoder_out.flatten(1).unsqueeze(1).cpu()
                # voxel0 = nn.functional.normalize(voxel0,dim=-1).cpu()
                
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
    for train_i in range(iters_scale_factor * num_iterations_per_epoch):
        with torch.cuda.amp.autocast(dtype=data_type):
            optimizer.zero_grad()
            loss=0.

            actual_idx = train_i % num_iterations_per_epoch

            voxel_list = [voxel_iters[f"subj0{s}_iter{actual_idx}"].detach().to(device) for s in subj_list]
            image = image_iters[actual_idx].detach()
            image = image.to(device)

            clip_target = clip_img_embedder(image)
            assert not torch.any(torch.isnan(clip_target))

            if epoch < int(mixup_pct * num_epochs):
                perm_list = [perm_iters[f"subj0{s}_iter{actual_idx}"].detach().to(device) for s in subj_list]
                perm = torch.cat(perm_list, dim=0)
                betas_list = [betas_iters[f"subj0{s}_iter{actual_idx}"].detach().to(device) for s in subj_list]
                betas = torch.cat(betas_list, dim=0)
                select_list = [select_iters[f"subj0{s}_iter{actual_idx}"].detach().to(device) for s in subj_list]
                select = torch.cat(select_list, dim=0)

            voxel_ridge_list = [mindeye.ridge(voxel_list[si],si) for si,s in enumerate(subj_list)]
            clip_voxels = torch.cat(voxel_ridge_list, dim=0)

#             backbone, clip_voxels = mindeye.backbone(voxel_ridge)

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
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            lr_scheduler.step()

    mindeye.eval()
    if local_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
            for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):  
                loss=0.     

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
                voxels_raw = torch.Tensor(voxels_raw).clamp(0,1).to(device)
                voxels_raw = utils.crop_or_pad(voxels_raw, img_size)
                
                ## Process it through pretrained MAE ##
                encoder_out = model(voxels_raw, encoder_mask=tube_mask)
                # encoder_out = bn(encoder_out)
                
                voxel = encoder_out.flatten(1).unsqueeze(1)
                # voxel = nn.functional.normalize(voxel,dim=-1)

                assert len(image) == num_test_eval

                clip_target = clip_img_embedder(image.float())
                
                clip_voxels_norm = nn.functional.normalize(voxel.flatten(1), dim=-1)

                clip_voxels = mindeye.ridge(voxel,0) # 0th index of subj_list
                
#                 backbone, clip_voxels = mindeye.backbone(voxel_ridge)

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
            if wandb_log: wandb.log(logs)
            
    # Save model checkpoint
    if (ckpt_saving) and (epoch % ckpt_interval == 0):
        save_ckpt('last')

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    gc.collect()

