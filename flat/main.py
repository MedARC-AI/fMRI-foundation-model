import os
import shutil
import sys
import json
import yaml
import numpy as np
import copy
import math
import time
import random
from tqdm import tqdm
import webdataset as wds
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
import utils
from flat_models import *

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
# following fixes a Conv3D CUDNN_NOT_SUPPORTED error
torch.backends.cudnn.benchmark = True

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

# Load parameters from yaml config
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

print("\n__CONFIG__")
for attribute_name in config.keys():
    print(f"{attribute_name} = {config[attribute_name]}")
    globals()[attribute_name] = config[f'{attribute_name}']
print("\n")

# Create outdir for ckpt and config.yaml
outdir = os.path.abspath(f'checkpoints/{model_name}')
print("outdir", outdir)

# Load previous config.yaml if available
if os.path.exists(f"{outdir}/config.yaml"):
    config = yaml.load(open(f"{outdir}/config.yaml", 'r'), Loader=yaml.FullLoader)
    print(f"Loaded config.yaml from ckpt folder {outdir}")

    # create global variables from the config
    print("\n__REPLACING_CONFIG__")
    for attribute_name in config.keys():
        print(f"{attribute_name} = {config[attribute_name]}")
        globals()[attribute_name] = config[f'{attribute_name}']
    print("\n")

data_type = torch.float32 # change depending on your mixed_precision
global_batch_size = batch_size * world_size

# FSDP Setup
if distributed:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy
    import functools
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
    print(f"setting device to cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda',local_rank)
    dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
    print(f"\nSuccessfully set cuda:{local_rank} | global_rank{global_rank} | node{node}")
    dist.barrier() 
    print(f"global_rank{global_rank} passed barrier")
else:
    device = torch.device('cuda')

print("PID of this process =",os.getpid())
print("device =", device, "distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)

# seed all random functions
utils.seed_everything(seed + global_rank)

from util.hcp_flat import load_hcp_flat_mask
from util.hcp_flat import create_hcp_flat
from util.losses import *
import util.visualize as vis

if utils.is_interactive(): # Use less samples per epoch for debugging
    num_samples_per_epoch = 2000
    test_num_samples_per_epoch = 2000

model = mae_vit_small_fmri(
    decoder_embed_dim=decoder_embed_dim,
    t_patch_size=t_patch_size,
    pred_t_dim=pred_t_dim,
    decoder_depth=4,
    cls_embed=cls_embed,
    norm_pix_loss=norm_pix_loss,
    no_qkv_bias=no_qkv_bias,
    sep_pos_embed=sep_pos_embed,
    trunc_init=trunc_init,
)

if use_contrastive_loss:
    model.simclr_handler = SimCLRHandler(model.embed_dim).to(device)
if use_vic_loss:
    model.vicreg_handler = VICRegHandler(model.embed_dim).to(device)

# state = torch.load("checkpoints/checkpoint-00099.pth", map_location="cpu")
# model.load_state_dict(state["model"])

num_batches = num_samples_per_epoch // (num_devices * batch_size)
test_num_batches = test_num_samples_per_epoch // (num_devices * batch_size)
print("num_batches", num_batches)
print("test_num_batches", test_num_batches)

## Train ##
train_dataset = create_hcp_flat(root=hcp_flat_path, 
                    training=True, frames=num_frames)
train_dl = wds.WebLoader(
    train_dataset.batched(batch_size, partial=False),
    batch_size=None,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)
train_dl = train_dl.with_epoch(num_batches)

## Test ##
test_dataset = create_hcp_flat(root=hcp_flat_path,
                    training=False, frames=num_frames)
test_dl = wds.WebLoader(
    test_dataset.batched(batch_size, partial=False),
    batch_size=None,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)
test_dl = test_dl.with_epoch(test_num_batches)

print(f"\nChecking distributed setup on global_rank {global_rank}...")
from util.video_vit import Attention
if distributed:
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
else:
    print(f"\nNot training distributed! global_rank {global_rank}")
    model.to(device)

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
opt_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.05},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]

lr = base_lr * global_batch_size / 256
print(f"multiply base lr {base_lr} by effective batch size {global_batch_size}")
print(f"lr = {lr}")

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=lr, betas=(0.9, 0.95))

def adjust_learning_rate(optimizer, epoch, warmup_epochs=5, min_lr=0.0):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr_ = lr * epoch / warmup_epochs
    else:
        lr_ = min_lr + (lr - min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - warmup_epochs)
                / (num_epochs - warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_
    return lr_

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
        
        # save the config.yaml
        if not os.path.exists(f"{outdir}/config.yaml"):
            with open(f"{outdir}/config.yaml", 'w') as file:
                yaml.dump(config, file)
            print(f"saved {outdir}/config.yaml!")

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

epoch = 0
if resume_from_ckpt:
    print("\n---resuming from ckpt_path---\n", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    epoch = checkpoint['epoch']+1
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
    model.load_state_dict(checkpoint['model_state_dict'])
    # total_steps_done = epoch*num_iterations_per_epoch
    # for _ in range(total_steps_done):
    #     lr_scheduler.step()
    del checkpoint
    torch.cuda.empty_cache()

if utils.is_interactive():
    wandb_log = False
    ckpt_saving = False
if local_rank==0 and wandb_log: # only use main process for wandb logging
    import wandb
    wandb_project = 'fMRI-foundation-model'
    print(f"wandb {wandb_project} run {model_name}")
    # need to configure wandb beforehand in terminal with "wandb init"!
    wandb_config = {
      "model_name": model_name,
      "global_batch_size": global_batch_size,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "cls_embed": cls_embed,
      "norm_pix_loss": norm_pix_loss,
      "t_patch_size": t_patch_size,
      "pred_t_dim": pred_t_dim,
      "mask_ratio": mask_ratio,
      "num_frames": num_frames,
      "sep_pos_embed": sep_pos_embed,
      "decoder_embed_dim": decoder_embed_dim,
      "use_contrastive_loss": use_contrastive_loss,
      "num_params": num_params,
      "base_lr": base_lr,
      "lr": lr,
      "num_samples_per_epoch": num_samples_per_epoch,
      "test_num_samples_per_epoch": test_num_samples_per_epoch,
      "num_epochs": num_epochs,
      "grad_clip": grad_clip,
      "ckpt_interval": ckpt_interval,
      "ckpt_saving": ckpt_saving,
      "print_interval": print_interval,
      "seed": seed,
      "distributed": distributed,
      "num_devices": num_devices,
      "world_size": world_size,
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

# lrs, train_losses, test_losses = [], [], []
# train_losses1, train_losses2, train_losses3, train_losses4 = [], [], [], []

epoch = 0
lrs, train_losses, recon_losses, contrastive_losses, vic_losses = [], [], [], [], []
cos_sim_encoder_output, cos_sim_decoder_output, cos_sim_encoder_output_patchwise = [], [], []
probe_losses, probe_accs, test_losses, test_accs = [], [], [], []
cos_sim_encoder_output_patchwise_test, cos_sim_encoder_output_test = [], []

mse = nn.MSELoss()
l1 = nn.L1Loss()
crossentropy = nn.CrossEntropyLoss()

if use_contrastive_loss:
    contrastive_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs)

grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
progress_bar = tqdm(range(epoch, num_epochs), disable=local_rank!=0, desc="Overall")
for epoch in progress_bar:
    model.train()
    for train_i, batch in enumerate(train_dl):
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, train_i / num_batches + epoch)

        input_func = batch[1]

        input_func = input_func.to(dtype=data_type, device=device, non_blocking=True)
        if len(input_func.shape) == 6:
            b, r, c, t, h, w = input_func.shape
            input_func = input_func.reshape(b * r, c, t, h, w)

        ids_shuffle, ids_restore = get_ids_shuffle(input_func.shape[0], input_func.device, model)

        # adjust number to keep relative to image mask
        if model.img_mask is not None:
            len_keep = int(model.patch_embed.t_grid_size * model.n_mask_patches * (1 - mask_ratio))
        else:
            len_keep = int(ids_restore.shape[1] * (1 - mask_ratio))

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        lats = model.forward_encoder_with_mask(input_func, ids_keep)

        if use_decoder:
            mask = torch.ones_like(ids_restore)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            
            pred = model.forward_decoder(lats, ids_restore)
            recon_loss = model.forward_loss(input_func, pred, mask)
            
            recon_losses.append(recon_loss.item())
            loss = recon_loss
        else:
            recon_loss = torch.nan
            recon_losses.append(recon_loss)
            loss = 0
        
        if use_contrastive_loss and not use_vic_loss:
            ids_keep2 = ids_shuffle[:, :2*len_keep]
            lats2 = model.forward_encoder_with_mask(input_func, ids_keep2)
            
            enc_cls_token = lats[:, :1]
            enc_cls_token2 = lats2[:,:1]
            
            temp = contrastive_temps[epoch]

            all_cls = torch.cat([enc_cls_token, enc_cls_token2], dim=0)
            all_cls_proj = model.simclr_handler(all_cls)

            contr_loss = SimCLRHandler.simclr_loss(all_cls_proj, temp)
            
            contrastive_losses.append(contr_loss.item())
            loss += (contr_loss * contrastive_loss_weight)
        # new loss
        elif use_vic_loss:
            ids_keep2 = ids_shuffle[:, :2*len_keep]
            lats2 = model.forward_encoder_with_mask(input_func, ids_keep2)

            enc_cls_token = lats[:, :1]
            enc_cls_token2 = lats2[:,:1]

            l1 = lats
            l2 = lats2[:, :lats.shape[1]]

            l1_proj = model.vicreg_handler(l1)
            l2_proj = model.vicreg_handler(l2)

            vic_loss = VICRegHandler.vicreg_loss(l1_proj, l2_proj, gamma=gamma, lamda=lamda, mu=mu, nu=nu, 
                                                        rand_frac=rand_frac, use_vic_cls=use_vic_cls)

            if use_contrastive_loss:
                temp = contrastive_temps[epoch]
                all_cls = torch.cat([enc_cls_token, enc_cls_token2], dim=0)
                all_cls_proj = model.simclr_handler(all_cls)

                contr_loss = SimCLRHandler.simclr_loss(all_cls_proj, temp)
                contrastive_losses.append(contr_loss.item())
            else:
                contr_loss = 0
                contrastive_losses.append(0)
            
            vic_losses.append(vic_loss.item())
            loss += (contr_loss * contrastive_loss_weight + vic_loss * vic_loss_weight)
        else:
            vic_losses.append(0)
            contrastive_losses.append(0)

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        torch.cuda.synchronize()

        cos_sim_encoder_output_patchwise.append(utils.patchwise_cosine_similarity(lats)[~torch.eye(lats.shape[1], dtype=bool)[None].expand(lats.shape[0],-1,-1)].mean().item())
        cos_sim_encoder_output.append(utils.batchwise_cosine_similarity(lats.flatten(1)/1e3,lats.flatten(1)/1e3)[~torch.eye(len(lats),dtype=torch.bool)].mean().item())
        if use_decoder:
            cos_sim_decoder_output.append(utils.batchwise_cosine_similarity(pred, pred)[~torch.eye(len(pred),dtype=torch.bool)].mean().item())

        lrs.append(optimizer.param_groups[0]["lr"])
        train_losses.append(loss.item())
        
        if ((train_i%print_interval)==0 or (train_i==num_batches-1)) and train_i>0:
            print(f"Ep. {epoch} | loss {np.mean(train_losses[-print_interval:]):.3f} | lr {optimizer.param_groups[0]['lr']} | {train_i}/{num_batches}")
            if wandb_log: 
                logs = {"train/loss": np.mean(train_losses[-print_interval:])}
                # epoch_1000x as x-axis calibrates different curves when batch size changes
                epoch_1000x = int((train_i / num_batches + epoch) * 1000)
                wandb.log(logs, step=epoch_1000x)

    if utils.is_interactive() or wandb_log:
        print(f"Ep. {epoch} | loss {np.mean(train_losses[-print_interval:]):.3f} | lr {optimizer.param_groups[0]['lr']} | {train_i}/{num_batches}")
        with torch.no_grad():
            if norm_pix_loss:
                normed_input_func, patch_info = model.patchify(input_func, alter_patch_info=False, return_patch_info=True)
                target_mean = normed_input_func.mean(dim=-1, keepdim=True)
                target_var = normed_input_func.var(dim=-1, keepdim=True)
                normed_input_func = (normed_input_func - target_mean) / (target_var + 1.0e-6) ** 0.5
                normed_input_func = model.unpatchify(normed_input_func, patch_info=patch_info)

                vis_out = vis.plot_mask_pred(
                    model, normed_input_func, pred, mask, 
                    mean=0.5, std=0.2, 
                )
            else:
                vis_out = vis.plot_mask_pred(
                    model, input_func, pred, mask, 
                    mean=0.5, std=0.2, 
                )

            if wandb_log:
                logs = {"train/mask_pred": wandb.Image(vis_out)}
                epoch_1000x = int((train_i / num_batches + epoch) * 1000)
                wandb.log(logs, step=epoch_1000x)
            else:
                display(vis_out)
        
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
    }
        
    # Evaluate performance on held-out test dataset
    model.eval()
    with torch.no_grad():
        for test_i, batch in enumerate(test_dl):
            input_func = batch[1]

            input_func = input_func.to(dtype=data_type, device=device, non_blocking=True)
            if len(input_func.shape) == 6:
                b, r, c, t, h, w = input_func.shape
                input_func = input_func.reshape(b * r, c, t, h, w)

            if not use_contrastive_loss:
                loss, pred, mask, latent = model(input_func, mask_ratio=mask_ratio, use_contrastive_loss=use_contrastive_loss)
            else:
                loss1, loss2, loss3, pred1, pred2, mask1, mask2, true_mask, latent1, latent2 = model(input_func, mask_ratio=mask_ratio, use_contrastive_loss=use_contrastive_loss)
                pred, mask, latent = pred1, mask1, latent1
                
                # contrastive loss
                temp = contrastive_temps[epoch]
                logits = (nn.functional.normalize(latent1[:,:1].flatten(1),dim=-1) @
                            nn.functional.normalize(latent2[:,:1].flatten(1),dim=-1).T) / temp
                labels = torch.arange(len(logits)).long().to(device)
                contr_loss = (crossentropy(logits, labels) + crossentropy(logits.T, labels)) / 2
                
                loss = loss1 + loss2 + loss3 + contr_loss

            test_losses.append(loss.item())

            cos_sim_encoder_output_patchwise_test.append(utils.patchwise_cosine_similarity(latent)[~torch.eye(latent.shape[1], dtype=bool)[None].expand(latent.shape[0],-1,-1)].mean().item())
            cos_sim_encoder_output_test.append(utils.batchwise_cosine_similarity(latent.flatten(1)/1e3, latent.flatten(1)/1e3)[~torch.eye(len(latent),dtype=torch.bool)].mean().item())

            if test_i%print_interval==0 and test_i>0:
                print(f"Test | loss {np.mean(test_losses[-print_interval:]):.3f} | {test_i}/{test_num_batches}")

    print(f"Test | iter {test_i} | loss {np.mean(test_losses[-test_i:]):.3f}")
    if wandb_log: 
        logs.update({
            "test/loss": np.mean(test_losses[-test_i:]),
            "test/cos_sim_encoder_output": np.mean(cos_sim_encoder_output_test[-(test_i + 1) :]),
            "test/cos_sim_encoder_output_patchwise": np.mean(cos_sim_encoder_output_patchwise_test[-(test_i + 1) :]),
        })
        wandb.log(logs)

    # Plot progress (first sample in batch)
    if utils.is_interactive() or wandb_log:
        with torch.no_grad():
            if norm_pix_loss:
                normed_input_func, patch_info = model.patchify(input_func, alter_patch_info=False, return_patch_info=True)
                target_mean = normed_input_func.mean(dim=-1, keepdim=True)
                target_var = normed_input_func.var(dim=-1, keepdim=True)
                normed_input_func = (normed_input_func - target_mean) / (target_var + 1.0e-6) ** 0.5
                normed_input_func = model.unpatchify(normed_input_func, patch_info=patch_info)

                vis_out = vis.plot_mask_pred(
                    model, normed_input_func, pred, mask, 
                    mean=0.5, std=0.2, 
                )
            else:
                vis_out = vis.plot_mask_pred(
                    model, input_func, pred, mask, 
                    mean=0.5, std=0.2, 
                )

            if wandb_log:
                logs = {"test/mask_pred": wandb.Image(vis_out)}
                wandb.log(logs)
            else:
                display(vis_out)

    # Save model checkpoint
    if ckpt_saving and epoch>0 and ((epoch % ckpt_interval == 0) or (epoch==num_epochs-1)):
        save_ckpt(model,f"epoch{epoch}")

    # wait for other GPUs to catch up if needed
    if distributed: dist.barrier()
    
    # close any open plots
    plt.close()
    
if distributed: dist.destroy_process_group()