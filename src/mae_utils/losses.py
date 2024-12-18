import os
import random
import numpy as np
import torch
from einops import rearrange
import re
import torch.nn.functional as F
import torch.nn as nn

def get_ids_shuffle(batch_size, device, model, mask_ratio=0.75):
    N = batch_size
    T = model.patch_embed.t_grid_size
    H, W = model.patch_embed.grid_size
    L = T * H * W

    noise = torch.rand(N, L, device=device)  # noise in [0, 1]

    # shift missing patches to not be selected
    if model.img_mask is not None:
        noise = noise.view(N, T, H * W)
        noise = noise + (1.0 - model.patch_mask)
        noise = noise.view(N, L)

    # sort noise for each sample
    ids_shuffle = torch.argsort(
        noise, dim=1
    )  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    return ids_shuffle, ids_restore

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
    def vicreg_loss(l1, l2, gamma=1.0, lamda=25, mu=25, nu=1, rand_frac=0.2, use_vic_cls=True, eps=1e-4):
        if use_vic_cls:
            # always keep cls and pick a random set of tokens
            rand_indices = torch.cat([torch.tensor([0]), 1+torch.randperm(l1.shape[1]-1)])[:int(rand_frac*l1.shape[1])]
        else:
            # drop cls tokens from loss calc
            l1 = l1[:, 1:]
            l2 = l2[:, 1:]
            rand_indices = torch.randperm(l1.shape[1])[:int(rand_frac*l1.shape[1])]

        std_l1 = torch.sqrt(l1.flatten(1).var(dim=0)+eps)  # nxd
        std_l2 = torch.sqrt(l2.flatten(1).var(dim=0)+eps)  # nxd
        var_loss = F.relu(gamma - std_l1).mean() + F.relu(gamma - std_l2).mean()
        del std_l1, std_l2

        sim_loss = F.mse_loss(l1, l2)

        l1 = l1 - l1.mean(0, keepdim=True)  # b,n,d
        l2 = l2 - l2.mean(0, keepdim=True)                
        
        l1_sub = l1[:, rand_indices]
        del l1
        cov_l1 = torch.bmm(l1_sub.permute(1,2,0), l1_sub.permute(1,0,2))/(l1_sub.shape[0]-1)  # 0.1*n,d,d
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
    def simclr_loss(lats, temp=0.006, distributed=False):
        if not distributed:
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
        else:
            rank = int(os.getenv('LOCAL_RANK'))
            world_size = int(os.getenv('WORLD_SIZE'))

            all_lats = torch.cat(torch.distributed.nn.all_gather(lats), dim=0)
            logits = (nn.functional.normalize(lats.flatten(1),dim=-1) @
                        nn.functional.normalize(all_lats.flatten(1),dim=-1).T) / temp

            # Calculate offsets for positive pairs based on rank
            batch_size = lats.shape[0]
            pos_offset = rank * batch_size

            # common label structure
            labels = torch.diag_embed(
                torch.ones(batch_size // 2), offset=batch_size // 2
            ) + torch.diag_embed(torch.ones(batch_size // 2), offset=-batch_size // 2)
            
            # assign labels to correct position
            all_labels = torch.zeros_like(logits)
            all_labels[:, pos_offset:pos_offset+batch_size] = labels.to(lats.device)

            mask = torch.ones((batch_size, batch_size), dtype=bool).bool()
            torch.diagonal(mask).fill_(False)
            all_mask = torch.ones_like(logits).bool()
            all_mask[:, pos_offset:pos_offset+batch_size] = mask

            all_labels = all_labels[all_mask].reshape(batch_size, world_size*batch_size-1)
            logits = logits[all_mask].reshape(*all_labels.shape)

            contr_loss = -(logits.log_softmax(-1) * all_labels).sum(-1).mean()

            return contr_loss