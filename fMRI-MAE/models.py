import os
import sys
import json
import argparse
import numpy as np
import copy
import math
from einops import rearrange
from einops.layers.torch import Rearrange
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def posemb_sincos_4d(patches, temperature=10000, dtype=torch.float32):
    _, f, d, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x, t = torch.meshgrid(
        torch.arange(f, device=device),
        torch.arange(d, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij')

    fourier_dim = dim // 8

    omega = torch.arange(fourier_dim, device=device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z, y, x, t = [v.flatten()[:, None] * omega[None, :] for v in [z, y, x, t]]

    pe = torch.cat((z.sin(), z.cos(), y.sin(), y.cos(), x.sin(), x.cos(), t.sin(), t.cos()), dim=1)
    pe = F.pad(pe, (0, dim - (fourier_dim * 8)))
    return pe.type(dtype)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, dim, depth, heads, mlp_dim, num_encoder_patches, num_decoder_patches, channels, dim_head=64):
        super().__init__()
        image_depth, image_height, image_width = image_size
        patch_depth, patch_height, patch_width = image_patch_size

        # Check divisibility
        assert image_depth % patch_depth == 0 and image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by the frame patch size'

        num_patches = (image_depth // patch_depth) * (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_depth * patch_height * patch_width * frame_patch_size

        self.patchify = Rearrange('b c (f pf) (d pd) (h ph) (w pw) -> b f d h w (pd ph pw pf c)', pd=patch_depth, ph=patch_height, pw=patch_width, pf=frame_patch_size)
        
        self.unpatchify = nn.Sequential(
                                Rearrange('b (f d h w) (pd ph pw pf c) -> b f d h w (pd ph pw pf c)', 
                                    c=channels,d=image_depth, h=image_height, w=image_width,
                                    pd=patch_depth, ph=patch_height, pw=patch_width, pf=frame_patch_size)
                            )
        
        self.patch_to_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.encoder_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.decoder_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        
        self.posemb_sincos_4d = posemb_sincos_4d(torch.zeros(1, frames, 
                                    image_depth // patch_depth, image_height // patch_height, image_width // patch_width, dim))
        
        dec_seq_len = num_encoder_patches + num_decoder_patches
        self.encoder_to_decoder = nn.Linear(dim, mlp_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, num_decoder_patches, mlp_dim))
        self.decoder_proj = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim)
        )

    def forward(self, x, encoder_mask=None, decoder_mask=None, verbose=False):
        # ENCODER
        if decoder_mask is None:
            if verbose: print(x.shape)
            x = self.patch_to_emb(self.patchify(x))
            if verbose: print(x.shape)
            x = rearrange(x, 'b ... d -> b (...) d')
            if verbose: print(x.shape)
            if verbose: print("pe", self.posemb_sincos_4d.shape)
            x = x + self.posemb_sincos_4d.to(x.device)
            if verbose: print("x", x.shape)
            x = x[:,~encoder_mask]
            if verbose: print("masked",x.shape)
            x = self.encoder_transformer(x)
            if verbose: print(x.shape)
        else: # DECODER
            if verbose: print(x.shape)
            x = self.encoder_to_decoder(x)
            if verbose: print(x.shape)
            pos_embed = self.posemb_sincos_4d.to(x.device)
            if verbose: print("pe", pos_embed.shape)
            pos_emd_encoder = pos_embed[~encoder_mask]
            pos_emd_decoder = pos_embed[~decoder_mask]
            if verbose: print("pos_emd_encoder",pos_emd_encoder.shape)
            if verbose: print("pos_emd_decoder",pos_emd_decoder.shape)
            x = torch.cat([x + pos_emd_encoder, (self.mask_token + pos_emd_decoder).repeat(len(x),1,1)], dim=1)
            if verbose: print("x_concat", x.shape)
            x = self.decoder_transformer(x)
            if verbose: print(x.shape)
            x = self.decoder_proj(x)
            if verbose: print("proj", x.shape)
        return x 