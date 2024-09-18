# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import trunc_normal_
from util.logging import master_print as print
from util.hcp_flat import load_hcp_flat_mask

from util.video_vit import Block, PatchEmbed

ROOT = Path(__file__).parent


class VisionTransformer(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=400,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_frames=16,
        t_patch_size=2,
        no_qkv_bias=False,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        dropout=0.5,
        sep_pos_embed=True,
        cls_embed=True,
        img_mask=None,
        global_pool: Literal["avg", "cls", "spatial"] = "avg",
        **kwargs,
    ):
        super().__init__()

        self.sep_pos_embed = sep_pos_embed
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size
        self.cls_embed = cls_embed
        self.global_pool = global_pool

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim), requires_grad=True
            )  # fixed or not?

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.initialize_mask(img_mask)

        self.dropout = nn.Dropout(dropout)
        if global_pool == "spatial":
            self.spatial_pool = SpatialPool(self.n_mask_patches, num_classes)
        self.head = nn.Linear(embed_dim, num_classes)

        torch.nn.init.normal_(self.head.weight, std=0.02)

    def initialize_mask(self, img_mask):
        if img_mask is not None:
            img_mask = torch.as_tensor(img_mask > 0).float()

            patch_mask = rearrange(
                img_mask,
                "(h ph) (w pw) -> (h w) (ph pw)",
                ph=self.patch_embed.patch_size[0],
                pw=self.patch_embed.patch_size[1],
            ).any(dim=1).float()
            patch_mask_indices, = patch_mask.nonzero(as_tuple=True)

            self.register_buffer("img_mask", img_mask)
            self.register_buffer("patch_mask", patch_mask)
            self.register_buffer("patch_mask_indices", patch_mask_indices)
            self.n_mask_patches = len(patch_mask_indices)
        else:
            self.register_buffer("img_mask", None)
            self.register_buffer("patch_mask", None)
            self.register_buffer("patch_mask_indices", None)
            self.n_mask_patches = None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
            "pos_embed_spatial",
            "pos_embed_temporal",
            "pos_embed_class",
        }

    def forward_features(self, x, global_pool=False):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape  # T: temporal; L: spatial

        x = x.view([N, T * L, C])

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            pos_embed = self.pos_embed[:, :, :]
        x = x + pos_embed

        # drop patches outside image mask
        if self.img_mask is not None:
            if self.cls_embed:
                cls_tokens, x = x[:, :1, :], x[:, 1:, :]
            x = x.view([N, T, L, C])
            x = x[:, :, self.patch_mask_indices]
            x = x.view([N, T * self.n_mask_patches, C])
            if self.cls_embed:
                x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if global_pool:
            if self.cls_embed:
                x = x[:, 1:, :]
            x = x.mean(dim=1)
        return x

    def forward_head(self, x):
        # classifier
        if self.global_pool == "spatial":
            if self.cls_embed:
                x = x[:, 1:, :]
            N, _, C = x.shape
            T = self.patch_embed.t_grid_size
            L = self.n_mask_patches
            x = x.view(N, T, L, C).mean(dim=1)  # temporal mean
            x = self.spatial_pool(x)  # (batch, classes, dim)
        elif self.global_pool == "cls":
            assert self.cls_embed
            x = x[:, 0, :]
        else:
            if self.cls_embed:
                x = x[:, 1:, :]
            x = x.mean(dim=1)

        x = self.norm(x)
        x = self.dropout(x)

        if self.global_pool == "spatial":
            x = (self.head.weight * x).sum(dim=2) + self.head.bias
        else:
            x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def mask_fill(self, x):
        N, L, C = x.shape
        T = self.patch_embed.t_grid_size
        H, W = self.patch_embed.grid_size
        assert L == T * self.n_mask_patches

        x = x.view(N, T, -1, C)
        x_ = torch.zeros([N, T, H * W, C], dtype=x.dtype, device=x.device)
        x = x_.scatter(
            2, self.patch_mask_indices.view(1, 1, -1, 1).expand(N, T, -1, C), x,
        )
        return x


class SpatialPool(nn.Module):
    """
    Pool a sequence of features with a learned attention weight per class.

    Args:
        seq_len: Length of the sequence, N.
        num_classes: Number of classes, K.
        drop: Dropout probability.

    Shape:
        - Input: (B, N, C)
        - Output: (B, K, C)
    """

    def __init__(self, seq_len: int, num_classes: int):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.empty(num_classes, seq_len))
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=self.seq_len**-0.5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.weight, dim=1)
        output = attn @ input
        return output

    def extra_repr(self) -> str:
        return f"{self.seq_len}, {self.num_classes}"


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_small_patch16_fmri(**kwargs):
    model = VisionTransformer(
        img_size=(144, 320),
        patch_size=16,
        in_chans=1,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_mask=load_hcp_flat_mask(),
        **kwargs,
    )
    return model


def vit_base_patch16_fmri(**kwargs):
    model = VisionTransformer(
        img_size=(144, 320),
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_mask=load_hcp_flat_mask(),
        **kwargs,
    )
    return model


def vit_large_patch16_fmri(**kwargs):
    model = VisionTransformer(
        img_size=(144, 320),
        patch_size=16,
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_mask=load_hcp_flat_mask(),
        **kwargs,
    )
    return model


def vit_huge_patch16_fmri(**kwargs):
    model = VisionTransformer(
        img_size=(144, 320),
        patch_size=16,
        in_chans=1,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_mask=load_hcp_flat_mask(),
        **kwargs,
    )
    return model


def load_hcp_flat_mask() -> torch.Tensor:
    mask = np.load(ROOT / "hcp-flat_mask.npy")
    mask = torch.as_tensor(mask)
    return mask
