# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# MAE-ST: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from util import video_vit
from util.logging import master_print as print
from util.hcp_flat import load_hcp_flat_mask
from util.misc import group_reduce


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_frames=16,
        t_patch_size=2,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=True,
        trunc_init=False,
        cls_embed=True,
        pred_t_dim=8,
        img_mask=None,
        **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim
        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames

        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

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
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                video_vit.Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                video_vit.Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_pred_patch_size * patch_size**2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_mask(img_mask)
        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_mask(self, img_mask):
        if img_mask is not None:
            img_mask = torch.as_tensor(img_mask > 0).float()

            H, W = img_mask.shape
            img_mask_patches = self.patchify(
                img_mask
                .view(1, 1, 1, H, W)
                .repeat(1, self.patch_embed.in_chans, self.pred_t_dim, 1, 1)
            )

            patch_mask = rearrange(
                img_mask,
                "(h ph) (w pw) -> (h w) (ph pw)",
                ph=self.patch_embed.patch_size[0],
                pw=self.patch_embed.patch_size[1],
            ).any(dim=1).float()
            patch_mask_indices, = patch_mask.nonzero(as_tuple=True)

            self.register_buffer("img_mask", img_mask)
            self.register_buffer("img_mask_patches", img_mask_patches)
            self.register_buffer("patch_mask", patch_mask)
            self.register_buffer("patch_mask_indices", patch_mask_indices)
            self.n_mask_patches = len(patch_mask_indices)
        else:
            self.register_buffer("img_mask", None)
            self.register_buffer("img_mask_patches", None)
            self.register_buffer("patch_mask", None)
            self.register_buffer("patch_mask_indices", None)
            self.n_mask_patches = None

    def patchify(self, imgs):
        """
        imgs: (N, C, T, H, W)
        x: (N, L, patch_size**2 *C)
        """
        N, C, T, H, W = imgs.shape
        ph, pw = self.patch_embed.patch_size
        u = self.t_pred_patch_size
        assert H % ph == 0 and W % pw == 0 and T % u == 0
        h = H // ph
        w = W // pw
        t = T // u

        x = imgs.reshape(shape=(N, C, t, u, h, ph, w, pw))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * ph * pw * C))
        self.patch_info = (N, C, T, H, W, ph, pw, u, t, h, w)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        N, C, T, H, W, ph, pw, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, ph, pw, C))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, C, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        T = self.patch_embed.t_grid_size
        H, W = self.patch_embed.grid_size
        assert L == T * H * W

        # adjust number to keep relative to image mask
        if self.img_mask is not None:
            len_keep = int(T * self.n_mask_patches * (1 - mask_ratio))
        else:
            len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # shift missing patches to not be selected
        if self.img_mask is not None:
            noise = noise.view(N, T, H * W)
            noise = noise + (1.0 - self.patch_mask)
            noise = noise.view(N, L)

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        x = x.view(N, -1, C)
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
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
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H, W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn

        # drop patches outside image mask
        if self.img_mask is not None:
            if self.cls_embed:
                decoder_cls_tokens, x = x[:, :1, :], x[:, 1:, :]
            x = x.view([N, T, H * W, C])
            x = x[:, :, self.patch_mask_indices]
            x = x.view([N, T * self.n_mask_patches, C])
            if self.cls_embed:
                x = torch.cat((decoder_cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]

        # fill outside mask with zeros
        if self.img_mask is not None:
            C = x.shape[-1]
            x = x.view([N, T, self.n_mask_patches, C])
            x_ = torch.zeros([N, T, H * W, C], dtype=x.dtype, device=x.device)
            x = x_.scatter(
                2, self.patch_mask_indices.view(1, 1, -1, 1).expand(N, T, -1, C), x,
            )
            x = x.view([N, T * H * W, C])

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, T, H, W]
        pred: [N, t*h*w, u*p*p*C]
        mask: [N, t*h*w], 0 is keep, 1 is remove,
        """
        # Nb, this way of selecting target frames picks the first frame for each
        # segment, *except* for the last one. This is from the official implementation
        # but might consider changing.
        _imgs = torch.index_select(
            imgs,
            2,
            torch.linspace(
                0,
                imgs.shape[2] - 1,
                self.pred_t_dim,
            )
            .long()
            .to(imgs.device),
        )
        target = self.patchify(_imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        if self.img_mask is not None:
            # exclude missing pixels from loss
            mask = mask.unsqueeze(-1) * self.img_mask_patches
        else:
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*C]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    @torch.no_grad()
    def masked_recon(self, imgs, pred, mask):
        # imgs: [N, C, T, H, W]
        # pred: [N, t*h*w, u*p*p*C]
        # mask: [N, t*h*w], 0 is keep, 1 is remove,
        target = torch.index_select(
            imgs,
            2,
            torch.linspace(
                0,
                imgs.shape[2] - 1,
                self.pred_t_dim,
            )
            .long()
            .to(imgs.device),
        )

        # this caches patch info for unpatchify
        # necessary if batch size is different from training
        self.patchify(target)

        target = torch.einsum("ncthw->nthwc", target)

        pred = self.unpatchify(pred.detach())
        pred = torch.einsum("ncthw->nthwc", pred)

        mask = mask.unsqueeze(-1).repeat(
            1, 1, self.patch_embed.patch_size[0]**2 * imgs.shape[1]
        )  # (N, T*H*W, p*p*c)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum("ncthw->nthwc", mask)

        # masked image
        im_masked = target * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = target * (1 - mask) + pred * mask
        return target, pred, mask, im_masked, im_paste

    @torch.no_grad()
    def denoise_recon(self, groups, imgs, pred, mask):
        # imgs: [N, C, T, H, W]
        N, _, T = imgs.shape[:3]
        assert groups.shape == (N, T), "invalid shape for groups"

        # target, pred, mask: [N, T / t, H, W, C]
        target, pred, mask, _, _ = self.masked_recon(imgs, pred, mask)
        pred = mask * pred  # 1 is removing, 0 is keeping

        groups = torch.index_select(
            groups,
            1,
            torch.linspace(
                0,
                T - 1,
                self.pred_t_dim,
            )
            .long()
            .to(imgs.device),
        )

        # flatten N and T
        target = target.flatten(0, 1)
        pred = pred.flatten(0, 1)
        mask = mask.flatten(0, 1)
        groups = groups.flatten(0, 1)

        group_ids, pred = group_reduce(pred, groups)
        _, counts = group_reduce(mask, groups)
        _, target = group_reduce(target, groups, reduction="mean")

        mask = (counts > 0).float()
        pred = mask * (pred / counts.clamp(min=1.0))
        return group_ids, target, pred, counts


def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_small_patch16_fmri(**kwargs):
    model = MaskedAutoencoderViT(
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


def mae_vit_base_patch16_fmri(**kwargs):
    model = MaskedAutoencoderViT(
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


def mae_vit_large_patch16_fmri(**kwargs):
    model = MaskedAutoencoderViT(
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


def mae_vit_huge_patch16_fmri(**kwargs):
    model = MaskedAutoencoderViT(
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
