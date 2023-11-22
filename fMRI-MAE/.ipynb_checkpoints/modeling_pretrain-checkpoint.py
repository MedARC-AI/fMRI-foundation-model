import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import to_3tuple


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_fmrimae_small_patch13',
    'pretrain_fmrimae_base_patch13', 
    'pretrain_fmrimae_large_patch13', 
    'pretrain_fmrimae_huge_patch13',
]

def get_pos_embed(pos_embed, x, token_shape, max_shape=(4, 10, 10, 10)):
    embeding = []
    embed_dim = pos_embed.shape[-1]
    reshaped_pos_embed = pos_embed.reshape(*max_shape, -1)
    for shape in token_shape:
        start = [(L - l) // 2 for l, L in zip(shape, max_shape)]
        end = [s + l for s, l in zip(start, shape)]
        embed = reshaped_pos_embed[start[0]:end[0], start[1]:end[1], start[2]:end[2], start[3]:end[3]].reshape(-1, embed_dim)
        embed = torch.cat([
            embed, 
            torch.zeros((x.shape[1] - embed.shape[0], embed_dim), device=embed.device, dtype=embed.dtype)
        ])
        embeding.append(embed)
    return torch.stack(embeding)

# borrowed from huggingface
def invert_attention_mask(encoder_attention_mask: torch.Tensor, dtype=None) -> torch.Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    if dtype is not None:
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(encoder_extended_attention_mask.dtype).min

    return encoder_extended_attention_mask


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=(65, 78, 65), patch_size=13, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_frames=8, tubelet_size=2, use_checkpoint=False,
                 use_learnable_pos_emb=False, max_patch_shape=(4, 10, 10, 10)):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, num_frames=num_frames, tubelet_size=tubelet_size)
        self.use_checkpoint = use_checkpoint

        self.max_patch_shape = max_patch_shape
        num_patches = max_patch_shape[0] * max_patch_shape[1] * max_patch_shape[2] * max_patch_shape[3]
        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask, token_shape, attn_mask=None):
        B, _, C = x.shape
        x = self.patch_embed(x)
        
        x = x + get_pos_embed(self.pos_embed, x, token_shape, self.max_patch_shape).type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        attn_mask = attn_mask[~mask].reshape(B, -1) 
        attn_mask = invert_attention_mask(attn_mask, dtype=x.dtype)

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis, attn_mask)
        else:   
            for blk in self.blocks:
                x_vis = blk(x_vis, attn_mask)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask, token_shape, attn_mask):
        x = self.forward_features(x, mask, token_shape, attn_mask)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=13, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=600, tubelet_size=2, use_checkpoint=False
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = to_3tuple(patch_size)
        assert num_classes == tubelet_size * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_checkpoint = use_checkpoint


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, attn_mask, return_token_num):
        return_zero_token_mask = attn_mask
        attn_mask = invert_attention_mask(attn_mask, dtype=x.dtype)
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x, attn_mask)
        else:   
            for blk in self.blocks:
                x = blk(x, attn_mask)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
            return_zero_token_mask = return_zero_token_mask[:, -return_token_num:]
        else:
            x = self.head(self.norm(x))

        return x * return_zero_token_mask.unsqueeze(-1)

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=(65, 78, 65), 
                 patch_size=13, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 num_frames=8,
                 tubelet_size=2,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 max_patch_shape=(4, 10, 10, 10),
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb,
            max_patch_shape=max_patch_shape)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.max_patch_shape = max_patch_shape
        num_patches = max_patch_shape[0] * max_patch_shape[1] * max_patch_shape[2] * max_patch_shape[3]
        self.pos_embed = get_sinusoid_encoding_table(num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, token_shape, attn_mask=None):
        # B, N, C = x.shape
        x_vis = self.encoder(x, mask, token_shape, attn_mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = get_pos_embed(self.pos_embed, x, token_shape, self.max_patch_shape).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        attn_mask_vis = attn_mask[~mask].reshape(B, -1)
        attn_mask_mask = attn_mask[mask].reshape(B, -1)
        attn_mask_full = torch.cat([attn_mask_vis, attn_mask_mask], dim=1)
        x = self.decoder(x_full, attn_mask_full, pos_emd_mask.shape[1]) # [B, N_mask, 2 * 13 * 13 * 13]

        return x

@register_model
def pretrain_fmrimae_small_patch13(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=(65, 78, 65),
        patch_size=13,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=4394, 
        decoder_embed_dim=192, 
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_fmrimae_base_patch13(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=(65, 78, 65),
        patch_size=13, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=4394,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 
@register_model
def pretrain_fmrimae_large_patch13(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=(65, 78, 65),
        patch_size=13, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=4394, 
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_fmrimae_huge_patch13(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=(65, 78, 65),
        patch_size=13, 
        encoder_embed_dim=1280, 
        encoder_depth=32, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=4394, 
        decoder_embed_dim=640,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
