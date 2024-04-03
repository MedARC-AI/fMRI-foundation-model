# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def posemb_sincos_4d(patches, temperature=10000, dtype=torch.float32):
    _, f, d, h, w, dim, device, dtype = (*patches.shape, patches.device, patches.dtype)

    z, y, x, t = torch.meshgrid(
        torch.arange(f, device=device),
        torch.arange(d, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )

    fourier_dim = dim // 8

    omega = torch.arange(fourier_dim, device=device) / (fourier_dim - 1)
    omega = 1.0 / (temperature**omega)

    z, y, x, t = [v.flatten()[:, None] * omega[None, :] for v in [z, y, x, t]]

    pe = torch.cat(
        (z.sin(), z.cos(), y.sin(), y.cos(), x.sin(), x.cos(), t.sin(), t.cos()), dim=1
    )
    pe = F.pad(pe, (0, dim - (fourier_dim * 8)))
    return pe.type(dtype)



class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    import inspect 
    inspect.getsourcefile(Mamba)
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        # print module name 
        
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x

 
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0) 


class VisionMamba(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            frame_patch_size=4,
            encoder_depth=24, 
            decoder_depth=24,
            embed_dim=192, 
            channels=3, 
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            # video
            kernel_size=1, 
            num_frames=8, 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')
       

        # pretrain parameters
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        patch_depth, patch_height, patch_width = patch_size
        self.patchify = Rearrange(
            "b c (f pf) (d pd) (h ph) (w pw) -> b f d h w (pd ph pw pf c)",
            pd=patch_depth,
            ph=patch_height,
            pw=patch_width,
            pf=frame_patch_size,
        )
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, 
        #     kernel_size=kernel_size,
        #     in_chans=channels, embed_dim=embed_dim
        # )
        self.patch_dim = channels * patch_depth * patch_height * patch_width * frame_patch_size
        self.patch_to_emb = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        image_depth, image_height, image_width = img_size
        self.posemb_sincos_4d = posemb_sincos_4d(
                torch.zeros(
                    1,
                    num_frames,
                    image_depth // patch_depth,
                    image_height // patch_height,
                    image_width // patch_width,
                    self.embed_dim,
                )
            )
        print ("posemb_sincos_4d", self.posemb_sincos_4d.shape)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        # self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        encoder_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]
        # inter_dpr = [0.0] + dpr
        encoder_inter_dpr = [0.0] + encoder_dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        print (embed_dim, "rms_norm", rms_norm, "residual_in_fp32", residual_in_fp32, "fused_add_norm", fused_add_norm, "bimamba", bimamba, "ssm_cfg", ssm_cfg)
        self.encoder_layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=encoder_inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(encoder_depth)
            ]
        )
        decoder_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
        decoder_inter_dpr = [0.0] + decoder_dpr
        self.decoder_layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=decoder_inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(decoder_depth)
            ]
        ) 
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
  

        # original init
        self.apply(segm_init_weights)
        # trunc_normal_(self.pos_embed, std=.02)

        # mamba init 
        self.apply(
            partial(
                _init_weights,
                n_layer=encoder_depth + decoder_depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.encoder_layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {"pos_embed", "cls_token", "temporal_pos_embedding"}
        return {"cls_token", "mask_token", "posemb_sincos_4d"}
    
    def get_num_layers(self):
        return len(self.encoder_layers) + len(self.decoder_layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params, encoder_mask=None, decoder_mask=None, verbose=False):
        if decoder_mask is None:
            # print("inp", x.shape)
            x = self.patchify(x)
            # print("patchify", x.shape)
            x = self.patch_to_emb(x)
            # print("patched_emb", x.shape)
            B, T, D, H, W, C = x.shape
            # x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
            x = rearrange(x, "b ... d -> b (...) d")
            # print ("rearrange", x.shape)
            # position embeddings
            x = x[:, encoder_mask] + self.posemb_sincos_4d[encoder_mask].to(x.device)
            x = self.pos_drop(x)
            # print ("after encoder embeddings", x.shape)
            
            # print("before mamba", x.shape) 
            # mamba impl
            residual = None
            hidden_states = x
            for idx, layer in enumerate(self.encoder_layers):
                if self.use_checkpoint and idx < self.checkpoint_num:
                    hidden_states, residual = layer(
                        hidden_states, residual, inference_params=inference_params,
                        use_checkpoint=True
                    )
                else:
                    hidden_states, residual = layer(
                        hidden_states, residual, inference_params=inference_params
                    )
    
            if not self.fused_add_norm:
                if residual is None:
                    residual = hidden_states
                else:
                    residual = residual + self.drop_path(hidden_states)
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
            # print ("hidden_states", hidden_states.shape)
            return hidden_states 
        else:  # DECODER
            if verbose: print("decoder input", x.shape)
            B, _, _ = x.shape
            pos_emd_encoder = self.posemb_sincos_4d[encoder_mask].to(x.device)
            pos_emd_decoder = self.posemb_sincos_4d[decoder_mask].to(x.device)
            # print("pos_emd_encoder", pos_emd_encoder.shape, "pos_emd_decoder", pos_emd_decoder.shape)
            x = torch.cat([x + pos_emd_encoder, 
                                self.mask_token.repeat(B, pos_emd_decoder.size(0), 1) + pos_emd_decoder], 
                                dim=1) 

            x = self.pos_drop(x)
            # print("decoder input after pos", x.shape)
            residual = None
            hidden_states = x
            for idx, layer in enumerate(self.decoder_layers):
                if self.use_checkpoint and idx < self.checkpoint_num:
                    hidden_states, residual = layer(
                        hidden_states, residual, inference_params=inference_params,
                        use_checkpoint=True
                    )
                else:
                    hidden_states, residual = layer(
                        hidden_states, residual, inference_params=inference_params
                    )

            if not self.fused_add_norm:
                if residual is None:
                    residual = hidden_states
                else:
                    residual = residual + self.drop_path(hidden_states)
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
            return hidden_states

    def forward(self, x, inference_params=None, encoder_mask=None, decoder_mask=None, verbose=False):
        x = self.forward_features(x, inference_params, encoder_mask, decoder_mask, verbose)
        # x = self.head(self.head_drop(x))
        return x


def videomamba_middle_pretrain(pretrained=False, **kwargs):
    model = VisionMamba(
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

    
def get_mamba(size, **kwargs):
    if size == 'middle':
        return videomamba_middle_pretrain(**kwargs) 
 
if __name__ == '__main__':
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8
    img_size = 224
    
    model = videomamba_middle_pretrain(num_frames=num_frames).cuda()
    mask = torch.cat([
        torch.ones(1, 8 * int(14 * 14 * 0.75)),
        torch.zeros(1, 8 * int(14 * 14 * 0.25)),
    ], dim=-1).to(torch.bool)
    print(model(torch.rand(1, 3, num_frames, img_size, img_size), mask)[1].shape)
