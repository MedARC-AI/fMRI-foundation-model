from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MindEyeModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
class RidgeRegression(nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_sizes, out_features, seq_len=1): 
        super().__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
        self.seq_len = seq_len
    def forward(self, x, subj_idx):
        out = torch.cat([self.linears[subj_idx](x[:,seq]).unsqueeze(1) for seq in range(self.seq_len)], dim=1)
        return out
class BrainNetwork(nn.Module):
    def __init__(self, h=1024, in_dim=15724, out_dim=768, seq_len=1, n_blocks=4, drop=.15, clip_size=768, clip_scale=1):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.clip_scale = clip_scale
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)
            
    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
    
    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )
        
    def forward(self, x):
        # make empty tensors
        c = torch.Tensor([0.])
        
        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0,2,1)
        for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0,2,1)
            
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0,2,1)
            
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        if self.clip_scale>0:
            c = self.clip_proj(backbone)
        
        return backbone, c
    
    
import open_clip
import kornia
class FrozenOpenCLIPImageEmbedder(nn.Module):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        init_device="cpu",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        l2_norm_tokens=False,
        only_tokens=False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device(init_device),
            pretrained=version,
            cache_dir=cache_dir,
        )
        del model.transformer
        self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.antialias = antialias

        self.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )
        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens
        if only_tokens:
            assert output_tokens
        self.only_tokens = only_tokens
        self.l2_norm_tokens = l2_norm_tokens
        if l2_norm_tokens:
            assert output_tokens

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        tokens = None
        if self.output_tokens:
            z, tokens = z[0], z[1]
        z = z.to(image.dtype)
        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            z = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device)
                )[:, None]
                * z
            )
            if tokens is not None:
                tokens = (
                    expand_dims_like(
                        torch.bernoulli(
                            (1.0 - self.ucg_rate)
                            * torch.ones(tokens.shape[0], device=tokens.device)
                        ),
                        tokens,
                    )
                    * tokens
                )
        if self.unsqueeze_dim:
            z = z[:, None, :]
        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            if self.only_tokens:
                return tokens
            return tokens, z
        if self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None, :]
            else:
                z_ = z
            return repeat(z_, "b 1 d -> b n d", n=self.max_length), z
        elif self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat(
                (
                    z,
                    torch.zeros(
                        z.shape[0],
                        self.max_length - z.shape[1],
                        z.shape[2],
                        device=z.device,
                    ),
                ),
                1,
            )
            return z_pad, z_pad[:, 0, ...]
        return z

    def encode_with_vision_transformer(self, img):
        # if self.max_crops > 0:
        #    img = self.preprocess_by_cropping(img)
        if img.dim() == 5:
            assert self.max_crops == img.shape[1]
            img = rearrange(img, "b n c h w -> (b n) c h w")
        img = self.preprocess(img)
        if not self.output_tokens:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        else:
            assert self.model.visual.output_tokens
            x, tokens = self.model.visual(img)
            if self.l2_norm_tokens:
                token_shape = tokens.shape
                tokens = tokens.flatten(1)
                tokens = torch.nn.functional.normalize(tokens, dim=-1)
                tokens = (tokens - .0002) / .0015
                tokens = tokens.view(token_shape)
                tokens = (tokens * 1.0957) + .1598
        if self.max_crops > 0:
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)
            # drop out between 0 and all along the sequence axis
            x = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate)
                    * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
                )
                * x
            )
            if tokens is not None:
                tokens = rearrange(tokens, "(b n) t d -> b t (n d)", n=self.max_crops)
                logpy.warning(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message."
                )
        if self.output_tokens:
            return x, tokens
        return x

    def encode(self, text):
        return self(text)