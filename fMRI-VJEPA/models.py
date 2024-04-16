from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from rope import RotaryPositionalEmbeddings4D

def my_split_by_node(urls): return urls


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


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        use_rope: bool = False,
        cls_token: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head**-0.5
        self.use_rope = use_rope
        self.cls_token = cls_token
        self.norm = nn.LayerNorm(embed_dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Optional[nn.Module],
        mask: Optional[torch.Tensor] = None,
        ridge: Optional[torch.Tensor] = None,
        nn_emb: Optional[torch.Tensor] = None,
        nn_patch: Optional[torch.Tensor] = None,
    ):
        if ridge is not None:
            for ir,r in enumerate(ridge):
                x[ir] = r(x[ir])
        
        x = self.norm(x)

        if nn_emb is not None:
            x *= nn_emb

        if nn_patch is not None:
            x += nn_patch

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)
        if self.use_rope:
            if pos_embed is None:
                raise ValueError(
                    "For RoPE embeddings `pos_embed` should be \
                    passed to the Attention forward."
                )
            # apply RoPE other than CLS token if it's included.
            if self.cls_token:
                q_cls = q[:, :, :1, :]
                k_cls = k[:, :, :1, :]
                q = q[:, :, 1:, :]
                k = k[:, :, 1:, :]
            q = pos_embed(q, mask=mask)
            k = pos_embed(k, mask=mask)
            if self.cls_token:
                q = torch.cat([q_cls, q], dim=2)
                k = torch.cat([k_cls, k], dim=2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # print("q", q.shape) # B num_heads N D
        # print("k_T", k.transpose(-1, -2).shape)
        # print("dots", dots.shape) # B num_heads N N
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        dim_head: int,
        mlp_dim: int,
        patch_dim: int,
        use_rope: bool = False,
        grid_height: Optional[int] = None,
        grid_width: Optional[int] = None,
        grid_depth: Optional[int] = None,
        grid_time: Optional[int] = None,
        cls_token: bool = False,
        
        **args,
    ):
        super().__init__()

        self.patch_to_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        ) 

        
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            embed_dim,
                            num_heads=num_heads,
                            dim_head=dim_head,
                            use_rope=use_rope,
                            cls_token=cls_token,
                        ),
                        FeedForward(embed_dim, mlp_dim),
                    ]
                )
            )
        # RoPE positional embeddings
        self.use_rope = use_rope
        if self.use_rope:
            self.rope_pos_emb = RotaryPositionalEmbeddings4D(
                d=dim_head,
                grid_depth=grid_depth,
                grid_height=grid_height,
                grid_width=grid_width,
                grid_time=grid_time,
            )

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        for attn, ff in self.layers:
            x = (
                attn(
                    x, 
                    pos_embed=self.rope_pos_emb if self.use_rope else None,
                    mask=mask,
                )
                + x
            )
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(
        self,
        *,
        x_encoder,
        y_encoder,
        predictor,
        image_size,
        image_patch_size,
        num_frames,
        frame_patch_size,
        channels=1,
        use_rope_emb: bool = False,
        use_cls_token: bool = False,
    ):
        super().__init__()
        image_depth, image_height, image_width = image_size
        patch_depth, patch_height, patch_width = image_patch_size

        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.predictor = predictor
        
        # Check divisibility
        assert (image_depth % patch_depth == 0 and image_height % patch_height == 0 and image_width % 
                patch_width == 0), "Image dimensions must be divisible by the patch size."
        assert (num_frames % frame_patch_size == 0), "num_frames must be divisible by the frame patch size"

        self.patch_dim = channels * patch_depth * patch_height * patch_width * frame_patch_size

        self.num_patches = image_size[0]//image_patch_size[0] * image_size[1]//image_patch_size[1] * image_size[2]//image_patch_size[2] * num_frames
        
        self.patchify = Rearrange(
            "b c (f pf) (d pd) (h ph) (w pw) -> b f d h w (pd ph pw pf c)",
            pd=patch_depth,
            ph=patch_height,
            pw=patch_width,
            pf=frame_patch_size,
        )

        self.unpatchify = nn.Sequential(
            Rearrange(
                "b (f d h w) (pd ph pw pf c) -> b f d h w (pd ph pw pf c)",
                c=channels,
                d=image_depth,
                h=image_height,
                w=image_width,
                pd=patch_depth,
                ph=patch_height,
                pw=patch_width,
                pf=frame_patch_size,
            )
        )        

        self.use_rope_emb = use_rope_emb

        if not self.use_rope_emb:
            self.posemb_sincos_4d = posemb_sincos_4d(
                torch.zeros(
                    1,
                    num_frames // frame_patch_size,
                    image_depth // patch_depth,
                    image_height // patch_height,
                    image_width // patch_width,
                    self.x_encoder.embed_dim,
                )
            )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.x_encoder.embed_dim))
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.x_encoder.embed_dim))

    def forward(self, x, encoder_mask=None, encoder_type=None, device="cuda", verbose=False):
        if encoder_type == "x": # X ENCODER
            if verbose: print("input shape", x.shape)
            x = self.patchify(x)
            if verbose: print("after patching", x.shape)
            x = self.x_encoder.patch_to_emb(x.to(device))
            if verbose: print("convert to embedding", x.shape)
            x = rearrange(x, "b ... d -> b (...) d")
            if verbose: print("flattening", x.shape)
            if not self.use_rope_emb:
                if verbose: print("positional embedding", self.posemb_sincos_4d.shape)
                x = x + self.posemb_sincos_4d.to(x.device)
            if verbose: print("current shape", x.shape)
            x = x[:, encoder_mask]
            if self.use_cls_token:
                cls_tokens = self.cls_token.expand(len(x), -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            if verbose: print("input to x_encoder:", x.shape)
            x = self.x_encoder(x, mask=encoder_mask if self.use_rope_emb else None)
            if verbose: print("final shape", x.shape)
            
        elif encoder_type == "y": # Y ENCODER
            with torch.no_grad():
                if verbose: print("input shape", x.shape)
                x = self.patchify(x)
                if verbose: print("after patching", x.shape)
                x = self.y_encoder.patch_to_emb(x.to(device))
                if verbose: print("convert to embedding", x.shape)
                x = rearrange(x, "b ... d -> b (...) d")
                if verbose: print("flattening", x.shape)
                if not self.use_rope_emb:
                    if verbose: print("positional embedding", self.posemb_sincos_4d.shape)
                    x = x + self.posemb_sincos_4d.to(x.device)
                if verbose: print("input to y_encoder:", x.shape)
                x = self.y_encoder(x, mask=encoder_mask if self.use_rope_emb else None)
                if verbose: print("after encoder", x.shape)
                # apply layer_norm to output of yencoder to ensure no drifting
                x = torch.nn.functional.layer_norm(x, (x.size(-1),))  # normalize over feature-dim  [B, N, D]
                if encoder_mask is not None:
                    x = x[:, ~encoder_mask]
                    if verbose: print("only use masked tokens", x.shape)
                if verbose: print("final shape", x.shape)
            
        else:  # PREDICTOR
            if verbose: print("input shape", x.shape)
            B, X, _ = x.shape
            masked_tokens = ~encoder_mask
            N = masked_tokens.sum()
            mask = None
            if not self.use_rope_emb:
                pos_embed = self.posemb_sincos_4d.to(x.device)
                if verbose: print("positional embedding", pos_embed.shape)
                pos_emd_mask = pos_embed[masked_tokens]
                if verbose: print("pos_emd_mask", pos_emd_mask.shape)
                if self.use_cls_token:
                    cls_tokens = x[:,:1,:]
                    x = x[:,1:,:]
                x = torch.cat([x, 
                        self.mask_token.repeat(B, N, 1) + pos_emd_mask], dim=1)
                if self.use_cls_token:
                    x = torch.cat([cls_tokens, x], dim=1)
            else:
                mask = torch.cat((torch.where(encoder_mask)[0], torch.where(decoder_mask)[0]))
                # No abs positional embeddings for RoPE
                x = torch.cat([x,self.mask_token.repeat(B, N - 1 if self.use_cls_token else N, 1)],dim=1)
            if verbose: print("concatenation", x.shape)
            if verbose: print("input to predictor:", x.shape)
            x = self.predictor(x, mask=mask)
            if verbose: print("only use masked tokens", x.shape)
            x = x[:, X:X+N, :]
            if verbose: print("final shape", x.shape)
        return x