from typing import Optional

import torch
import torch.nn as nn


class RotaryPositionalEmbeddings4D(nn.Module):
    def __init__(
        self,
        d: int,
        grid_height: int,
        grid_width: int,
        grid_depth: int,
        grid_time: int,
        base: int = 10_000,
    ):
        super().__init__()
        assert d % 4 == 0, f"{d} is not divisible by 4."
        self.base = base
        self.d = d
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_depth = grid_depth
        self.grid_time = grid_time
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache_1d(self, effective_d: int, seq_idx: torch.Tensor) -> torch.Tensor:
        seq_idx = seq_idx.reshape(-1)  # List of positions
        theta = 1.0 / (
            self.base ** (torch.arange(0, effective_d, 2).float() / effective_d)
        )

        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        return idx_theta2

    def _build_cache(self, x: torch.Tensor) -> None:
        # x: batched tensor
        # x.shape -> Batch, Seq Length, Embed Dim
        if self.cos_cached is not None and x.shape[1] <= self.cos_cached.shape[1]:
            # if cache is already built
            return
        # get the positions
        grid_h = torch.arange(self.grid_height, dtype=torch.float32)
        grid_w = torch.arange(self.grid_width, dtype=torch.float32)
        grid_d = torch.arange(self.grid_depth, dtype=torch.float32)
        grid_t = torch.arange(self.grid_time, dtype=torch.float32)

        grid = torch.meshgrid(
            grid_t, grid_d, grid_h, grid_w, indexing="xy"
        )  # This order should match with i/p
        grid = torch.stack(grid, axis=0)
        grid = grid.reshape(
            [4, 1, self.grid_time, self.grid_depth, self.grid_height, self.grid_width]
        )

        # Get the embedings
        emb_t = self._build_cache_1d(self.d // 4, grid[0])  # 1/4 in-case of 4D
        emb_d = self._build_cache_1d(self.d // 4, grid[1])
        emb_h = self._build_cache_1d(self.d // 4, grid[2])  # (T*D*H*W, embedding_dim/4)
        emb_w = self._build_cache_1d(self.d // 4, grid[3])
        emb = torch.concatenate(
            [emb_t, emb_d, emb_h, emb_w], axis=1
        )  # (T*D*H*W, embedding_dim)
        emb = emb.to(x.device)
        # cache sin and cos
        self.cos_cached = emb.cos()[
            None, None, :, :
        ]  # batch, Num Heads, Seq Len, Embed Dim
        self.sin_cached = emb.sin()[None, None, :, :]

    def _neg_half(self, x: torch.Tensor) -> torch.Tensor:
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Args:
        -----
        x: query or key vector
        mask: boolean vector of length sequence length.
        True for the non-masked positions.
        """
        self._build_cache(x)
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (
            (x_rope * self.cos_cached[:, :, : x.shape[1], :])
            + (neg_half_x * self.sin_cached[:, :, : x.shape[1], :])
            if mask is None
            else (
                x_rope * self.cos_cached[:, :, mask, :]
                + neg_half_x * self.sin_cached[:, :, mask, :]
            )
        )

        return torch.cat((x_rope, x_pass), dim=-1)


if __name__ == "__main__":
    # img size 4, 64 64 48
    # patch size 1, 8, 8, 8
    # num of patches across time, depth, height, and width -> 4, 8, 8, 6
    rot_embed = RotaryPositionalEmbeddings4D(
        d=512, grid_depth=8, grid_height=8, grid_width=6, grid_time=4
    )
    query_ = torch.randn(
        10, 1, 8 * 8 * 6 * 4, 512
    )  # Batch, Heads, Num Tokens/Seq Length, Embedding Dims
    print(rot_embed(query_).shape)
