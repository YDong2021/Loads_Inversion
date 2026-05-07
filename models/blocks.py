"""Mid-level modules built out of models.layers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BasicBlock1D, DepthwiseSeparableConv1d, SelectiveSSM


# ---------------------------------------------------------------------------
# 1. ResNetStage — a sequence of BasicBlock1D with optional down-sampling head
# ---------------------------------------------------------------------------

class ResNetStage(nn.Module):
    """A sequence of ``num_blocks`` BasicBlock1D layers.  If ``stride_first``
    is 2, the first block changes (channels, length)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        kernel_size: int = 3,
        stride_first: int = 1,
        first_block_residual: bool = True,
    ) -> None:
        super().__init__()
        blocks = []
        # first block: may downsample + change width
        blocks.append(
            BasicBlock1D(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride_first,
                use_residual=first_block_residual,
            )
        )
        for _ in range(num_blocks - 1):
            blocks.append(
                BasicBlock1D(out_channels, out_channels,
                             kernel_size=kernel_size, stride=1,
                             use_residual=True)
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# ---------------------------------------------------------------------------
# 2. MambaBlock — mirrors the diagram in prompt.assets
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """One Mamba block.

    Input shape: (B, D, L).  The block projects to 2×E·D, splits into two
    branches, runs DWConv + SiLU + SSM on one branch, SiLU on the other, then
    element-wise multiplies (gating) and projects back to D.  A residual
    connection and a LayerNorm close the block.
    """

    def __init__(
        self,
        d_model: int = 96,
        d_state: int = 16,
        expand_factor: int = 4,
        conv_kernel: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert expand_factor % 2 == 0, "expand_factor must be even (path split)"
        d_inner = d_model * expand_factor // 2      # per branch feature dim
        self.d_model = d_model
        self.d_inner = d_inner

        # 4× projection (= two branches of expand_factor//2 · D each)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=True)
        self.conv    = DepthwiseSeparableConv1d(d_inner, kernel_size=conv_kernel)
        self.ssm     = SelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=True)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, L)  ->  (B, D, L)"""
        residual = x
        # channels-last for Linear
        x_l = x.transpose(1, 2)                        # (B, L, D)
        proj = self.in_proj(x_l)                       # (B, L, 2*d_inner)
        a, b = proj.chunk(2, dim=-1)                   # each (B, L, d_inner)

        # Path A: Conv1d -> SiLU -> SSM
        a = a.transpose(1, 2)                          # (B, d_inner, L)
        a = self.conv(a)
        a = F.silu(a)
        a = self.ssm(a)                                # (B, d_inner, L)
        a = a.transpose(1, 2)                          # (B, L, d_inner)

        # Path B: SiLU
        b = F.silu(b)

        # gated merge
        y = a * b                                      # (B, L, d_inner)
        y = self.out_proj(y)                           # (B, L, d_model)
        y = self.dropout(y)
        y = y.transpose(1, 2)                          # (B, D, L)

        y = y + residual
        # LayerNorm applied on the feature dim
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return y
