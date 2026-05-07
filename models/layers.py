"""Elementary layers shared across classifier and regressor."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. BasicBlock1D — two 1×3 conv + BN + ReLU + residual (ResNet basic style)
# ---------------------------------------------------------------------------

class BasicBlock1D(nn.Module):
    """ResNet "BasicBlock" adapted to 1-D.  Two conv1d layers with BN+ReLU and
    an identity / 1x1-projection residual path.  The first block of each stage
    can take ``stride=2`` to halve the time length while the channel dimension
    is doubled via a 1x1 conv on the shortcut path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.use_residual = use_residual
        if not use_residual:
            self.shortcut: Optional[nn.Module] = None
        elif stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        if self.use_residual:
            y = y + self.shortcut(x)
        return F.relu(y, inplace=True)


# ---------------------------------------------------------------------------
# 2. Depthwise-separable Conv1d (Mamba block's conv path)
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv1d(nn.Module):
    """DW conv (per-channel) + PW conv (1x1 mix).  Padding = kernel // 2 so the
    time length is preserved (even with odd kernels)."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size,
                            padding=kernel_size // 2, groups=channels, bias=True)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ---------------------------------------------------------------------------
# 3. Selective SSM (minimal PyTorch implementation of Mamba's S6 core)
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """Discrete Selective State-Space Model — simplified for clarity.

    Inputs x have shape (B, D, L) where D is the feature dim and L the time
    length.  For every element we produce data-dependent (``Δ``, ``B`` and
    ``C``) matrices, then run a sequential recurrence::

        h_{t+1} = exp(A · Δ_t) · h_t + Δ_t · B_t · x_t
        y_t     = C_t · h_t + D · x_t

    ``A`` is a learned negative diagonal matrix (per feature), which gives
    stable decay.  Parameters closely follow the *Mamba* paper.
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int | str = "auto") -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        if dt_rank == "auto":
            dt_rank = max(1, d_model // 16)
        self.dt_rank = int(dt_rank)

        # x -> (Δ, B, C)   (all data-dependent)
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # log A: (d_model, d_state) — negative semi-definite via -exp
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # skip D: (d_model,)
        self.D = nn.Parameter(torch.ones(d_model))

        # initialise dt_proj so softplus(dt) lands in a sensible range ~ [0.001, 0.1]
        nn.init.uniform_(self.dt_proj.weight, a=-0.05, b=0.05)
        with torch.no_grad():
            dt_bias = -torch.log(torch.rand(d_model) * (0.1 - 1e-3) + 1e-3)
            self.dt_proj.bias.copy_(dt_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, L)  ->  y: (B, D, L)"""
        B_, D, L = x.shape
        x_t = x.transpose(1, 2)                         # (B, L, D)

        # data-dependent Δ, B, C
        proj = self.x_proj(x_t)                         # (B, L, dt_rank + 2*N)
        dt, Bb, Cc = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))               # (B, L, D)

        A = -torch.exp(self.A_log.float())              # (D, N)

        # discretize: deltaA = exp(A · dt), deltaB = dt * B
        #   shapes: deltaA (B, L, D, N), deltaB_x (B, L, D, N)
        deltaA = torch.exp(dt.unsqueeze(-1) * A)                 # (B, L, D, N)
        deltaB_x = (dt.unsqueeze(-1) * Bb.unsqueeze(-2)) * x_t.unsqueeze(-1)
        # NOTE: unsqueeze(-2) puts B into (B,L,1,N) so it broadcasts over D

        # sequential scan (explicit loop — O(L); adequate for L ≈ 2500)
        h = x.new_zeros(B_, D, self.d_state)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_x[:, t]                # (B, D, N)
            y_t = (h * Cc[:, t].unsqueeze(1)).sum(dim=-1)        # (B, D)
            ys.append(y_t)
        y = torch.stack(ys, dim=-1)                              # (B, D, L)
        y = y + self.D.view(1, D, 1) * x
        return y
