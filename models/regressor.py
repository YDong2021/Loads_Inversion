"""Mamba-based load-shape regressor.

End-to-end pipeline (see prompt.assets/image-20260507100723481.png):

    response (B, 3, L)
        |                                 pos_id (B,)
        v                                    |
    Input projection: conv1d 3 -> 96         |
                         |                   v
                         v               Fourier PE
                         +---------⊕----------+
                                   |
                      8 × (MambaBlock, default)
                                   |
                 Output projection: conv1d 96 -> 48 -> 1
                                   |
                     predicted load (B, L)

Three variants differ only in the number of Mamba blocks (6 / 8 / 10).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .blocks import MambaBlock
from .positional_encoding import FourierPositionalEncoding


# ===========================================================================
# Base regressor
# ===========================================================================

class MambaRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 96,
        n_positions: int = 50,
        pe_base: float = 1000.0,
        num_blocks: int = 8,
        ssm_state_dim: int = 16,
        conv_kernel: int = 7,
        expand_factor: int = 4,
        dropout: float = 0.1,
        output_hidden: int = 48,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # 1) input projection (conv1d 3 -> D)
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=True)

        # 2) Fourier positional encoding on pos_id in [0, n_positions)
        self.pos_enc = FourierPositionalEncoding(
            num_positions=n_positions,
            embed_dim=hidden_dim,
            base=pe_base,
        )

        # 3) stacked Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=hidden_dim,
                d_state=ssm_state_dim,
                expand_factor=expand_factor,
                conv_kernel=conv_kernel,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # 4) two-stage 1x1 conv output projection D -> D/2 -> 1
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, output_hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(output_hidden, 1, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, response: torch.Tensor, pos_id: torch.Tensor) -> torch.Tensor:
        """
        response : (B, 3, L)   already globally normalized
        pos_id   : (B,) long in [0, n_positions)

        Returns the predicted load curve ``(B, L)``.
        """
        x = self.input_proj(response)              # (B, D, L)
        pe = self.pos_enc(pos_id)                  # (B, D, 1) — broadcast over L
        x = x + pe
        for block in self.blocks:
            x = block(x)                            # (B, D, L)
        y = self.output_proj(x)                     # (B, 1, L)
        return y.squeeze(1)                         # (B, L)


# ===========================================================================
# Variants — different number of Mamba blocks
# ===========================================================================

class MambaRegressor6(MambaRegressor):
    def __init__(self, **kw):
        kw["num_blocks"] = 6
        super().__init__(**kw)


class MambaRegressor8(MambaRegressor):
    def __init__(self, **kw):
        kw["num_blocks"] = 8
        super().__init__(**kw)


class MambaRegressor10(MambaRegressor):
    def __init__(self, **kw):
        kw["num_blocks"] = 10
        super().__init__(**kw)


_REGRESSOR_REGISTRY = {
    "mamba_6":  MambaRegressor6,
    "mamba_8":  MambaRegressor8,
    "mamba_10": MambaRegressor10,
}


def build_regressor(model_cfg: Dict) -> MambaRegressor:
    name = model_cfg["name"]
    if name not in _REGRESSOR_REGISTRY:
        raise KeyError(f"unknown regressor variant: {name}. "
                       f"Available: {list(_REGRESSOR_REGISTRY)}")
    kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    # "num_blocks" inside kwargs is ignored since variants pin it
    kwargs.pop("num_blocks", None)
    return _REGRESSOR_REGISTRY[name](**kwargs)
