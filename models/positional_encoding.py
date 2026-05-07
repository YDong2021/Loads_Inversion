"""Fourier Positional Encoding for discrete position IDs.

As specified in prompt.md::

    e_{pos, 2i}     = sin(pos / base ** (2i / D))
    e_{pos, 2i+1}   = cos(pos / base ** (2i / D))

where D = num_positions - 1 and ``base`` defaults to 1000.  The resulting
embedding is broadcast over the time dimension and added to the response
features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    def __init__(
        self,
        num_positions: int = 50,
        embed_dim: int = 96,
        base: float = 1000.0,
    ) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even (sin/cos pairs)"
        self.num_positions = num_positions
        self.embed_dim = embed_dim
        self.base = float(base)

        D = max(num_positions - 1, 1)
        # exponents 2i / D for i = 0 .. embed_dim/2 - 1
        i = torch.arange(embed_dim // 2, dtype=torch.float32)
        exponents = (2.0 * i) / float(D)
        div_term = torch.pow(torch.tensor(self.base), exponents)   # (embed_dim/2,)

        pos = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)  # (P, 1)
        angles = pos / div_term.unsqueeze(0)                                 # (P, embed_dim/2)

        pe = torch.zeros(num_positions, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        # register as non-trainable buffer; use .float() to ensure dtype match
        self.register_buffer("pe_table", pe, persistent=False)

    def forward(self, pos_id: torch.Tensor) -> torch.Tensor:
        """pos_id: (B,) long tensor in [0, num_positions).
        Returns: (B, embed_dim, 1) — ready to be added to (B, D, L) features."""
        emb = self.pe_table[pos_id]                 # (B, embed_dim)
        return emb.unsqueeze(-1)
