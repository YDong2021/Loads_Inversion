"""ResNet-1D position classifier.

Per prompt.md the network starts with a 1×1 conv (3 → 128), then three stages
of 1×3 convs with channels [128, 256, 512].  Every two consecutive conv layers
form a BasicBlock with a residual connection; the very first block of the
whole network does not have a residual (mirroring the prompt description).

Four variants differ only in the number of conv layers per stage:
    [6, 8, 10]  (default)
    [4, 6, 8]
    [8, 8, 8]
    [8, 10, 12]

Each "stage X has N conv layers" is implemented as N//2 BasicBlock1D modules
(each block holds two convs).  Stage-to-stage transition uses stride-2.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .blocks import ResNetStage


# ===========================================================================
# Core network
# ===========================================================================

class ResNet1D(nn.Module):
    """Configurable 1-D ResNet for 3-channel response classification."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 50,
        stem_channels: int = 128,
        stage_channels: List[int] = (128, 256, 512),
        stage_blocks:   List[int] = (6, 8, 10),     # conv count per stage
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(stage_channels) == 3 and len(stage_blocks) == 3, \
            "exactly 3 stages are expected"
        for n in stage_blocks:
            assert n >= 2 and n % 2 == 0, \
                f"each stage needs an even # of convs (>=2); got {n}"

        # Stem: 1×1 conv raises channels to stem_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 1 — stride=1 throughout, first block without residual (per prompt)
        self.stage1 = ResNetStage(
            in_channels=stem_channels,
            out_channels=stage_channels[0],
            num_blocks=stage_blocks[0] // 2,
            kernel_size=kernel_size,
            stride_first=1,
            first_block_residual=False,
        )
        # Stage 2 — stride=2 on its first block (also doubles channels)
        self.stage2 = ResNetStage(
            in_channels=stage_channels[0],
            out_channels=stage_channels[1],
            num_blocks=stage_blocks[1] // 2,
            kernel_size=kernel_size,
            stride_first=2,
            first_block_residual=True,
        )
        # Stage 3 — stride=2 on its first block
        self.stage3 = ResNetStage(
            in_channels=stage_channels[1],
            out_channels=stage_channels[2],
            num_blocks=stage_blocks[2] // 2,
            kernel_size=kernel_size,
            stride_first=2,
            first_block_residual=True,
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(stage_channels[2], num_classes)

        self._init_weights()

    # standard Kaiming init for conv + linear
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # --- feature extractor (used by the regressor to get pos-id features) ---
    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x                              # (B, C, L')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = self.global_pool(f).flatten(1)     # (B, C)
        f = self.dropout(f)
        return self.fc(f)                      # (B, num_classes)


# ===========================================================================
# Variants — thin subclasses registered for config lookup
# ===========================================================================

class ResNet1D_6_8_10(ResNet1D):
    def __init__(self, **kw):
        kw.setdefault("stage_blocks", [6, 8, 10])
        super().__init__(**kw)


class ResNet1D_4_6_8(ResNet1D):
    def __init__(self, **kw):
        kw.setdefault("stage_blocks", [4, 6, 8])
        super().__init__(**kw)


class ResNet1D_8_8_8(ResNet1D):
    def __init__(self, **kw):
        kw.setdefault("stage_blocks", [8, 8, 8])
        super().__init__(**kw)


class ResNet1D_8_10_12(ResNet1D):
    def __init__(self, **kw):
        kw.setdefault("stage_blocks", [8, 10, 12])
        super().__init__(**kw)


_CLASSIFIER_REGISTRY = {
    "resnet1d_6_8_10": ResNet1D_6_8_10,
    "resnet1d_4_6_8":  ResNet1D_4_6_8,
    "resnet1d_8_8_8":  ResNet1D_8_8_8,
    "resnet1d_8_10_12": ResNet1D_8_10_12,
}


def build_classifier(model_cfg: Dict) -> ResNet1D:
    name = model_cfg["name"]
    if name not in _CLASSIFIER_REGISTRY:
        raise KeyError(f"unknown classifier variant: {name}. "
                       f"Available: {list(_CLASSIFIER_REGISTRY)}")
    kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    # drop "num_blocks" if someone typed it; we use "stage_blocks"
    kwargs.pop("num_blocks", None)
    return _CLASSIFIER_REGISTRY[name](**kwargs)
