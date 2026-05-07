"""models/ — layers, blocks, networks, losses.

The ``build_*`` helpers below are the only symbols used by entry scripts.
Other modules expose internal building blocks for compositional reuse.
"""
from .classifier import (
    build_classifier,
    ResNet1D,
    ResNet1D_6_8_10,
    ResNet1D_4_6_8,
    ResNet1D_8_8_8,
    ResNet1D_8_10_12,
)
from .regressor import (
    build_regressor,
    MambaRegressor,
    MambaRegressor6,
    MambaRegressor8,
    MambaRegressor10,
)
from .losses import build_loss, CombinedImpactLoss
from .positional_encoding import FourierPositionalEncoding

__all__ = [
    # classifier
    "build_classifier",
    "ResNet1D", "ResNet1D_6_8_10", "ResNet1D_4_6_8",
    "ResNet1D_8_8_8", "ResNet1D_8_10_12",
    # regressor
    "build_regressor",
    "MambaRegressor", "MambaRegressor6", "MambaRegressor8", "MambaRegressor10",
    # loss / PE
    "build_loss", "CombinedImpactLoss",
    "FourierPositionalEncoding",
]
