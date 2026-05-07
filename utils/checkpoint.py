"""Checkpoint save/load with best-metric tracking and resume support."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state: Dict[str, Any],
    ckpt_dir: str | Path,
    filename: str = "last.pth",
) -> Path:
    """Save a state dict (model + optimizer + extra fields) under ``ckpt_dir``."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    torch.save(state, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint; optionally populate ``model`` / ``optimizer``.

    Returns the raw state dict so the caller can pick auxiliary fields
    (``epoch``, ``best_metric``, …).
    """
    state = torch.load(str(path), map_location=map_location)
    if model is not None and "model" in state:
        model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state
