"""YAML config loader supporting ``_base_`` inheritance.

Example::

    # exp.yaml
    _base_: default.yaml
    train: { lr: 2.0e-4 }

``_base_`` is resolved relative to the current file. Dict values are merged
recursively; lists / scalars are overwritten.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file, resolving ``_base_`` inheritance chains."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg.pop("_base_", None)
    if base is not None:
        base_path = (path.parent / base).resolve()
        base_cfg = load_config(base_path)
        merged = _deep_update(base_cfg, cfg)
        return merged
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
