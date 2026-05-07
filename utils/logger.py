"""Unified logger: stdout + rotating file + (optional) TensorBoard."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def build_logger(
    name: str,
    log_dir: str | Path,
    level: int = logging.INFO,
    filename: str = "train.log",
) -> logging.Logger:
    """Return a logger writing to both console and ``log_dir/filename``."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # avoid duplicated handlers when called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = RotatingFileHandler(
        log_dir / filename, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def build_tb_writer(log_dir: str | Path):
    """Best-effort TensorBoard SummaryWriter (returns ``None`` if unavailable)."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover
        return None
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))
