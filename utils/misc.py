"""Small miscellaneous helpers (timers, counting, device pick)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def timer(tag: str = "", printer=print) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    printer(f"[timer] {tag}: {dt:.3f}s")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
