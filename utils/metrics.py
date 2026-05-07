"""Online metric accumulators for classification / regression heads."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch


@dataclass
class ClassificationMetrics:
    """Running top-1 accuracy + (optional) confusion matrix."""

    num_classes: int
    _n_correct: int = 0
    _n_total:   int = 0
    _conf:      np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._conf = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        tgt = target.detach().cpu().numpy()
        self._n_correct += int((pred == tgt).sum())
        self._n_total   += int(tgt.size)
        for t, p in zip(tgt, pred):
            self._conf[int(t), int(p)] += 1

    def compute(self) -> Dict[str, float]:
        acc = self._n_correct / max(self._n_total, 1)
        return {"acc_top1": acc, "n_total": self._n_total}

    def reset(self) -> None:
        self._n_correct = 0
        self._n_total = 0
        self._conf.fill(0)

    @property
    def confusion_matrix(self) -> np.ndarray:
        return self._conf.copy()


@dataclass
class RegressionMetrics:
    """Running MSE / MAE / peak-error / normalized waveform correlation."""

    _sum_se: float = 0.0
    _sum_ae: float = 0.0
    _n_elements: int = 0
    _peak_errors: List[float] = field(default_factory=list)
    _corrs: List[float] = field(default_factory=list)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().float()
        target = target.detach().float()
        diff = pred - target
        self._sum_se += float((diff ** 2).sum())
        self._sum_ae += float(diff.abs().sum())
        self._n_elements += int(diff.numel())
        # per-sample peak error & corr
        p_peak = pred.abs().amax(dim=-1)
        t_peak = target.abs().amax(dim=-1)
        self._peak_errors.extend(((p_peak - t_peak) / t_peak.clamp_min(1e-12))
                                 .cpu().numpy().tolist())
        # pearson correlation (per waveform)
        p_mean = pred.mean(dim=-1, keepdim=True)
        t_mean = target.mean(dim=-1, keepdim=True)
        num = ((pred - p_mean) * (target - t_mean)).sum(dim=-1)
        den = torch.sqrt(((pred - p_mean) ** 2).sum(-1) *
                         ((target - t_mean) ** 2).sum(-1)).clamp_min(1e-12)
        self._corrs.extend((num / den).cpu().numpy().tolist())

    def compute(self) -> Dict[str, float]:
        mse = self._sum_se / max(self._n_elements, 1)
        mae = self._sum_ae / max(self._n_elements, 1)
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": mae,
            "peak_err_rel_mean": float(np.mean(self._peak_errors)) if self._peak_errors else 0.0,
            "peak_err_rel_abs_mean": float(np.mean(np.abs(self._peak_errors))) if self._peak_errors else 0.0,
            "pearson_corr_mean": float(np.mean(self._corrs)) if self._corrs else 0.0,
        }

    def reset(self) -> None:
        self._sum_se = 0.0
        self._sum_ae = 0.0
        self._n_elements = 0
        self._peak_errors.clear()
        self._corrs.clear()
