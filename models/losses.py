"""Loss functions for the loads-inversion project.

1.  Classifier — plain CrossEntropy (built by :func:`build_loss`).
2.  Regressor — :class:`CombinedImpactLoss` implementing

        L = α · L_MSE + β · L_grad + γ · L_stage + δ · L_peak

    where every component is defined in ``prompt.assets`` and the stage
    weights (w_accel, w_inertia, w_decay) encode a "trapezoidal phase mask":

        * accel   — rising edge (signal  <= plateau, going up)
        * inertia — plateau     (signal  > plateau_thresh)
        * decay   — falling edge (signal <= plateau, going down)

    All components operate on 1-D curves (B, T).  The stage mask is computed
    per-sample using thresholds on the *target* signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Stage mask helper (acts on the target curve)
# =====================================================================

def trapezoid_stage_mask(
    f: torch.Tensor,
    accel_low: float = 0.02,
    plateau:   float = 0.95,
) -> torch.Tensor:
    """Return a (B, 3, T) binary mask: [p_accel, p_inertia, p_decay] per sample.

    The three stages partition the "pulse support" of the target signal:
        * accel   — rising edge (before the plateau maximum)
        * inertia — plateau samples (f >= plateau · peak)
        * decay   — falling edge (after the plateau maximum)
    Samples outside the pulse (f < accel_low · peak) belong to none.
    """
    # peak per batch element
    peak = f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)          # (B, 1)
    f_abs = f.abs()
    active   = f_abs >= accel_low * peak                                 # pulse support
    plateau_m = f_abs >= plateau   * peak                                # plateau region

    # center = argmax along time (first occurrence of peak)
    center = f_abs.argmax(dim=-1, keepdim=True)                          # (B, 1)
    t_idx = torch.arange(f.shape[-1], device=f.device).unsqueeze(0)      # (1, T)

    rising = (t_idx <= center) & active & (~plateau_m)
    falling = (t_idx >  center) & active & (~plateau_m)

    mask = torch.stack([rising, plateau_m, falling], dim=1).float()      # (B, 3, T)
    return mask


# =====================================================================
# Combined impact loss
# =====================================================================

class CombinedImpactLoss(nn.Module):
    """α·MSE + β·grad + γ·stage + δ·peak (see prompt.md)."""

    def __init__(
        self,
        alpha: float = 0.8,
        beta:  float = 0.4,
        gamma: float = 1.0,
        delta: float = 0.8,
        stage_weights: Dict[str, float] | None = None,
        stage_thresholds: Dict[str, float] | None = None,
        peak_window: int = 5,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        sw = stage_weights or {}
        self.w_accel   = float(sw.get("accel",   1.3))
        self.w_inertia = float(sw.get("inertia", 1.0))
        self.w_decay   = float(sw.get("decay",   0.2))
        th = stage_thresholds or {}
        self.accel_low = float(th.get("accel_low", 0.02))
        self.plateau   = float(th.get("plateau",   0.95))
        self.peak_window = int(peak_window)

    # --------- individual terms ---------
    @staticmethod
    def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction="mean")

    @staticmethod
    def _grad(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d_pred = pred[..., 1:] - pred[..., :-1]
        d_tgt  = target[..., 1:] - target[..., :-1]
        return F.mse_loss(d_pred, d_tgt, reduction="mean")

    def _stage(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = trapezoid_stage_mask(
            target, accel_low=self.accel_low, plateau=self.plateau
        )                                                    # (B, 3, T)
        weights = mask[:, 0] * self.w_accel \
                + mask[:, 1] * self.w_inertia \
                + mask[:, 2] * self.w_decay                   # (B, T)
        se = (pred - target) ** 2
        # normalise by number of weighted samples to keep scale ~ MSE
        denom = weights.sum(dim=-1).clamp_min(1.0)            # (B,)
        per_sample = (weights * se).sum(dim=-1) / denom        # (B,)
        return per_sample.mean()

    def _peak(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # argmax of target, then ±peak_window window
        T = target.shape[-1]
        center = target.abs().argmax(dim=-1)                   # (B,)
        idx = torch.arange(T, device=target.device).unsqueeze(0)
        lo = (center - self.peak_window).clamp_min(0).unsqueeze(1)
        hi = (center + self.peak_window).clamp_max(T - 1).unsqueeze(1)
        peak_mask = (idx >= lo) & (idx <= hi)                  # (B, T)
        se = (pred - target) ** 2
        denom = peak_mask.sum(dim=-1).clamp_min(1).float()
        per_sample = (se * peak_mask.float()).sum(dim=-1) / denom
        return per_sample.mean()

    # --------- main ---------
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert pred.shape == target.shape, f"shape mismatch: {pred.shape} vs {target.shape}"
        l_mse   = self._mse(pred, target)
        l_grad  = self._grad(pred, target)
        l_stage = self._stage(pred, target)
        l_peak  = self._peak(pred, target)

        total = (self.alpha * l_mse + self.beta * l_grad
                 + self.gamma * l_stage + self.delta * l_peak)
        return {
            "loss":    total,
            "mse":     l_mse.detach(),
            "grad":    l_grad.detach(),
            "stage":   l_stage.detach(),
            "peak":    l_peak.detach(),
        }


# =====================================================================
# Factory
# =====================================================================

def build_loss(loss_cfg: Dict) -> nn.Module:
    name = loss_cfg["name"]
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)))
    if name == "combined_impact":
        return CombinedImpactLoss(
            alpha=loss_cfg.get("alpha", 0.8),
            beta=loss_cfg.get("beta", 0.4),
            gamma=loss_cfg.get("gamma", 1.0),
            delta=loss_cfg.get("delta", 0.8),
            stage_weights=loss_cfg.get("stage_weights"),
            stage_thresholds=loss_cfg.get("stage_thresholds"),
            peak_window=loss_cfg.get("peak_window", 5),
        )
    raise KeyError(f"unknown loss name: {name}")
