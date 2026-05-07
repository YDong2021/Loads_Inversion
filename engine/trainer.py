"""Training-loop helpers: optimizer, scheduler, one-epoch runner.

Both the classifier and the regressor reuse :func:`train_one_epoch` by
supplying different ``forward_fn`` / ``loss_fn`` callables.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, _LRScheduler


# =====================================================================
# Optimizer / scheduler factories
# =====================================================================

def build_optimizer(params, train_cfg: Dict) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=float(train_cfg.get("momentum", 0.9)),
                   weight_decay=wd)
    raise KeyError(f"unknown optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: Dict,
    steps_per_epoch: int,
) -> Optional[_LRScheduler]:
    name = str(train_cfg.get("scheduler", "none")).lower()
    if name == "cosine":
        total = int(train_cfg["epochs"]) * max(steps_per_epoch, 1)
        return CosineAnnealingLR(optimizer, T_max=total, eta_min=float(train_cfg.get("eta_min", 0.0)))
    if name == "step":
        return StepLR(optimizer,
                      step_size=int(train_cfg.get("step_size", 30)),
                      gamma=float(train_cfg.get("gamma", 0.1)))
    if name == "none":
        return None
    raise KeyError(f"unknown scheduler: {name}")


# =====================================================================
# Generic one-epoch trainer
# =====================================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[_LRScheduler],
    device: torch.device,
    forward_fn: Callable[[torch.nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
    loss_fn: Callable[[torch.Tensor, Dict[str, torch.Tensor]], Any],
    *,
    epoch: int,
    logger=None,
    tb_writer=None,
    log_every: int = 20,
    grad_clip: float = 0.0,
    amp: bool = False,
    global_step: int = 0,
    on_loss=None,
) -> Dict[str, float]:
    """Run a single training epoch.

    ``forward_fn``    : ``(model, batch) -> prediction``
    ``loss_fn``       : ``(prediction, batch) -> {"loss": Tensor, ...}`` or scalar Tensor
    ``on_loss``       : optional callback ``(loss_dict, batch)`` for metric updates
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    running = {"loss": 0.0}
    n_batches = 0
    for i, batch in enumerate(loader):
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            pred = forward_fn(model, batch)
            loss_out = loss_fn(pred, batch)
            if isinstance(loss_out, dict):
                loss = loss_out["loss"]
                log_items = {k: float(v) if torch.is_tensor(v) else float(v)
                             for k, v in loss_out.items() if k != "loss"}
            else:
                loss = loss_out
                log_items = {}

        if on_loss is not None:
            on_loss(pred, batch, loss_out)

        if not torch.isfinite(loss):
            if logger: logger.warning("non-finite loss -- skipping step")
            continue

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        # aggregate
        running["loss"] += float(loss.detach())
        for k, v in log_items.items():
            running[k] = running.get(k, 0.0) + v
        n_batches += 1
        global_step += 1

        if logger and (i + 1) % log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            comp = " ".join(f"{k}={v:.4f}" for k, v in log_items.items())
            logger.info(f"[epoch {epoch}][{i+1}/{len(loader)}] "
                        f"loss={float(loss):.4f} {comp} lr={lr_now:.2e}")
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", float(loss), global_step)
            for k, v in log_items.items():
                tb_writer.add_scalar(f"train/{k}", v, global_step)

    n_batches = max(n_batches, 1)
    out = {k: v / n_batches for k, v in running.items()}
    out["_global_step"] = global_step
    return out
