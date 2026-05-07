"""Entry script: train the Mamba-based load-shape regressor.

The regressor uses the dense ``pos_id`` as an extra conditioning input
through a Fourier positional encoding.  During training we always use
the **ground-truth** pos_id (teacher forcing).  The pretrained classifier
is only used at inference time (see ``infer.py`` / ``eval.py``).

Usage
-----
>>> python train_regressor.py --config configs/regressor_mamba8.yaml
>>> python train_regressor.py --config configs/regressor_mamba6.yaml --resume checkpoints/regressor_mamba8/last.pth
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import build_datasets
from engine import (
    build_optimizer,
    build_scheduler,
    evaluate_regressor,
    train_one_epoch,
)
from models import build_loss, build_regressor
from utils import (
    build_logger,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
)
from utils.logger import build_tb_writer


# -----------------------------------------------------------------------------
# Adapters for train_one_epoch (regressor task, teacher forcing)
# -----------------------------------------------------------------------------

def _reg_forward(model, batch):
    # teacher forcing: ground-truth pos_id drives the positional encoding
    return model(batch["response"], batch["pos_id"])


def _make_reg_loss(criterion):
    def _fn(pred, batch):
        return criterion(pred, batch["force"])   # returns dict
    return _fn


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="Path to a regressor YAML config.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    assert cfg.get("task") == "regressor", "this script is for the regressor task"

    exp_name = cfg.get("exp_name", "regressor")
    log_dir = Path(cfg["log_dir"]) / exp_name
    ckpt_dir = Path(cfg["ckpt_dir"]) / exp_name
    logger = build_logger(exp_name, log_dir)
    tb = build_tb_writer(log_dir / "tb")

    (log_dir / "config.resolved.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    logger.info(f"device = {device}")

    teacher_forcing = bool(cfg.get("teacher_forcing", True))
    logger.info(f"teacher_forcing = {teacher_forcing}")

    # ---- data ----
    train_ds, val_ds, test_ds, info = build_datasets(cfg["data"], seed=int(cfg.get("seed", 42)))
    logger.info(f"datasets: train={info['n_train']} val={info['n_val']} test={info['n_test']} "
                f"target_len={info['target_len']} norm_scale={info['norm_scale']:.6f}")

    train_cfg = cfg["train"]
    bs = int(train_cfg["batch_size"])
    nw = int(cfg.get("num_workers", 4))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=True,
                              persistent_workers=nw > 0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=nw > 0)

    # ---- model / loss / optim ----
    model = build_regressor(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model {cfg['model']['name']}: {n_params/1e6:.2f} M params")

    criterion = build_loss(cfg["loss"]).to(device)
    optimizer = build_optimizer(model.parameters(), train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, steps_per_epoch=len(train_loader))

    # ---- resume ----
    start_epoch = 0
    best_metric = float("inf")        # val loss: lower = better
    global_step = 0
    patience_left = int(train_cfg.get("early_stop_patience", 0))
    if args.resume is not None and Path(args.resume).is_file():
        state = load_checkpoint(args.resume, model=model, optimizer=optimizer,
                                map_location=device)
        start_epoch = int(state.get("epoch", 0))
        best_metric = float(state.get("best_metric", float("inf")))
        global_step = int(state.get("global_step", 0))
        logger.info(f"resumed from {args.resume} (epoch={start_epoch}, best_loss={best_metric:.6f})")

    # ---- training loop ----
    epochs = int(train_cfg["epochs"])
    loss_fn = _make_reg_loss(criterion)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            forward_fn=_reg_forward,
            loss_fn=loss_fn,
            epoch=epoch,
            logger=logger,
            tb_writer=tb,
            log_every=int(train_cfg.get("log_every", 20)),
            grad_clip=float(train_cfg.get("grad_clip", 0.0)),
            amp=bool(train_cfg.get("amp", False)),
            global_step=global_step,
        )
        global_step = int(stats.pop("_global_step", global_step))

        # ---- validation (teacher forcing) ----
        val = evaluate_regressor(
            model, val_loader, device, criterion,
            use_pred_pos=False, classifier=None,
        )
        dt = time.time() - t0
        comp = " ".join(
            f"{k}={val.get(k, 0.0):.4f}" for k in ("mse", "grad", "stage", "peak") if k in val
        )
        logger.info(
            f"[epoch {epoch}] train_loss={stats['loss']:.4f}  "
            f"val_loss={val['loss']:.4f}  {comp}  "
            f"val_rmse={val.get('rmse', 0.0):.6f}  "
            f"val_corr={val.get('pearson_corr_mean', 0.0):.4f}  "
            f"val_peak_err={val.get('peak_err_rel_abs_mean', 0.0):.4f}  ({dt:.1f}s)"
        )
        if tb is not None:
            for k, v in val.items():
                try:
                    tb.add_scalar(f"val/{k}", float(v), epoch)
                except Exception:
                    pass

        # ---- checkpoint ----
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_metric": best_metric,
            "global_step": global_step,
            "config": cfg,
        }
        save_checkpoint(state, ckpt_dir, filename="last.pth")
        if val["loss"] < best_metric:
            best_metric = float(val["loss"])
            state["best_metric"] = best_metric
            save_checkpoint(state, ckpt_dir, filename="best.pth")
            logger.info(f"  ↓ new best val_loss={best_metric:.6f}, saved best.pth")
            patience_left = int(train_cfg.get("early_stop_patience", 0))
        else:
            if int(train_cfg.get("early_stop_patience", 0)) > 0:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info(f"early stop at epoch {epoch} (best val_loss={best_metric:.6f})")
                    break

    logger.info(f"training done. best val loss = {best_metric:.6f}")
    if tb is not None:
        tb.close()


if __name__ == "__main__":
    main()
