"""End-to-end evaluation of the classifier + regressor pipeline on the
held-out test split.

Produces:
* ``metrics.json``     — aggregate classifier / regressor metrics
* ``waveforms/*.png``  — sample prediction vs ground-truth plots
* ``preds.npz``        — raw arrays (true, pred, pos_gt, pos_pred, force_id)

Usage
-----
>>> python eval.py \\
...     --regressor-ckpt checkpoints/regressor_mamba8/best.pth \\
...     --classifier-ckpt checkpoints/classifier_6_8_10/best.pth \\
...     --out-dir outputs/eval_mamba8

The regressor checkpoint carries the resolved config; the classifier
checkpoint does the same. We rebuild both models from their saved configs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import build_datasets
from engine import evaluate_classifier, evaluate_regressor
from models import build_classifier, build_loss, build_regressor
from utils import load_checkpoint, set_seed
from utils.logger import build_logger


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_ckpt_cfg(ckpt_path: str, device) -> Dict:
    state = torch.load(ckpt_path, map_location=device)
    if "config" not in state:
        raise RuntimeError(f"checkpoint {ckpt_path} has no 'config' field; "
                           "cannot rebuild the model blindly.")
    return state


def _plot_waveform(force_true: np.ndarray, force_pred: np.ndarray,
                   pos_gt: int, pos_pred: int, force_id: int,
                   save_path: Path) -> None:
    """Save a simple PNG with ground-truth vs predicted load (no styling)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(force_true.shape[-1])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t, force_true, label="ground truth", linewidth=1.2)
    ax.plot(t, force_pred, label="prediction",  linewidth=1.0, alpha=0.85)
    ax.set_xlabel("time step")
    ax.set_ylabel("force")
    ax.set_title(f"force_id={force_id}  pos_gt={pos_gt}  pos_pred={pos_pred}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regressor-ckpt",  type=str, required=True)
    p.add_argument("--classifier-ckpt", type=str, default=None,
                   help="If provided, pos_id is predicted by the classifier "
                        "(end-to-end pipeline).  Otherwise teacher-forcing "
                        "ground-truth pos_id is used.")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-plots", type=int, default=20,
                   help="Number of sample waveform plots to save.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("eval", out_dir, filename="eval.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device = {device}")

    # ---- load regressor ----
    reg_state = _load_ckpt_cfg(args.regressor_ckpt, device)
    reg_cfg = reg_state["config"]
    set_seed(int(reg_cfg.get("seed", 42)))

    # build data (reuses the regressor's data config to match the split that
    # was used at training time)
    train_ds, val_ds, test_ds, info = build_datasets(reg_cfg["data"],
                                                     seed=int(reg_cfg.get("seed", 42)))
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    logger.info(f"{args.split} set: {len(ds)} samples (target_len={info['target_len']})")

    regressor = build_regressor(reg_cfg["model"]).to(device)
    regressor.load_state_dict(reg_state["model"])
    regressor.eval()

    criterion = build_loss(reg_cfg["loss"]).to(device)

    # ---- optionally load classifier ----
    classifier = None
    cls_cfg: Optional[Dict] = None
    if args.classifier_ckpt is not None:
        cls_state = _load_ckpt_cfg(args.classifier_ckpt, device)
        cls_cfg = cls_state["config"]
        classifier = build_classifier(cls_cfg["model"]).to(device)
        classifier.load_state_dict(cls_state["model"])
        classifier.eval()

    # ---- aggregate metrics ----
    metrics_all: Dict[str, Dict[str, float]] = {}

    if classifier is not None:
        cls_stats = evaluate_classifier(
            classifier, loader, device,
            num_classes=int(cls_cfg["model"]["num_classes"]),
        )
        metrics_all["classifier"] = cls_stats
        logger.info(f"classifier — acc_top1={cls_stats['acc_top1']:.4f}  "
                    f"loss={cls_stats['loss']:.4f}")

    reg_stats = evaluate_regressor(
        regressor, loader, device, criterion,
        use_pred_pos=classifier is not None, classifier=classifier,
    )
    metrics_all["regressor"] = reg_stats
    logger.info(
        "regressor — "
        + "  ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in reg_stats.items())
    )

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_all, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- dump raw preds + a few plots ----
    logger.info("collecting predictions for waveform plots...")
    preds_list, truths_list = [], []
    pos_gt_list, pos_pr_list, fid_list = [], [], []

    for batch in loader:
        resp = batch["response"].to(device, non_blocking=True)
        force = batch["force"]
        pos_gt = batch["pos_id"]
        fid = batch["force_id"]
        if classifier is not None:
            pos_pr = classifier(resp).argmax(dim=1)
        else:
            pos_pr = pos_gt.to(device)
        pred = regressor(resp, pos_pr).cpu()

        preds_list.append(pred.numpy())
        truths_list.append(force.numpy())
        pos_gt_list.append(pos_gt.numpy())
        pos_pr_list.append(pos_pr.cpu().numpy())
        fid_list.append(fid.numpy())

    preds   = np.concatenate(preds_list, axis=0)
    truths  = np.concatenate(truths_list, axis=0)
    pos_gt  = np.concatenate(pos_gt_list)
    pos_pr  = np.concatenate(pos_pr_list)
    force_id = np.concatenate(fid_list)
    np.savez(out_dir / "preds.npz",
             pred=preds, true=truths,
             pos_gt=pos_gt, pos_pred=pos_pr, force_id=force_id)

    # pick a deterministic but diverse subset for plotting
    n_plot = min(args.num_plots, preds.shape[0])
    rng = np.random.default_rng(0)
    sel = rng.choice(preds.shape[0], size=n_plot, replace=False)
    plot_dir = out_dir / "waveforms"
    for k, idx in enumerate(sel):
        _plot_waveform(
            force_true=truths[idx], force_pred=preds[idx],
            pos_gt=int(pos_gt[idx]), pos_pred=int(pos_pr[idx]),
            force_id=int(force_id[idx]),
            save_path=plot_dir / f"sample_{k:03d}_fid{int(force_id[idx])}_pos{int(pos_gt[idx])}.png",
        )
    logger.info(f"saved {n_plot} waveform plots to {plot_dir}")
    logger.info(f"metrics & raw preds saved to {out_dir}")


if __name__ == "__main__":
    main()
