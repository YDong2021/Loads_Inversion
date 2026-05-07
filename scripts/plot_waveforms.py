"""Plot waveforms from a ``preds.npz`` produced by ``eval.py``.

Supports two modes:

1. **overlay** (default): for each sample, plot ground-truth and predicted
   load on the same axes.
2. **grid**: draw a single figure containing an NxM grid of samples.

Usage
-----
>>> python -m scripts.plot_waveforms \\
...     --preds outputs/eval_mamba8/preds.npz \\
...     --out-dir outputs/eval_mamba8/waveforms_extra \\
...     --num-samples 24 --mode grid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_overlay(preds: np.ndarray, truths: np.ndarray,
                 pos_gt: np.ndarray, pos_pr: np.ndarray,
                 force_id: np.ndarray, sel: np.ndarray, out_dir: Path) -> None:
    plt = _setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, idx in enumerate(sel):
        t = np.arange(preds.shape[-1])
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(t, truths[idx], label="ground truth", linewidth=1.2)
        ax.plot(t, preds[idx],  label="prediction",  linewidth=1.0, alpha=0.85)
        ax.set_xlabel("time step")
        ax.set_ylabel("force")
        ax.set_title(f"force_id={int(force_id[idx])}  "
                     f"pos_gt={int(pos_gt[idx])}  pos_pred={int(pos_pr[idx])}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{k:03d}.png", dpi=120)
        plt.close(fig)


def plot_grid(preds: np.ndarray, truths: np.ndarray,
              pos_gt: np.ndarray, pos_pr: np.ndarray,
              force_id: np.ndarray, sel: np.ndarray,
              out_path: Path, rows: int, cols: int) -> None:
    plt = _setup_matplotlib()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.0), squeeze=False)
    t = np.arange(preds.shape[-1])
    for i, idx in enumerate(sel[: rows * cols]):
        ax = axes[i // cols][i % cols]
        ax.plot(t, truths[idx], linewidth=0.9)
        ax.plot(t, preds[idx],  linewidth=0.9, alpha=0.85)
        ax.set_title(f"f{int(force_id[idx])} p_gt{int(pos_gt[idx])} p_hat{int(pos_pr[idx])}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    # hide unused cells
    for j in range(len(sel), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=str, required=True,
                   help="Path to preds.npz written by eval.py")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=24)
    p.add_argument("--mode", choices=["overlay", "grid"], default="overlay")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.preds)
    preds  = data["pred"]
    truths = data["true"]
    pos_gt = data["pos_gt"]
    pos_pr = data["pos_pred"]
    fid    = data["force_id"]

    rng = np.random.default_rng(args.seed)
    n = preds.shape[0]
    k = min(args.num_samples, n)
    sel = rng.choice(n, size=k, replace=False)

    out_dir = Path(args.out_dir)
    if args.mode == "overlay":
        plot_overlay(preds, truths, pos_gt, pos_pr, fid, sel, out_dir)
        print(f"wrote {k} overlay PNGs to {out_dir}")
    else:
        out_path = out_dir / f"grid_{args.rows}x{args.cols}.png"
        plot_grid(preds, truths, pos_gt, pos_pr, fid, sel,
                  out_path, args.rows, args.cols)
        print(f"wrote grid plot to {out_path}")


if __name__ == "__main__":
    main()
