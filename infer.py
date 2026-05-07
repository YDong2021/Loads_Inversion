"""Inference for a single (response) input.

Given a 3-channel response curve stored in an ``.npy`` or ``.h5`` file,
run the classifier to estimate the impact **location** (pos_id / node id)
and then the regressor (conditioned on the predicted pos_id) to
reconstruct the **load curve**.

Usage
-----
>>> python infer.py \\
...     --regressor-ckpt checkpoints/regressor_mamba8/best.pth \\
...     --classifier-ckpt checkpoints/classifier_6_8_10/best.pth \\
...     --input path/to/response.npy \\
...     --out-dir outputs/infer_single

``response.npy`` must be an array with shape ``(3, L)`` or ``(B, 3, L)``.
Values are assumed to already be in physical units — they will be divided
by ``norm_scale`` that was stored inside the regressor checkpoint
(via its ``config.data.norm_scale`` replicate if present, otherwise
recomputed by rebuilding ``build_datasets``).

Outputs
-------
* ``prediction.npz``   — predicted load (B, L), pos_pred (B,), logits (B, num_classes)
* ``plots/*.png``      — one PNG per input sample
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from data import build_datasets
from models import build_classifier, build_regressor
from utils.logger import build_logger


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_input(path: str) -> np.ndarray:
    """Load a response tensor of shape (3, L) or (B, 3, L) from .npy / .npz / .h5."""
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix == ".npz":
        z = np.load(p)
        arr = z[z.files[0]]
    elif p.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            key = list(f.keys())[0]
            arr = f[key][...]
    else:
        raise ValueError(f"unsupported input format: {p.suffix}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]                           # (1, 3, L)
    assert arr.ndim == 3 and arr.shape[1] == 3, f"expected (B, 3, L); got {arr.shape}"
    return arr


def _get_norm_scale(reg_state: dict, device) -> float:
    """Best-effort recovery of the global peak normalizer scale used at training time."""
    cfg = reg_state["config"]
    # preferred path: cached in the config itself
    if "norm_scale" in cfg.get("data", {}):
        return float(cfg["data"]["norm_scale"])
    # fall-back: rebuild train set once and read it from info
    _, _, _, info = build_datasets(cfg["data"], seed=int(cfg.get("seed", 42)))
    return float(info["norm_scale"])


def _plot_single(force_pred: np.ndarray, pos_pred: int, save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(force_pred.shape[-1])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t, force_pred, linewidth=1.2)
    ax.set_xlabel("time step")
    ax.set_ylabel("force")
    ax.set_title(f"predicted load  (pos_id={pos_pred})")
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
    p.add_argument("--classifier-ckpt", type=str, required=True)
    p.add_argument("--input", type=str, required=True,
                   help="Path to a (3, L) or (B, 3, L) response tensor.")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("infer", out_dir, filename="infer.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device = {device}")

    # ---- load models ----
    cls_state = torch.load(args.classifier_ckpt, map_location=device)
    reg_state = torch.load(args.regressor_ckpt,  map_location=device)
    cls_cfg = cls_state["config"]
    reg_cfg = reg_state["config"]

    classifier = build_classifier(cls_cfg["model"]).to(device)
    classifier.load_state_dict(cls_state["model"])
    classifier.eval()

    regressor = build_regressor(reg_cfg["model"]).to(device)
    regressor.load_state_dict(reg_state["model"])
    regressor.eval()

    norm_scale = _get_norm_scale(reg_state, device)
    logger.info(f"using norm_scale = {norm_scale:.6g}")

    # ---- load input ----
    resp = _load_input(args.input)                # (B, 3, L)
    B, C, L = resp.shape
    target_len = int(reg_cfg["data"].get("target_len", L))
    if L != target_len:
        raise ValueError(
            f"input length {L} != model target_len {target_len}; "
            f"resample the signal before calling infer.py"
        )
    resp_t = torch.from_numpy(resp / norm_scale).to(device)

    # ---- forward ----
    logits = classifier(resp_t)                    # (B, num_classes)
    pos_pred = logits.argmax(dim=1)                # (B,)
    force_pred = regressor(resp_t, pos_pred).cpu().numpy()
    logits_np  = logits.cpu().numpy()
    pos_pred_np = pos_pred.cpu().numpy()

    np.savez(out_dir / "prediction.npz",
             pred=force_pred, pos_pred=pos_pred_np, logits=logits_np)
    logger.info(f"saved {out_dir / 'prediction.npz'} (B={B})")

    # ---- plots ----
    if not args.no_plot:
        plot_dir = out_dir / "plots"
        for i in range(B):
            _plot_single(
                force_pred=force_pred[i],
                pos_pred=int(pos_pred_np[i]),
                save_path=plot_dir / f"sample_{i:03d}_pos{int(pos_pred_np[i])}.png",
            )
        logger.info(f"wrote {B} plot(s) to {plot_dir}")


if __name__ == "__main__":
    main()
