"""Validation / evaluation loops for classifier and regressor."""
from __future__ import annotations

from typing import Dict

import torch

from utils.metrics import ClassificationMetrics, RegressionMetrics


@torch.no_grad()
def evaluate_classifier(model, loader, device, num_classes: int = 50) -> Dict[str, float]:
    model.eval()
    metrics = ClassificationMetrics(num_classes=num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_sum, n = 0.0, 0
    for batch in loader:
        resp = batch["response"].to(device, non_blocking=True)
        tgt  = batch["pos_id"].to(device, non_blocking=True)
        logits = model(resp)
        loss_sum += float(loss_fn(logits, tgt)) * resp.size(0)
        n += resp.size(0)
        metrics.update(logits, tgt)
    out = metrics.compute()
    out["loss"] = loss_sum / max(n, 1)
    return out


@torch.no_grad()
def evaluate_regressor(model, loader, device, loss_fn, use_pred_pos: bool = False,
                       classifier=None) -> Dict[str, float]:
    """Evaluate the regressor.

    ``use_pred_pos=False`` — use ground-truth pos_id (validation during training).
    ``use_pred_pos=True``  — use classifier's argmax as pos_id (full inference).
    """
    model.eval()
    if use_pred_pos:
        assert classifier is not None, "classifier needed for prediction-based pos_id"
        classifier.eval()

    metrics = RegressionMetrics()
    loss_sum, n = 0.0, 0
    loss_components: Dict[str, float] = {}
    for batch in loader:
        resp = batch["response"].to(device, non_blocking=True)
        force = batch["force"].to(device, non_blocking=True)
        pos_gt = batch["pos_id"].to(device, non_blocking=True)
        if use_pred_pos:
            logits = classifier(resp)
            pos_id = logits.argmax(dim=1)
        else:
            pos_id = pos_gt

        pred = model(resp, pos_id)
        out = loss_fn(pred, force)
        if isinstance(out, dict):
            loss = out["loss"]
            for k, v in out.items():
                if k == "loss":
                    continue
                loss_components[k] = loss_components.get(k, 0.0) + float(v) * resp.size(0)
        else:
            loss = out
        loss_sum += float(loss) * resp.size(0)
        n += resp.size(0)
        metrics.update(pred, force)

    agg = metrics.compute()
    agg["loss"] = loss_sum / max(n, 1)
    for k, v in loss_components.items():
        agg[k] = v / max(n, 1)
    return agg
