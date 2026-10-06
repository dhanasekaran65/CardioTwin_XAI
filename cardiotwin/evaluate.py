"""Discrimination + calibration metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def evaluate(y, p, threshold: float | None = None) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    thr = youden_threshold(y, p) if threshold is None else threshold
    pred = p >= thr
    tp, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    fp, fn = ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
    return {
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "threshold": float(thr),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "accuracy": float((tp + tn) / len(y)),
    }
