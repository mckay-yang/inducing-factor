import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from typing import Dict


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    metrics = {}
    try:
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        metrics["brier_score"] = float("nan")
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"] = float("nan")
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        metrics["log_loss"] = float("nan")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    reliability = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            reliability.append({
                "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "observed_freq": float(y_true[mask].mean()),
                "predicted_prob": float(y_prob[mask].mean()),
                "count": int(mask.sum()),
            })
    metrics["reliability"] = reliability
    return metrics