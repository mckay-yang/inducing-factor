"""Temporal train/validation split tools and validation gating for prediction."""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)

from inducing_factor.core.evaluation import evaluate_calibration

logger = logging.getLogger(__name__)


def temporal_holdout_split(
    df: pd.DataFrame,
    train_end_year: int,
    val_years: Optional[list[int]] = None,
    year_col: str = "year",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame by year into train and validation sets.

    Args:
        df: Input DataFrame with a year column.
        train_end_year: Latest year to include in training.
        val_years: Explicit validation years. If None, auto-derived as all
            years strictly after train_end_year present in the data.
        year_col: Name of the year column.

    Returns:
        (train_df, val_df) tuple.
    """
    if val_years is None:
        val_years = sorted(df[df[year_col] > train_end_year][year_col].unique().tolist())

    train_mask = df[year_col] <= train_end_year
    val_mask = df[year_col].isin(val_years)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    n_train = len(train_df)
    n_val = len(val_df)
    if n_train == 0:
        logger.warning("Temporal split produced empty training set (train_end_year=%d)", train_end_year)
    if n_val == 0:
        logger.warning("Temporal split produced empty validation set (val_years=%s)", val_years)

    logger.info(
        "Temporal split: train=%d rows (≤%d), val=%d rows (years=%s)",
        n_train, train_end_year, n_val, val_years,
    )
    return train_df, val_df


def compute_validation_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    y_pred_class: Optional[np.ndarray] = None,
    class_labels: Optional[dict] = None,
) -> dict:
    """Compute standard validation metrics for disturbance prediction.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_pred_prob: Predicted probabilities.
        y_pred_class: Predicted class labels (1-4). Optional - only for per-class.
        class_labels: Mapping from class ID to name, e.g. {1: "fire", ...}.

    Returns:
        Dict with keys: auc, pr_auc, brier_score, calibration, n_val, prevalence.
        If y_pred_class and class_labels provided, also includes per_class_metrics.
    """
    metrics: dict = {}

    if len(np.unique(y_true)) < 2:
        logger.warning("Validation set has only one class — metrics may be degenerate")
        metrics["auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        metrics["brier_score"] = float("nan")
        metrics["calibration_slope"] = float("nan")
        metrics["n_val"] = int(len(y_true))
        metrics["prevalence"] = float(y_true.mean())
        return metrics

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        metrics["auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_pred_prob))
    except Exception:
        metrics["pr_auc"] = float("nan")

    try:
        metrics["brier_score"] = float(brier_score_loss(y_true, y_pred_prob))
    except Exception:
        metrics["brier_score"] = float("nan")

    calibration = evaluate_calibration(y_true, y_pred_prob)
    metrics["calibration"] = calibration.get("reliability", [])
    metrics["calibration_slope"] = _estimate_calibration_slope(calibration)

    metrics["n_val"] = int(len(y_true))
    metrics["prevalence"] = float(y_true.mean())

    if y_pred_class is not None and class_labels is not None:
        per_class = {}
        present_classes = [c for c in class_labels if (y_pred_class == c).any()]
        for cls_id in present_classes:
            cls_mask = y_pred_class == cls_id
            cls_true = y_true[cls_mask]
            if len(cls_true) == 0:
                continue
            per_class[class_labels[cls_id]] = {
                "count": int(cls_mask.sum()),
                "prevalence": float(cls_true.mean()),
            }
        metrics["per_class"] = per_class

    return metrics


def _estimate_calibration_slope(calibration: dict) -> float:
    """Estimate calibration slope from reliability data using simple linear fit."""
    reliability = calibration.get("reliability", [])
    if len(reliability) < 2:
        return float("nan")
    x = np.array([r["predicted_prob"] for r in reliability])
    y = np.array([r["observed_freq"] for r in reliability])
    w = np.array([r["count"] for r in reliability])
    mask = w > 0
    if mask.sum() < 2:
        return float("nan")
    x, y, w = x[mask], y[mask], w[mask]
    A = np.vstack([x, np.ones_like(x)]).T
    W = np.diag(w)
    try:
        coeffs = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ y, rcond=None)[0]
        return float(coeffs[0])
    except Exception:
        return float("nan")


def check_validation_gate(
    metrics: dict,
    thresholds: Optional[dict] = None,
) -> Tuple[bool, str]:
    """Check whether validation metrics pass configured thresholds.

    Args:
        metrics: Dict from compute_validation_metrics.
        thresholds: Dict like {"auc_min": 0.60, "pr_auc_min": 0.05}. If None,
            no gate is enforced (always passes).

    Returns:
        (passed, reason) tuple.
    """
    if thresholds is None or not thresholds:
        return True, "no thresholds configured — gate open"

    failures = []
    for key, threshold in thresholds.items():
        if key.endswith("_min"):
            metric_name = key[:-4]
            value = metrics.get(metric_name, float("nan"))
            if not np.isfinite(value) or value < threshold:
                failures.append(f"{metric_name}={value:.4f} < {threshold}")

    if failures:
        return False, "validation gate failed: " + "; ".join(failures)
    return True, f"validation gate passed — all metrics above thresholds (n_val={metrics.get('n_val', 0)})"
