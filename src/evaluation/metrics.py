"""Classification and probability-quality metrics (positive class = 1)."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, roc_auc_score, roc_curve)

PROB_METRICS = ["roc_auc", "pr_auc", "brier", "log_loss", "ece"]
LABEL_METRICS = ["sensitivity", "specificity", "precision", "f1", "tn", "fp", "fn", "tp"]


def check_probabilities(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("Probabilities must be 1-D (positive-class probability)")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return p


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Equal-width-bin ECE: weighted mean |observed rate - mean predicted| over bins."""
    y, p = np.asarray(y), check_probabilities(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def compute_metrics(y, p, threshold: float = 0.5, n_bins: int = 10) -> dict:
    """All metrics. Hard-label metrics use ``p >= threshold``.

    precision / f1 are NaN when undefined (no predicted positives); ROC/PR AUC need both classes.
    """
    y = np.asarray(y).astype(int)
    p = check_probabilities(p)
    if len(y) != len(p):
        raise ValueError("y and p have different lengths")
    if len(np.unique(y)) < 2:
        raise ValueError("Both classes are required to compute AUC metrics")
    threshold = np.asarray(threshold, dtype=float)  # scalar, or one threshold per row
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
            "ece": expected_calibration_error(y, p, n_bins),
            "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "threshold": float(np.mean(threshold))}


def youden_threshold(y, p) -> float:
    """Threshold maximising sensitivity + specificity - 1 (ties -> lowest threshold). Use on TRAINING-side data only."""
    y, p = np.asarray(y).astype(int), check_probabilities(p)
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    j[~np.isfinite(thr)] = -np.inf  # drop the artificial "predict nothing" point
    best = np.flatnonzero(j == j.max())[-1]  # roc_curve thresholds are decreasing -> last = lowest
    return float(thr[best])


def reliability_table(y, p, n_bins: int = 10, strategy: str = "quantile"):
    frac, mean_p = calibration_curve(np.asarray(y), check_probabilities(p), n_bins=n_bins, strategy=strategy)
    return mean_p, frac


def calibration_slope_intercept(y, p, eps: float = 1e-6) -> tuple[float, float]:
    """Logistic recalibration y ~ a + b*logit(p), unpenalised. b < 1 suggests over-confident probabilities."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(check_probabilities(p), eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(z, np.asarray(y).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])
