"""Load saved evaluation results for display and compute exploratory threshold summaries."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data.loader import load_config, resolve
from src.evaluation.metrics import compute_metrics, reliability_table

MODEL_ORDER = ["Bayesian Network", "Naive Bayes", "MLP"]


def results_available(cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return (resolve(cfg["paths"]["metrics"]) / "oof_predictions.csv").exists()


def load_results(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    m = resolve(cfg["paths"]["metrics"])
    return {"oof": pd.read_csv(m / "oof_predictions.csv"),
            "fold": pd.read_csv(m / "summary_fold_level.csv"),
            "pooled": pd.read_csv(m / "summary_pooled_repeats.csv"),
            "ci": pd.read_csv(m / "bootstrap_ci.csv"),
            "diff": pd.read_csv(m / "bootstrap_paired_differences.csv"),
            "attribution": pd.read_csv(m / "mlp_attribution.csv", index_col=0)}


def curve_points(oof: pd.DataFrame, kind: str, repeat: int) -> pd.DataFrame:
    rows = []
    for model, d in oof[oof.repeat == repeat].groupby("model"):
        if kind == "roc":
            x, y, _ = roc_curve(d.y_true, d.p_pos)
        else:
            y, x, _ = precision_recall_curve(d.y_true, d.p_pos)
        rows.append(pd.DataFrame({"x": x, "y": y, "model": model}))
    return pd.concat(rows, ignore_index=True)


def calibration_points(oof: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for model, d in oof.groupby("model"):
        mp, fr = reliability_table(d.y_true, d.p_pos, n_bins, "quantile")
        rows.append(pd.DataFrame({"predicted": mp, "observed": fr, "model": model}))
    return pd.concat(rows, ignore_index=True)


def metrics_at_threshold(oof: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Hard-label metrics at one threshold, averaged over the pooled out-of-fold predictions of each repeat.

    Exploratory: choosing a threshold by looking at these labels is optimistic; the study used a fixed 0.5."""
    rows = []
    for model, d in oof.groupby("model"):
        per = [compute_metrics(g.y_true, g.p_pos, threshold) for _, g in d.groupby("repeat")]
        rows.append({"model": model, **{k: sum(p[k] for p in per) / len(per) for k in
                                         ("sensitivity", "specificity", "precision", "f1", "tn", "fp", "fn", "tp")}})
    out = pd.DataFrame(rows).set_index("model").reindex(MODEL_ORDER)
    return out
