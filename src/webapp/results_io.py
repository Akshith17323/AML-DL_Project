"""Load saved evaluation results for display and compute exploratory threshold summaries."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data.loader import PHASE2_CONFIG_PATH, load_config, resolve
from src.evaluation.metrics import compute_metrics, reliability_table
from src.models.demo_models_phase2 import PHASE2_MODEL_NAMES
from src.models.pipelines import MODELS

MODEL_ORDER = list(MODELS)
PHASE2_MODEL_ORDER = list(PHASE2_MODEL_NAMES)


def results_available(cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return (resolve(cfg["paths"]["metrics"]) / "oof_predictions.csv").exists()


def load_results(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    m = resolve(cfg["paths"]["metrics"])
    return {
        "oof": pd.read_csv(m / "oof_predictions.csv"),
        "fold": pd.read_csv(m / "summary_fold_level.csv"),
        "pooled": pd.read_csv(m / "summary_pooled_repeats.csv"),
        "ci": pd.read_csv(m / "bootstrap_ci.csv"),
        "diff": pd.read_csv(m / "bootstrap_paired_differences.csv"),
    }


def phase2_results_available(cfg: dict | None = None) -> bool:
    """Check whether Phase 2 pooled out-of-fold evaluation results exist."""
    cfg = cfg or load_config(PHASE2_CONFIG_PATH)
    return (resolve(cfg["paths"]["metrics"]) / "oof_pooled.csv").exists()


def load_phase2_results(cfg: dict | None = None) -> dict:
    """Load Phase 2 multi-cohort evaluation metrics, tables, and intervals."""
    cfg = cfg or load_config(PHASE2_CONFIG_PATH)
    m = resolve(cfg["paths"]["metrics"])
    out = {}
    for key, filename in [
        ("oof_pooled", "oof_pooled.csv"),
        ("oof_cross_cohort", "oof_cross-cohort.csv"),
        ("summary_pooled", "summary_pooled.csv"),
        ("summary_cross_cohort", "summary_cross-cohort.csv"),
        ("ci_pooled", "bootstrap_ci_pooled.csv"),
        ("ci_cross_cohort", "bootstrap_ci_cross-cohort.csv"),
        ("ablations", "ablations.csv"),
        ("hyperparameters", "selected_hyperparameters_pooled.csv"),
    ]:
        file_path = m / filename
        out[key] = pd.read_csv(file_path) if file_path.exists() else pd.DataFrame()
    return out


def curve_points(oof: pd.DataFrame, kind: str, repeat: int = 0) -> pd.DataFrame:
    rows = []
    subset = oof[oof.repeat == repeat] if "repeat" in oof.columns else oof
    for model, d in subset.groupby("model"):
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
    """Hard-label metrics at one threshold, averaged over the pooled out-of-fold predictions.

    Exploratory: choosing a threshold by looking at these labels is optimistic; the study used a fixed 0.5."""
    rows = []
    has_repeat = "repeat" in oof.columns
    for model, d in oof.groupby("model"):
        if has_repeat:
            per = [compute_metrics(g.y_true, g.p_pos, threshold) for _, g in d.groupby("repeat")]
            rows.append({
                "model": model,
                **{k: sum(p[k] for p in per) / len(per) for k in
                   ("accuracy", "sensitivity", "specificity", "precision", "f1", "tn", "fp", "fn", "tp")}
            })
        else:
            m = compute_metrics(d.y_true, d.p_pos, threshold)
            rows.append({"model": model, **m})
    order = MODEL_ORDER if set(oof["model"].unique()) <= set(MODEL_ORDER) else [m for m in PHASE2_MODEL_ORDER if m in oof["model"].unique()]
    out = pd.DataFrame(rows).set_index("model").reindex(order)
    return out
