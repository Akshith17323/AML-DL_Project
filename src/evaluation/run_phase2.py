"""Run all-cohort Phase 2 nested search and evaluation.

Examples:
    python -m src.evaluation.run_phase2 --pilot
    python -m src.evaluation.run_phase2 --protocol pooled
    python -m src.evaluation.run_phase2 --protocol cross-cohort
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from src.data.loader import (PHASE2_CONFIG_PATH, cohort_file_hashes, get_xy,
                             load_all_cohorts, load_config, resolve)
from src.evaluation.cv import fold_hash
from src.evaluation.metrics import compute_metrics, reliability_table
from src.evaluation.phase2_search import make_search
from src.models.phase2 import phase2_models


def pooled_splits(y, cfg: dict):
    cv = StratifiedKFold(cfg["cv"]["outer_folds"], shuffle=True, random_state=cfg["project"]["seed"])
    return list(cv.split(np.zeros(len(y)), y))


def cross_cohort_splits(X, y, groups):
    return list(LeaveOneGroupOut().split(X, y, groups))


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_model(name: str, model_key: str, estimator, X, y, cohorts, outer,
                   cfg: dict, *, n_trials: int | None = None, inner_folds: int | None = None):
    """Outer evaluation loop; OptunaSearchCV owns all hyperparameter candidate selection."""
    predictions, selections, trials = [], [], []
    for outer_id, (train, test) in enumerate(outer):
        started = time.perf_counter()
        search = make_search(estimator, model_key, cfg, seed=cfg["project"]["seed"] + outer_id,
                             n_trials=n_trials, inner_folds=inner_folds)
        search.fit(X.iloc[train], y.iloc[train])
        probability = search.predict_proba(X.iloc[test])
        if probability.shape != (len(test), 2) or not np.allclose(probability.sum(axis=1), 1):
            raise RuntimeError(f"{name}: invalid probability output")
        predictions.append(pd.DataFrame({
            "model": name, "outer_id": outer_id, "patient": test,
            "cohort": cohorts.iloc[test].to_numpy(), "y_true": y.iloc[test].to_numpy(),
            "p_neg": probability[:, 0], "p_pos": probability[:, 1],
        }))
        selections.append({
            "model": name, "outer_id": outer_id,
            "best_inner_neg_log_loss": float(search.best_score_),
            "best_params": json.dumps({k: _plain(v) for k, v in search.best_params_.items()}, sort_keys=True),
            "runtime_seconds": time.perf_counter() - started,
        })
        trial_frame = search.study_.trials_dataframe().copy()
        trial_frame.insert(0, "outer_id", outer_id); trial_frame.insert(0, "model", name)
        trials.append(trial_frame)
        print(f"[{name}] outer {outer_id + 1}/{len(outer)} score={search.best_score_:.4f}", flush=True)
    return (pd.concat(predictions, ignore_index=True), pd.DataFrame(selections),
            pd.concat(trials, ignore_index=True))


def summarise(oof: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    threshold = cfg["evaluation"]["primary_threshold"]
    for model, data in oof.groupby("model", sort=False):
        rows.append({"model": model, "cohort": "ALL", **compute_metrics(data.y_true, data.p_pos, threshold)})
        for cohort, subset in data.groupby("cohort", sort=False):
            if subset.y_true.nunique() == 2:
                rows.append({"model": model, "cohort": cohort,
                             **compute_metrics(subset.y_true, subset.p_pos, threshold)})
    return pd.DataFrame(rows)


def bootstrap_intervals(oof: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Paired patient-level percentile intervals for pooled OOF predictions."""
    models = list(oof.model.drop_duplicates())
    reference = oof[oof.model == models[0]].sort_values("patient")
    patients = reference.patient.to_numpy(); y = reference.y_true.to_numpy(dtype=int)
    probabilities = {}
    for model in models:
        data = oof[oof.model == model].sort_values("patient")
        if not np.array_equal(data.patient.to_numpy(), patients):
            raise ValueError("Models do not have aligned out-of-fold patients")
        probabilities[model] = data.p_pos.to_numpy()
    rng = np.random.default_rng(cfg["project"]["seed"])
    metrics = ["roc_auc", "pr_auc", "brier", "log_loss", "accuracy", "sensitivity", "specificity"]
    samples = {model: {metric: [] for metric in metrics} for model in models}
    for _ in range(cfg["evaluation"]["n_bootstrap"]):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size != 2:
            continue
        for model in models:
            values = compute_metrics(y[index], probabilities[model][index],
                                     cfg["evaluation"]["primary_threshold"])
            for metric in metrics:
                samples[model][metric].append(values[metric])
    alpha = (1 - cfg["evaluation"]["ci_level"]) / 2
    rows = []
    for model in models:
        point = compute_metrics(y, probabilities[model], cfg["evaluation"]["primary_threshold"])
        for metric in metrics:
            values = np.asarray(samples[model][metric])
            rows.append({"model": model, "metric": metric, "estimate": point[metric],
                         "ci_low": np.quantile(values, alpha), "ci_high": np.quantile(values, 1 - alpha)})
    return pd.DataFrame(rows)


def make_figures(oof: pd.DataFrame, cfg: dict, protocol: str) -> list[str]:
    figures = resolve(cfg["paths"]["figures"]); figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    colors = dict(zip(oof.model.drop_duplicates(), sns.color_palette("colorblind", oof.model.nunique())))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for model, data in oof.groupby("model", sort=False):
        fpr, tpr, _ = roc_curve(data.y_true, data.p_pos)
        axes[0].plot(fpr, tpr, label=model, color=colors[model])
        predicted, observed = reliability_table(data.y_true, data.p_pos,
                                                cfg["evaluation"]["calibration_bins"], "quantile")
        axes[1].plot(predicted, observed, "o-", label=model, color=colors[model], ms=3)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1); axes[0].set(xlabel="False-positive rate", ylabel="Sensitivity", title="ROC")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1); axes[1].set(xlabel="Predicted probability", ylabel="Observed fraction", title="Calibration")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.suptitle(f"Phase 2 out-of-fold results: {protocol}")
    fig.tight_layout(); name = f"curves_{protocol}.png"; fig.savefig(figures / name, dpi=150); plt.close(fig)
    return [name]


def analyse_saved(protocol="pooled", pilot=False) -> dict:
    cfg = load_config(PHASE2_CONFIG_PATH)
    if pilot:
        cfg = copy.deepcopy(cfg)
        cfg["paths"]["metrics"] = "results/phase2/pilot/metrics"
        cfg["paths"]["figures"] = "results/phase2/pilot/figures"
    out = resolve(cfg["paths"]["metrics"])
    oof = pd.read_csv(out / f"oof_{protocol}.csv")
    summary, intervals = summarise(oof, cfg), bootstrap_intervals(oof, cfg)
    summary.to_csv(out / f"summary_{protocol}.csv", index=False)
    intervals.to_csv(out / f"bootstrap_ci_{protocol}.csv", index=False)
    figures = make_figures(oof, cfg, protocol)
    manifest_path = out / f"manifest_{protocol}.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.update({"figures": figures, "bootstrap_samples": cfg["evaluation"]["n_bootstrap"]})
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"summary": summary, "intervals": intervals, "figures": figures}


def run(protocol="pooled", pilot=False, model_names: list[str] | None = None) -> dict:
    cfg = load_config(PHASE2_CONFIG_PATH)
    if pilot:
        cfg = copy.deepcopy(cfg)
        cfg["cv"].update({"outer_folds": 3, "inner_folds": 2, "stacking_folds": 2})
        cfg["models"]["mlp"].update({"max_epochs": 12, "patience": 3})
        cfg["search"]["n_trials"] = 1
        cfg["paths"]["root"] = "results/phase2/pilot"
        cfg["paths"]["metrics"] = "results/phase2/pilot/metrics"
        cfg["paths"]["figures"] = "results/phase2/pilot/figures"

    df = load_all_cohorts(config=cfg)
    X, y = get_xy(df); cohorts = df["cohort"]
    outer = pooled_splits(y, cfg) if protocol == "pooled" else cross_cohort_splits(X, y, cohorts)
    if pilot:
        outer = outer[:1]
    registry = phase2_models(cfg)
    if model_names:
        unknown = sorted(set(model_names) - set(registry))
        if unknown:
            raise ValueError(f"Unknown models: {unknown}")
        registry = {name: registry[name] for name in model_names}

    oofs, selections, trials = [], [], []
    for name, (model_key, estimator) in registry.items():
        oof, selected, history = evaluate_model(
            name, model_key, estimator, X, y, cohorts, outer, cfg,
            n_trials=cfg["search"]["n_trials"], inner_folds=cfg["cv"]["inner_folds"])
        oofs.append(oof); selections.append(selected); trials.append(history)
    oof = pd.concat(oofs, ignore_index=True)
    selected = pd.concat(selections, ignore_index=True)
    trial_history = pd.concat(trials, ignore_index=True)
    summary = summarise(oof, cfg)

    out = resolve(cfg["paths"]["metrics"]); out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out / f"oof_{protocol}.csv", index=False)
    selected.to_csv(out / f"selected_hyperparameters_{protocol}.csv", index=False)
    trial_history.to_csv(out / f"optuna_trials_{protocol}.csv", index=False)
    summary.to_csv(out / f"summary_{protocol}.csv", index=False)
    intervals = bootstrap_intervals(oof, cfg)
    intervals.to_csv(out / f"bootstrap_ci_{protocol}.csv", index=False)
    figures = make_figures(oof, cfg, protocol)
    manifest = {
        "pilot": pilot, "protocol": protocol, "n_patients": len(df),
        "cohort_counts": df["cohort"].value_counts().to_dict(),
        "cohort_file_sha256": cohort_file_hashes(config=cfg),
        "models": list(registry), "outer_folds_run": len(outer),
        "outer_fold_hash": fold_hash(outer), "inner_folds": cfg["cv"]["inner_folds"],
        "search": cfg["search"], "seed": cfg["project"]["seed"],
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "torch", "optuna")},
        "figures": figures, "python": platform.python_version(),
    }
    (out / f"manifest_{protocol}.json").write_text(json.dumps(manifest, indent=2))
    print(summary[summary.cohort == "ALL"].round(4).to_string(index=False))
    return {"oof": oof, "selected": selected, "trials": trial_history,
            "summary": summary, "intervals": intervals, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["pooled", "cross-cohort"], default="pooled")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--analyse", action="store_true", help="analyse existing predictions without retraining")
    parser.add_argument("--models", help="comma-separated display names; default runs all eight")
    args = parser.parse_args()
    if args.analyse:
        analyse_saved(args.protocol, args.pilot)
    else:
        run(args.protocol, args.pilot, args.models.split(",") if args.models else None)


if __name__ == "__main__":
    main()
