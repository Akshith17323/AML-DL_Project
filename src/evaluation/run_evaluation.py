"""Run the full nested-CV comparison and write metrics, tables and figures.

    python -m src.evaluation.run_evaluation            # cross-validation + analysis
    python -m src.evaluation.run_evaluation --analyse  # re-analyse saved out-of-fold predictions only
"""
from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data.loader import get_xy, load_cleveland, load_config, resolve
from src.evaluation.cv import evaluate_model, fold_hash
from src.evaluation.metrics import (calibration_slope_intercept, check_probabilities, compute_metrics,
                                    reliability_table)
from src.evaluation.splits import outer_folds, save_folds
from src.models.pipelines import build_bn_pipeline, build_mlp_pipeline, build_nb_pipeline

MODELS = {"Bayesian Network": ("bayesian_network", build_bn_pipeline),
          "Naive Bayes": ("naive_bayes", build_nb_pipeline),
          "MLP": ("mlp", build_mlp_pipeline)}
COLORS = dict(zip(MODELS, sns.color_palette("colorblind", 3)))
METRIC_COLS = ["roc_auc", "pr_auc", "brier", "log_loss", "ece", "sensitivity", "specificity", "precision", "f1"]
BOOT_METRICS = ["roc_auc", "pr_auc", "brier", "log_loss", "sensitivity", "specificity"]


# ------------------------------------------------------------------ validation
def validate_oof(oof: pd.DataFrame, y: pd.Series, outer) -> dict:
    n_rep = int(oof["repeat"].nunique())
    checks = {}
    models = sorted(oof["model"].unique())
    for m in models:
        d = oof[oof["model"] == m]
        p = check_probabilities(d["p_pos"].to_numpy())
        assert np.allclose(d["p_neg"] + d["p_pos"], 1), f"{m}: p_neg + p_pos != 1"
        assert (d["y_true"].to_numpy() == y.iloc[d["patient"]].to_numpy()).all(), f"{m}: labels misaligned"
        for r, g in d.groupby("repeat"):
            assert sorted(g["patient"]) == list(range(len(y))), f"{m}: repeat {r} predictions missing or duplicated"
        checks[m] = {"n_predictions": len(d), "n_repeats": n_rep, "every_patient_once_per_repeat": True,
                     "probabilities_valid": True, "min_p": float(p.min()), "max_p": float(p.max())}
    ref = oof[oof["model"] == models[0]].sort_values(["outer_id", "patient"])[["outer_id", "patient"]].to_numpy()
    for m in models[1:]:
        cur = oof[oof["model"] == m].sort_values(["outer_id", "patient"])[["outer_id", "patient"]].to_numpy()
        assert np.array_equal(ref, cur), f"{m}: outer-fold assignment differs from {models[0]}"
    # folds in the saved predictions must be exactly the configured folds
    for f, (tr, va) in enumerate(outer):
        for m in models:
            got = oof[(oof["model"] == m) & (oof["outer_id"] == f)]["patient"].to_numpy()
            assert sorted(got) == sorted(va), f"{m}: outer fold {f} validation set differs from configured folds"
    checks["identical_folds_across_models"] = True
    return checks


# ------------------------------------------------------------------ metrics
def fold_level_metrics(oof: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    thr0, bins = cfg["evaluation"]["primary_threshold"], cfg["evaluation"]["calibration_bins"]
    rows = []
    for (m, f), d in oof.groupby(["model", "outer_id"]):
        for scheme, thr in (("fixed_0.5", thr0), ("train_youden", d["youden_threshold"].iloc[0])):
            rows.append({"model": m, "repeat": d["repeat"].iloc[0], "fold": d["fold"].iloc[0], "outer_id": f,
                         "threshold_scheme": scheme, **compute_metrics(d["y_true"], d["p_pos"], thr, bins)})
    return pd.DataFrame(rows)


def pooled_metrics(oof: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    thr0, bins = cfg["evaluation"]["primary_threshold"], cfg["evaluation"]["calibration_bins"]
    rows = []
    for (m, r), d in oof.groupby(["model", "repeat"]):
        slope, icpt = calibration_slope_intercept(d["y_true"], d["p_pos"])
        for scheme, thr in (("fixed_0.5", thr0), ("train_youden", d["youden_threshold"].to_numpy())):
            rows.append({"model": m, "repeat": r, "threshold_scheme": scheme,
                         **compute_metrics(d["y_true"], d["p_pos"], thr, bins),
                         "calibration_slope": slope, "calibration_intercept": icpt})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(["model", "threshold_scheme"])[cols]
    out = pd.concat({"mean": g.mean(), "sd": g.std(ddof=1)}, axis=1)
    out.columns = [f"{c}_{s}" for s, c in out.columns]
    return out.reset_index()


def bootstrap(oof: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level percentile bootstrap of repeat-averaged pooled OOF metrics (threshold 0.5).

    Resamples patients, keeps each model's OOF predictions for the resampled patients, so model
    differences are paired. Captures sampling variability of the 303 patients only."""
    B, level = cfg["evaluation"]["n_bootstrap"], cfg["evaluation"]["ci_level"]
    rng = np.random.default_rng(cfg["project"]["seed"])
    models = list(MODELS)
    n = oof["patient"].nunique()
    # arrays [model][repeat] -> (y, p) ordered by patient
    arr = {m: [d.sort_values("patient")[["y_true", "p_pos"]].to_numpy() for _, d in
               oof[oof["model"] == m].groupby("repeat")] for m in models}
    y = arr[models[0]][0][:, 0].astype(int)
    thr0 = cfg["evaluation"]["primary_threshold"]
    boot = {m: {k: np.empty(B) for k in BOOT_METRICS} for m in models}
    for b in range(B):
        while True:
            idx = rng.integers(0, n, n)
            if 0 < y[idx].sum() < n:
                break
        for m in models:
            vals = [compute_metrics(y[idx], rep[idx, 1], thr0) for rep in arr[m]]
            for k in BOOT_METRICS:
                boot[m][k][b] = np.mean([v[k] for v in vals])
    point = {m: {k: np.mean([compute_metrics(y, rep[:, 1], thr0)[k] for rep in arr[m]]) for k in BOOT_METRICS}
             for m in models}
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    ci = [{"model": m, "metric": k, "estimate": point[m][k], "ci_low": np.percentile(boot[m][k], lo_q),
           "ci_high": np.percentile(boot[m][k], hi_q)} for m in models for k in BOOT_METRICS]
    diffs = []
    for i, a in enumerate(models):
        for c in models[i + 1:]:
            for k in BOOT_METRICS:
                d = boot[a][k] - boot[c][k]
                diffs.append({"comparison": f"{a} - {c}", "metric": k, "estimate": point[a][k] - point[c][k],
                              "ci_low": np.percentile(d, lo_q), "ci_high": np.percentile(d, hi_q),
                              "ci_includes_zero": bool(np.percentile(d, lo_q) <= 0 <= np.percentile(d, hi_q))})
    return pd.DataFrame(ci), pd.DataFrame(diffs)


# ------------------------------------------------------------------ figures
def make_figures(oof: pd.DataFrame, pooled: pd.DataFrame, cfg: dict) -> list[str]:
    fig_dir = resolve(cfg["paths"]["figures"]); fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    saved = []
    auc = pooled[pooled["threshold_scheme"] == "fixed_0.5"].groupby("model")[["roc_auc", "pr_auc"]].mean()
    prevalence = oof[oof["model"] == "MLP"]["y_true"].mean()

    def curves(kind):
        fig, ax = plt.subplots(figsize=(5.4, 5))
        for m in MODELS:
            for r, d in oof[oof["model"] == m].groupby("repeat"):
                if kind == "roc":
                    x, yv, _ = roc_curve(d["y_true"], d["p_pos"])
                else:
                    yv, x, _ = precision_recall_curve(d["y_true"], d["p_pos"])
                ax.plot(x, yv, color=COLORS[m], alpha=0.22, lw=1, drawstyle="steps-post")
            ax.plot([], [], color=COLORS[m], lw=2,
                    label=f"{m} ({'AUC' if kind == 'roc' else 'AP'} {auc.loc[m, 'roc_auc' if kind == 'roc' else 'pr_auc']:.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
            ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
                   title="ROC, out-of-fold (thin: 5 repeats)")
        else:
            ax.axhline(prevalence, color="k", ls="--", lw=1, label=f"prevalence {prevalence:.2f}")
            ax.set(xlabel="Recall (sensitivity)", ylabel="Precision", ylim=(0, 1.02),
                   title="Precision-recall, out-of-fold (thin: 5 repeats)")
        ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=8)
        fig.tight_layout(); name = f"{kind}_curves.png"; fig.savefig(fig_dir / name, dpi=150); plt.close(fig); saved.append(name)

    curves("roc"); curves("pr")

    # confusion matrices: mean pooled counts over repeats, two threshold schemes
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    for i, scheme in enumerate(["fixed_0.5", "train_youden"]):
        for j, m in enumerate(MODELS):
            d = pooled[(pooled["model"] == m) & (pooled["threshold_scheme"] == scheme)][["tn", "fp", "fn", "tp"]].mean()
            cm = np.array([[d.tn, d.fp], [d.fn, d.tp]])
            sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=np.vectorize(lambda v: f"{v:.1f}")(cm), fmt="",
                        cmap="Blues", vmin=0, vmax=1, cbar=False, ax=axes[i, j],
                        xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
            axes[i, j].set_title(f"{m}\n{'threshold 0.5' if i == 0 else 'train-fold Youden threshold'}", fontsize=9)
    fig.suptitle("Confusion matrices (mean counts over 5 repeats; shading = row-normalised)", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "confusion_matrices.png", dpi=150); plt.close(fig); saved.append("confusion_matrices.png")

    # calibration
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.6, 7), gridspec_kw={"height_ratios": [3, 1.3]})
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for m in MODELS:
        d = oof[oof["model"] == m]
        mp, fr = reliability_table(d["y_true"], d["p_pos"], cfg["evaluation"]["calibration_bins"], "quantile")
        ece = pooled[(pooled.model == m) & (pooled.threshold_scheme == "fixed_0.5")]["ece"].mean()
        ax.plot(mp, fr, "o-", color=COLORS[m], label=f"{m} (ECE {ece:.3f})", ms=4)
        ax2.hist(d["p_pos"], bins=20, range=(0, 1), histtype="step", color=COLORS[m], lw=1.6)
    ax.set(xlabel="Mean predicted probability (decile bins)", ylabel="Observed fraction positive",
           title="Calibration, out-of-fold (all repeats pooled)", xlim=(0, 1), ylim=(0, 1)); ax.legend(fontsize=8)
    ax2.set(xlabel="Predicted probability", ylabel="Count", title="Distribution of predictions")
    fig.tight_layout(); fig.savefig(fig_dir / "calibration_curves.png", dpi=150); plt.close(fig); saved.append("calibration_curves.png")
    return saved


# ------------------------------------------------------------------ driver
def analyse(cfg: dict, X, y, outer, oof: pd.DataFrame, sel: pd.DataFrame) -> dict:
    out = resolve(cfg["paths"]["metrics"])
    checks = validate_oof(oof, y, outer)
    fold = fold_level_metrics(oof, cfg); fold.to_csv(out / "fold_metrics.csv", index=False)
    pooled = pooled_metrics(oof, cfg); pooled.to_csv(out / "pooled_metrics_by_repeat.csv", index=False)
    cols = METRIC_COLS + ["tn", "fp", "fn", "tp"]
    s_fold = summarise(fold, cols); s_fold.to_csv(out / "summary_fold_level.csv", index=False)
    s_pool = summarise(pooled, cols + ["calibration_slope", "calibration_intercept"])
    s_pool.to_csv(out / "summary_pooled_repeats.csv", index=False)
    ci, diffs = bootstrap(oof, cfg)
    ci.to_csv(out / "bootstrap_ci.csv", index=False); diffs.to_csv(out / "bootstrap_paired_differences.csv", index=False)
    figs = make_figures(oof, pooled, cfg)
    # report-ready compact table
    tab = resolve(cfg["paths"]["report_tables"]); tab.mkdir(parents=True, exist_ok=True)
    f05 = s_fold[s_fold.threshold_scheme == "fixed_0.5"].set_index("model")
    compact = pd.DataFrame({k: [f"{f05.loc[m, k + '_mean']:.3f} ± {f05.loc[m, k + '_sd']:.3f}" for m in MODELS]
                            for k in METRIC_COLS}, index=list(MODELS))
    compact.index.name = "model"; compact.to_csv(tab / "model_comparison_fold_level.csv")
    manifest = {"seed": cfg["project"]["seed"], "outer": {"folds": cfg["cv"]["outer_folds"], "repeats": cfg["cv"]["outer_repeats"],
                "n_outer_fits_per_model": len(outer), "fold_hash": fold_hash(outer)},
                "inner_folds": cfg["cv"]["inner_folds"], "tuning": cfg["tuning"],
                "primary_threshold": cfg["evaluation"]["primary_threshold"],
                "secondary_threshold": "Youden J on inner out-of-fold predictions of the training fold",
                "n_bootstrap": cfg["evaluation"]["n_bootstrap"], "checks": checks, "figures": figs,
                "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "pgmpy", "torch", "scipy")},
                "python": platform.python_version()}
    (out / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"fold": fold, "pooled": pooled, "s_fold": s_fold, "s_pool": s_pool, "ci": ci, "diffs": diffs, "checks": checks}


def main(analyse_only: bool = False) -> None:
    cfg = load_config()
    X, y = get_xy(load_cleveland(config=cfg))
    outer = outer_folds(y, cfg)
    save_folds(outer)
    out = resolve(cfg["paths"]["metrics"])
    if analyse_only:
        oof, sel = pd.read_csv(out / "oof_predictions.csv"), pd.read_csv(out / "selected_hyperparameters.csv")
    else:
        oofs, sels = [], []
        for name, (key, builder) in MODELS.items():
            o, s = evaluate_model(name, builder(cfg), cfg["tuning"][key], X, y, outer, cfg, cfg["cv"]["outer_folds"])
            oofs.append(o); sels.append(s)
        oof, sel = pd.concat(oofs, ignore_index=True), pd.concat(sels, ignore_index=True)
        oof.to_csv(out / "oof_predictions.csv", index=False); sel.to_csv(out / "selected_hyperparameters.csv", index=False)
    res = analyse(cfg, X, y, outer, oof, sel)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\nchecks:", json.dumps(res["checks"], indent=1))
    print("\nfold-level mean/sd (threshold 0.5):")
    print(res["s_fold"][res["s_fold"].threshold_scheme == "fixed_0.5"].filter(regex="model|roc|pr_auc|brier|sens|spec|prec|f1|ece|log").round(3).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--analyse", action="store_true")
    main(ap.parse_args().analyse)
