"""Interpretation: out-of-fold feature attribution for models and a BN structure figure.

    python -m src.evaluation.interpret

Attribution is *descriptive of the fitted model*, not of the patients or of causal
mechanisms. For each outer fold of repeat 0 models are refitted on the training fold with the
hyper-parameters selected in that fold during evaluation, then the held-out fold is explained:
  * permutation importance: drop in ROC AUC when one raw predictor is shuffled among held-out rows;
  * SHAP (KernelExplainer, model output = P(disease)) with a training-fold background sample.
Both work on the 13 raw predictors (before one-hot encoding), through the full pipeline.
Refitted predictions are checked against the saved evaluation predictions.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.data.loader import FEATURE_COLUMNS, get_xy, load_cleveland, load_config, resolve
from src.evaluation.splits import outer_folds
from src.models.bayesian_network import EDGES, NODES, build_structure
from src.models.pipelines import build_gb_pipeline, build_mlp_pipeline, build_rf_pipeline

N_PERM_REPEATS = 30
N_BACKGROUND = 30
N_SHAP_SAMPLES = 500


def _fold_pipeline(cfg, sel_row):
    params = {k: sel_row[k] for k in sel_row.index if k.startswith("model__") and pd.notna(sel_row[k])}
    return clone(build_mlp_pipeline(cfg)).set_params(**params)


def permutation_importance_auc(pipe, Xva, yva, rng) -> pd.DataFrame:
    base = roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])
    rows = []
    for c in FEATURE_COLUMNS:
        drops = []
        for _ in range(N_PERM_REPEATS):
            Xp = Xva.copy(); Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops.append(base - roc_auc_score(yva, pipe.predict_proba(Xp)[:, 1]))
        rows.append({"feature": c, "auc_drop": float(np.mean(drops))})
    return pd.DataFrame(rows)


def shap_values_fold(pipe, Xtr, Xva, seed) -> np.ndarray:
    f = lambda a: pipe.predict_proba(pd.DataFrame(a, columns=FEATURE_COLUMNS))[:, 1]
    bg = Xtr.sample(N_BACKGROUND, random_state=seed).to_numpy(dtype=float)
    np.random.seed(seed)  # KernelExplainer samples coalitions from NumPy's global RNG
    expl = shap.KernelExplainer(f, bg)
    return np.asarray(expl.shap_values(Xva.to_numpy(dtype=float), nsamples=N_SHAP_SAMPLES, silent=True))


def run_attribution(cfg: dict) -> dict:
    X, y = get_xy(load_cleveland(config=cfg))
    outer = outer_folds(y, cfg)[:cfg["cv"]["outer_folds"]]  # repeat 0
    sel = pd.read_csv(resolve(cfg["paths"]["metrics"]) / "selected_hyperparameters.csv")
    sel = sel[sel.model == "MLP"].set_index("outer_id")
    oof = pd.read_csv(resolve(cfg["paths"]["metrics"]) / "oof_predictions.csv")
    oof = oof[(oof.model == "MLP") & (oof.repeat == 0)].set_index("patient")
    rng = np.random.default_rng(cfg["project"]["seed"])
    perm, shap_all, max_diff = [], [], 0.0
    for f, (tr, va) in enumerate(outer):
        pipe = _fold_pipeline(cfg, sel.loc[f]).fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        max_diff = max(max_diff, float(np.abs(p - oof.loc[va, "p_pos"].to_numpy()).max()))
        d = permutation_importance_auc(pipe, X.iloc[va], y.iloc[va], rng); d["fold"] = f; perm.append(d)
        sv = shap_values_fold(pipe, X.iloc[tr], X.iloc[va], cfg["project"]["seed"] + f)
        shap_all.append(pd.DataFrame(sv, columns=FEATURE_COLUMNS, index=va))
        print(f"fold {f}: refit-vs-saved max |dp| = {max_diff:.2e}", flush=True)
    perm = pd.concat(perm); sv = pd.concat(shap_all).sort_index()
    return {"perm": perm, "shap": sv, "X": X.loc[sv.index], "max_refit_diff": max_diff}


def plot_attribution(res: dict, cfg: dict) -> list[str]:
    fig_dir = resolve(cfg["paths"]["figures"]); sns.set_theme(style="whitegrid", context="notebook")
    shap_abs = res["shap"].abs().mean().rename("mean_abs_shap")
    perm = res["perm"].groupby("feature")["auc_drop"].agg(["mean", "std"]).rename(columns={"mean": "perm_auc_drop", "std": "perm_auc_drop_sd"})
    tab = pd.concat([shap_abs, perm], axis=1).sort_values("mean_abs_shap", ascending=False)
    tab.to_csv(resolve(cfg["paths"]["metrics"]) / "mlp_attribution.csv", index_label="feature")
    res["shap"].to_csv(resolve(cfg["paths"]["metrics"]) / "mlp_shap_values_oof.csv", index_label="patient")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    order = tab.index
    ax[0].barh(order, tab["mean_abs_shap"], color=sns.color_palette("colorblind")[0]); ax[0].invert_yaxis()
    ax[0].set(xlabel="mean |SHAP| (change in P(disease))", title="SHAP, held-out rows")
    ax[1].barh(order, tab["perm_auc_drop"], xerr=tab["perm_auc_drop_sd"], color=sns.color_palette("colorblind")[2])
    ax[1].set(xlabel="ROC AUC drop when shuffled (mean ± SD over 5 folds)", title="Permutation importance")
    fig.suptitle("MLP attribution (repeat-0 outer folds): describes the model, not causes", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "mlp_attribution_bars.png", dpi=150); plt.close(fig)
    plt.figure(figsize=(7, 5))
    shap.summary_plot(res["shap"].to_numpy(), res["X"], feature_names=FEATURE_COLUMNS, show=False, plot_size=None)
    plt.title("SHAP values (held-out rows)", fontsize=10); plt.tight_layout()
    plt.savefig(fig_dir / "mlp_shap_beeswarm.png", dpi=150); plt.close()
    return ["mlp_attribution_bars.png", "mlp_shap_beeswarm.png"]


def ensemble_permutation_importance(model_name: str, build_fn, cfg: dict) -> pd.DataFrame:
    """Compute out-of-fold permutation importance for an ensemble model across repeat-0 outer folds."""
    X, y = get_xy(load_cleveland(config=cfg))
    outer = outer_folds(y, cfg)[:cfg["cv"]["outer_folds"]]  # repeat 0
    rng = np.random.default_rng(cfg["project"]["seed"])
    perm = []
    pipe = build_fn(cfg)
    for f, (tr, va) in enumerate(outer):
        fitted = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        d = permutation_importance_auc(fitted, X.iloc[va], y.iloc[va], rng)
        d["fold"] = f
        d["model"] = model_name
        perm.append(d)
    df = pd.concat(perm, ignore_index=True)
    summary = df.groupby(["model", "feature"])["auc_drop"].agg(["mean", "std"]).reset_index()
    summary = summary.rename(columns={"mean": "perm_auc_drop", "std": "perm_auc_drop_sd"})
    return summary.sort_values("perm_auc_drop", ascending=False)


def run_ensemble_attribution(cfg: dict) -> dict[str, pd.DataFrame]:
    """Run permutation importance attribution for Random Forest and Gradient Boosting."""
    rf_perm = ensemble_permutation_importance("Random Forest", build_rf_pipeline, cfg)
    gb_perm = ensemble_permutation_importance("Gradient Boosting", build_gb_pipeline, cfg)
    metrics_dir = resolve(cfg["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rf_perm.to_csv(metrics_dir / "rf_permutation_importance.csv", index=False)
    gb_perm.to_csv(metrics_dir / "gb_permutation_importance.csv", index=False)
    return {"Random Forest": rf_perm, "Gradient Boosting": gb_perm}


def plot_ensemble_importance(res: dict[str, pd.DataFrame], cfg: dict) -> list[str]:
    """Generate bar plots comparing feature importance for RF and GB."""
    fig_dir = resolve(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (model_name, tab), color_idx in zip(axes, res.items(), [0, 2]):
        order = tab.sort_values("perm_auc_drop", ascending=False)
        ax.barh(order["feature"], order["perm_auc_drop"], xerr=order["perm_auc_drop_sd"],
                color=sns.color_palette("colorblind")[color_idx])
        ax.set(xlabel="ROC AUC drop when shuffled (mean ± SD)", title=f"{model_name} Permutation Importance")
        ax.invert_yaxis()
    fig.suptitle("Ensemble Models Feature Importance (repeat-0 outer folds)", fontsize=11)
    fig.tight_layout()
    fig_path = fig_dir / "ensemble_permutation_importance.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return ["ensemble_permutation_importance.png"]


def plot_bn_structure(cfg: dict) -> str:
    g = nx.DiGraph(); g.add_nodes_from(NODES)
    kinds = {}
    for a, b, _, k in EDGES:
        g.add_edge(a, b); kinds[(a, b)] = k
    for layer, nodes in enumerate(nx.topological_generations(g)):
        for n in nodes:
            g.nodes[n]["layer"] = layer
    pos = nx.multipartite_layout(g, subset_key="layer", align="vertical", scale=1.0)
    pal = dict(zip(["domain", "weak", "modelling"], sns.color_palette("colorblind", 3)))
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(g, pos, node_color=["#f2c14e" if n == "target" else "#dce6f2" for n in g], node_size=1900,
                           edgecolors="k", ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=9, ax=ax)
    for k, c in pal.items():
        es = [e for e in g.edges if kinds[e] == k]
        nx.draw_networkx_edges(g, pos, edgelist=es, edge_color=[c], arrows=True, arrowsize=16, node_size=1900,
                               connectionstyle="arc3,rad=0.12", width=1.6, ax=ax)
        ax.plot([], [], color=c, label=f"{k} assumption")
    ax.legend(loc="lower left", fontsize=8); ax.set_axis_off()
    ax.set_title("Assumed Bayesian-network structure (domain-informed; not learned, not a causal claim)", fontsize=10)
    fig.tight_layout(); path = resolve(cfg["paths"]["figures"]) / "bn_structure.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    assert set(build_structure().edges()) == set(g.edges())
    return path.name


def main() -> None:
    cfg = load_config()
    print("Running ensemble attribution (Random Forest & Gradient Boosting)...", flush=True)
    ens_res = run_ensemble_attribution(cfg)
    ens_figs = plot_ensemble_importance(ens_res, cfg)
    print("Ensemble attribution completed:", ens_figs)
    try:
        res = run_attribution(cfg)
        figs = plot_attribution(res, cfg) + [plot_bn_structure(cfg)]
        summary = {"max_abs_diff_refit_vs_saved_predictions": res["max_refit_diff"],
                   "perm_repeats": N_PERM_REPEATS, "shap_background": N_BACKGROUND, "shap_nsamples": N_SHAP_SAMPLES,
                   "shap_version": shap.__version__, "figures": figs + ens_figs}
        (resolve(cfg["paths"]["metrics"]) / "mlp_attribution_manifest.json").write_text(json.dumps(summary, indent=2))
        print(pd.read_csv(resolve(cfg["paths"]["metrics"]) / "mlp_attribution.csv").round(4).to_string(index=False))
        print(summary)
    except Exception as e:
        print(f"Skipping MLP attribution (dependencies/saved predictions not yet generated): {e}")


if __name__ == "__main__":
    main()
