"""Interpretation: out-of-fold permutation feature importance and a BN structure figure.

    python -m src.evaluation.interpret

Permutation importance = drop in ROC AUC when one raw predictor is shuffled among the held-out
patients (the link between that predictor and the outcome is broken, everything else is kept).
For each outer fold of repeat 0, Logistic Regression, Random Forest and Gradient Boosting are
refitted on the training fold with the hyper-parameters selected in that fold during evaluation,
and the held-out fold is scored. Refitted predictions are checked against the saved evaluation
predictions. Importance describes the fitted model, not the patients or causal mechanisms.
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
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.data.loader import FEATURE_COLUMNS, get_xy, load_cleveland, load_config, resolve
from src.evaluation.splits import outer_folds
from src.models.bayesian_network import EDGES, NODES, build_structure
from src.models.pipelines import build_gb_pipeline, build_lr_pipeline, build_rf_pipeline, params_from_selection

N_PERM_REPEATS = 30
IMPORTANCE_MODELS = {"Logistic Regression": ("logistic_regression", build_lr_pipeline),
                     "Random Forest": ("random_forest", build_rf_pipeline),
                     "Gradient Boosting": ("gradient_boosting", build_gb_pipeline)}


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


def run_importance(cfg: dict) -> dict:
    X, y = get_xy(load_cleveland(config=cfg))
    outer = outer_folds(y, cfg)[:cfg["cv"]["outer_folds"]]  # repeat 0
    m_dir = resolve(cfg["paths"]["metrics"])
    sel_all = pd.read_csv(m_dir / "selected_hyperparameters.csv")
    oof_all = pd.read_csv(m_dir / "oof_predictions.csv")
    rng = np.random.default_rng(cfg["project"]["seed"])
    rows, max_diff = [], {}
    for name, (key, builder) in IMPORTANCE_MODELS.items():
        sel = sel_all[sel_all.model == name].set_index("outer_id")
        oof = oof_all[(oof_all.model == name) & (oof_all.repeat == 0)].set_index("patient")
        max_diff[name] = 0.0
        for f, (tr, va) in enumerate(outer):
            pipe = clone(builder(cfg)).set_params(**params_from_selection(sel.loc[f], cfg["tuning"][key])).fit(X.iloc[tr], y.iloc[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]
            max_diff[name] = max(max_diff[name], float(np.abs(p - oof.loc[va, "p_pos"].to_numpy()).max()))
            d = permutation_importance_auc(pipe, X.iloc[va], y.iloc[va], rng); d["fold"] = f; d["model"] = name
            rows.append(d)
        print(f"{name}: refit-vs-saved max |dp| = {max_diff[name]:.2e}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    tab = (df.groupby(["model", "feature"])["auc_drop"].agg(["mean", "std"]).reset_index()
             .rename(columns={"mean": "auc_drop_mean", "std": "auc_drop_sd"}))
    tab.to_csv(m_dir / "feature_importance.csv", index=False)
    return {"table": tab, "max_refit_diff": max_diff}


def plot_importance(tab: pd.DataFrame, cfg: dict) -> str:
    fig_dir = resolve(cfg["paths"]["figures"]); sns.set_theme(style="whitegrid", context="notebook")
    order = (tab.groupby("feature")["auc_drop_mean"].mean().sort_values(ascending=False).index)
    fig, axes = plt.subplots(1, len(IMPORTANCE_MODELS), figsize=(13, 4.8), sharey=True)
    colors = dict(zip(["Logistic Regression", "Random Forest", "Gradient Boosting"],
                      [sns.color_palette("colorblind")[i] for i in (2, 4, 5)]))
    for ax, name in zip(axes, IMPORTANCE_MODELS):
        t = tab[tab.model == name].set_index("feature").loc[order]
        ax.barh(t.index, t["auc_drop_mean"], xerr=t["auc_drop_sd"], color=colors[name])
        ax.set(xlabel="ROC AUC drop when shuffled", title=name)
        ax.axvline(0, color="k", lw=0.8)
    axes[0].invert_yaxis()
    fig.suptitle("Permutation feature importance (held-out patients, repeat-0 outer folds; bars: mean ± SD over 5 folds)", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "feature_importance.png", dpi=150); plt.close(fig)
    return "feature_importance.png"


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
    res = run_importance(cfg)
    figs = [plot_importance(res["table"], cfg), plot_bn_structure(cfg)]
    summary = {"max_abs_diff_refit_vs_saved_predictions": res["max_refit_diff"], "perm_repeats": N_PERM_REPEATS,
               "figures": figs}
    (resolve(cfg["paths"]["metrics"]) / "feature_importance_manifest.json").write_text(json.dumps(summary, indent=2))
    piv = res["table"].pivot(index="feature", columns="model", values="auc_drop_mean")
    print(piv.loc[piv.mean(axis=1).sort_values(ascending=False).index].round(4).to_string())
    print(summary)


if __name__ == "__main__":
    main()
