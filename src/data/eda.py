"""Data-quality audit and exploratory plots for the Cleveland data.

Descriptive only: no models are fitted and no significance is claimed.
Run from the project root:  python -m src.data.eda
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.loader import (BINARY_TARGET, FEATURE_COLUMNS, ORIGINAL_TARGET,
                             load_cleveland, load_config, resolve)

CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Documented value domains (heart-disease.names). Continuous ranges are broad
# physiological plausibility bounds used only to flag, not to delete, records.
DOMAINS = {
    "sex": {0, 1}, "cp": {1, 2, 3, 4}, "fbs": {0, 1}, "restecg": {0, 1, 2},
    "exang": {0, 1}, "slope": {1, 2, 3}, "ca": {0, 1, 2, 3}, "thal": {3, 6, 7},
}
PLAUSIBLE = {"age": (18, 100), "trestbps": (70, 250), "chol": (100, 600),
             "thalach": (60, 220), "oldpeak": (0, 10)}

DESCRIPTIONS = {
    "age": "age in years", "sex": "1 = male, 0 = female",
    "cp": "chest pain type: 1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic",
    "trestbps": "resting blood pressure on admission (mm Hg)",
    "chol": "serum cholesterol (mg/dl)", "fbs": "fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "resting ECG: 0 normal, 1 ST-T abnormality, 2 probable/definite LV hypertrophy",
    "thalach": "maximum heart rate achieved", "exang": "exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "slope of peak exercise ST segment: 1 up, 2 flat, 3 down",
    "ca": "number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "3 normal, 6 fixed defect, 7 reversible defect",
    "num": "angiographic disease status: 0 = <50% diameter narrowing, 1-4 = >50% narrowing (increasing severity)",
}


def audit(df: pd.DataFrame) -> dict:
    out = {"n_rows": len(df), "duplicates": int(df.duplicated().sum()),
           "missing": {c: int(n) for c, n in df.isna().sum().items() if n},
           "rows_with_missing": df.index[df.isna().any(axis=1)].tolist(),
           "dtypes": {c: str(t) for c, t in df.dtypes.items()},
           "unique_counts": {c: int(df[c].nunique()) for c in df.columns},
           "invalid_categorical": {}, "implausible_continuous": {}, "non_integer_categorical": {}}
    for c, dom in DOMAINS.items():
        vals = set(df[c].dropna().unique())
        bad = sorted(vals - dom)
        if bad:
            out["invalid_categorical"][c] = bad
        if not all(float(v).is_integer() for v in vals):
            out["non_integer_categorical"][c] = True
    for c, (lo, hi) in PLAUSIBLE.items():
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        if n:
            out["implausible_continuous"][c] = n
    # chol == 0 / trestbps == 0 are known placeholder-zero artefacts in other sites
    out["zero_chol_or_trestbps"] = int(((df["chol"] == 0) | (df["trestbps"] == 0)).sum())
    out["ranges"] = {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                         "mean": round(float(df[c].mean()), 3), "median": float(df[c].median()),
                         "std": round(float(df[c].std()), 3)} for c in CONTINUOUS}
    out["value_counts"] = {c: {str(k): int(v) for k, v in df[c].value_counts(dropna=False).sort_index().items()}
                           for c in CATEGORICAL}
    return out


def target_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    orig = df[ORIGINAL_TARGET].value_counts().sort_index()
    binar = df[BINARY_TARGET].value_counts().sort_index()
    return {"original": {int(k): int(v) for k, v in orig.items()},
            "original_pct": {int(k): round(100 * v / n, 1) for k, v in orig.items()},
            "binary": {int(k): int(v) for k, v in binar.items()},
            "binary_pct": {int(k): round(100 * v / n, 1) for k, v in binar.items()},
            "minority_to_majority_ratio": round(float(binar.min() / binar.max()), 3)}


def association_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive Spearman correlations (features x features, feature-vs-target). No p-values."""
    corr = df[FEATURE_COLUMNS].corr(method="spearman")
    with_target = df[FEATURE_COLUMNS].apply(lambda s: s.corr(df[BINARY_TARGET], method="spearman"))
    return corr, with_target.rename("spearman_with_target").to_frame()


def redundant_pairs(corr: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= threshold:
                pairs.append({"a": a, "b": b, "spearman": round(float(r), 3)})
    return sorted(pairs, key=lambda d: -abs(d["spearman"]))


def positive_rate_by_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CATEGORICAL:
        g = df.groupby(c)[BINARY_TARGET].agg(["count", "mean"])
        for lvl, r in g.iterrows():
            rows.append({"feature": c, "level": lvl, "n": int(r["count"]),
                         "positive_rate": round(float(r["mean"]), 3)})
    return pd.DataFrame(rows)


def make_figures(df: pd.DataFrame, fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    saved: list[str] = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=150)
        plt.close(fig)
        saved.append(name)

    # target distributions
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    sns.countplot(data=df, x=ORIGINAL_TARGET, color="#4c72b0", ax=ax[0]); ax[0].set_title("Original target (num)")
    sns.countplot(data=df, x=BINARY_TARGET, color="#55a868", ax=ax[1]); ax[1].set_title("Binary target")
    for a in ax:
        for p in a.patches:
            a.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    save(fig, "eda_target_distribution.png")

    # histograms
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for a, c in zip(axes.flat, CONTINUOUS):
        sns.histplot(df[c], bins=25, ax=a, color="#4c72b0"); a.set_title(c)
    axes.flat[-1].axis("off")
    save(fig, "eda_histograms.png")

    # box plots by binary target
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.8))
    for a, c in zip(axes, CONTINUOUS):
        sns.boxplot(data=df, x=BINARY_TARGET, y=c, ax=a, hue=BINARY_TARGET, palette="deep", legend=False)
        a.set_title(f"{c} by target")
    save(fig, "eda_boxplots_by_target.png")

    # categorical counts
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for a, c in zip(axes.flat, CATEGORICAL):
        sns.countplot(data=df, x=c, color="#4c72b0", ax=a); a.set_title(c)
    save(fig, "eda_categorical_counts.png")

    # categorical vs target (positive rate)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for a, c in zip(axes.flat, CATEGORICAL):
        sns.barplot(data=df, x=c, y=BINARY_TARGET, errorbar=None, color="#c44e52", ax=a)
        a.set_title(f"positive rate by {c}"); a.set_ylim(0, 1)
    save(fig, "eda_positive_rate_by_category.png")

    # continuous vs target (KDE)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    for a, c in zip(axes, CONTINUOUS):
        sns.kdeplot(data=df, x=c, hue=BINARY_TARGET, common_norm=False, fill=True, palette="deep", ax=a)
        a.set_title(c)
    save(fig, "eda_continuous_by_target_kde.png")

    # correlation heatmap
    corr, _ = association_tables(df)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman correlation between predictors (descriptive)")
    save(fig, "eda_correlation_heatmap.png")

    # missingness
    fig, ax = plt.subplots(figsize=(8, 2.6))
    sns.heatmap(df[FEATURE_COLUMNS].isna().T, cbar=False, cmap="Greys", ax=ax, xticklabels=False)
    ax.set_title("Missing values (black = missing)")
    save(fig, "eda_missingness.png")
    return saved


def run(config: dict | None = None) -> dict:
    cfg = config or load_config()
    df = load_cleveland(config=cfg)
    fig_dir = resolve(cfg["paths"]["figures"])
    res_dir = resolve("results")
    corr, with_target = association_tables(df)
    summary = {
        "audit": audit(df),
        "target": target_summary(df),
        "redundant_pairs_abs_spearman_ge_0.5": redundant_pairs(corr),
        "spearman_with_binary_target": {k: round(float(v), 3) for k, v in with_target["spearman_with_target"].items()},
        "variable_descriptions": DESCRIPTIONS,
    }
    (res_dir / "eda_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    positive_rate_by_level(df).to_csv(res_dir / "eda_positive_rate_by_level.csv", index=False)
    corr.round(3).to_csv(res_dir / "eda_predictor_spearman.csv")
    summary["figures"] = make_figures(df, fig_dir)
    return summary


if __name__ == "__main__":
    s = run()
    print(json.dumps({k: v for k, v in s.items() if k != "variable_descriptions"}, indent=1, default=str))
