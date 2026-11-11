"""Multi-cohort Phase 2 data audit and compact comparison figures."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import (BINARY_TARGET, FEATURE_COLUMNS, PHASE2_CONFIG_PATH,
                             load_all_cohorts, load_config, resolve, verify_all_cohorts)


def run(config: dict | None = None) -> dict:
    cfg = config or load_config(PHASE2_CONFIG_PATH)
    df = load_all_cohorts(config=cfg)
    metrics = resolve(cfg["paths"]["metrics"]); metrics.mkdir(parents=True, exist_ok=True)
    figures = resolve(cfg["paths"]["figures"]); figures.mkdir(parents=True, exist_ok=True)

    prevalence = (df.groupby("cohort")[BINARY_TARGET].agg(patients="size", positive="sum", positive_rate="mean")
                    .reset_index())
    missing = (df.groupby("cohort")[FEATURE_COLUMNS].apply(lambda x: x.isna().sum())
                 .reset_index().melt("cohort", var_name="feature", value_name="missing"))
    prevalence.to_csv(metrics / "cohort_prevalence.csv", index=False)
    missing.to_csv(metrics / "cohort_missingness.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=prevalence, x="cohort", y="positive_rate", ax=axes[0], color="#4c72b0")
    axes[0].set(title="Disease prevalence by cohort", ylim=(0, 1), ylabel="Positive fraction")
    heat = missing.pivot(index="feature", columns="cohort", values="missing")
    sns.heatmap(heat, annot=True, fmt="g", cmap="Reds", ax=axes[1])
    axes[1].set_title("Missing predictor cells")
    fig.tight_layout(); fig.savefig(figures / "cohort_data_audit.png", dpi=150); plt.close(fig)

    report = verify_all_cohorts(df)
    report["figures"] = ["cohort_data_audit.png"]
    (metrics / "eda_summary.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
