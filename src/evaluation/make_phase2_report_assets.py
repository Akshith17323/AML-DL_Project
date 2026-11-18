"""Generate IEEE result tables from the audited full Phase 2 artifacts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import PROJECT_ROOT


def _model_table(metrics: Path, protocol: str, target: Path) -> None:
    data = pd.read_csv(metrics / f"summary_{protocol}.csv")
    data = data[data.cohort == "ALL"]
    ci = pd.read_csv(metrics / f"bootstrap_ci_{protocol}.csv")
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Model & ROC AUC [95\\% CI] & PR AUC & Brier & Log loss \\\\", "\\midrule"]
    for row in data.itertuples():
        model = str(row.model).replace("-", "--")
        interval = ci[(ci.model == row.model) & (ci.metric == "roc_auc")].iloc[0]
        lines.append(f"{model} & {row.roc_auc:.3f} [{interval.ci_low:.3f}, {interval.ci_high:.3f}] & "
                     f"{row.pr_auc:.3f} & {row.brier:.3f} & {row.log_loss:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")


def _ablation_table(metrics: Path, target: Path) -> None:
    data = pd.read_csv(metrics / "ablations.csv")
    lines = ["\\begin{tabular}{llrr}", "\\toprule",
             "Suite & Variant & ROC AUC & Log loss \\\\", "\\midrule"]
    for row in data.itertuples():
        label = str(row.variant).replace("_", "\\_")
        lines.append(f"{row.suite} & {label} & {row.roc_auc_mean:.3f} & {row.log_loss_mean:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    target.write_text("\n".join(lines) + "\n")


def main(metrics: Path | None = None) -> list[Path]:
    metrics = metrics or PROJECT_ROOT / "results/phase2/main/metrics"
    required = [metrics / f for f in ("summary_pooled.csv", "summary_cross-cohort.csv",
                                       "bootstrap_ci_pooled.csv", "bootstrap_ci_cross-cohort.csv", "ablations.csv")]
    absent = [str(path) for path in required if not path.exists()]
    if absent:
        raise FileNotFoundError(f"Run both full evaluations, analysis, and ablations first: {absent}")
    tables = PROJECT_ROOT / "reports/phase2/tables"; tables.mkdir(parents=True, exist_ok=True)
    targets = [tables / "pooled_results.tex", tables / "cross_cohort_results.tex", tables / "ablations.tex"]
    _model_table(metrics, "pooled", targets[0])
    _model_table(metrics, "cross-cohort", targets[1])
    _ablation_table(metrics, targets[2])
    for target in targets:
        print(target)
    return targets


if __name__ == "__main__":
    main()
