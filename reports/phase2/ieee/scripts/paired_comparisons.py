"""Paired patient-level bootstrap differences between Phase 2 models.

Reads the saved out-of-fold predictions of one Phase 2 run (no retraining) and
writes paired_differences_<protocol>.csv. Every bootstrap replicate resamples
the same patients for both models, so the interval reflects the paired design.

Usage (from the repository root):
    python reports/phase2/ieee/scripts/paired_comparisons.py \
        results/phase2/main/metrics reports/phase2/ieee/tables
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PAIRS = [
    ("BN-MLP Stack", "MLP"), ("BN-MLP Stack", "Bayesian Network"),
    ("MLP", "Bayesian Network"), ("MLP", "Logistic Regression"), ("MLP", "SVM"),
    ("BN-MLP Stack", "SVM"), ("SVM", "Bayesian Network"),
    ("Logistic Regression", "Bayesian Network"),
]
METRICS = {
    "roc_auc": roc_auc_score,
    "log_loss": lambda y, p: log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
    "brier": brier_score_loss,
}
N_BOOT, SEED = 2000, 42


def main(metrics_dir: str, out_dir: str) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for protocol in ("pooled", "cross-cohort"):
        oof = pd.read_csv(Path(metrics_dir) / f"oof_{protocol}.csv")
        frames = {m: d.sort_values("patient") for m, d in oof.groupby("model")}
        ref = next(iter(frames.values()))
        y = ref.y_true.to_numpy(dtype=int)
        for m, d in frames.items():
            if not np.array_equal(d.patient.to_numpy(), ref.patient.to_numpy()):
                raise ValueError(f"{m}: patients not aligned")
        prob = {m: d.p_pos.to_numpy() for m, d in frames.items()}
        rng = np.random.default_rng(SEED)
        draws = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
        draws = [i for i in draws if np.unique(y[i]).size == 2]
        rows = []
        for a, b in PAIRS:
            for name, f in METRICS.items():
                diffs = [f(y[i], prob[a][i]) - f(y[i], prob[b][i]) for i in draws]
                lo, hi = np.quantile(diffs, [0.025, 0.975])
                rows.append({"protocol": protocol, "model_a": a, "model_b": b, "metric": name,
                             "difference": f(y, prob[a]) - f(y, prob[b]),
                             "ci_low": lo, "ci_high": hi,
                             "excludes_zero": bool(lo > 0 or hi < 0),
                             "n_bootstrap": len(draws)})
        pd.DataFrame(rows).to_csv(out / f"paired_differences_{protocol}.csv", index=False)
        print(f"wrote paired_differences_{protocol}.csv")


if __name__ == "__main__":
    main(*sys.argv[1:3])
