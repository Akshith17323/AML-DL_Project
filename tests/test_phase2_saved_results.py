import json

import numpy as np
import pandas as pd
import pytest

from src.data.loader import PROJECT_ROOT

METRICS = PROJECT_ROOT / "results/phase2/main/metrics"
needs_results = pytest.mark.skipif(not (METRICS / "oof_pooled.csv").exists(),
                                   reason="run the full Phase 2 evaluation first")


@needs_results
@pytest.mark.parametrize("protocol,folds", [("pooled", 5), ("cross-cohort", 4)])
def test_full_phase2_predictions_trials_and_manifest(protocol, folds):
    oof = pd.read_csv(METRICS / f"oof_{protocol}.csv")
    models = oof.model.unique().tolist()
    assert len(models) == 8 and len(oof) == 8 * 920
    assert oof.p_pos.between(0, 1).all() and np.allclose(oof.p_pos + oof.p_neg, 1)
    assert (oof.groupby("model").patient.nunique() == 920).all()
    assert (oof.groupby(["model", "patient"]).size() == 1).all()

    trials = pd.read_csv(METRICS / f"optuna_trials_{protocol}.csv")
    assert len(trials) == 8 * folds * 12
    assert set(trials.state) == {"COMPLETE"}
    manifest = json.loads((METRICS / f"manifest_{protocol}.json").read_text())
    assert manifest["n_patients"] == 920 and manifest["outer_folds_run"] == folds
    assert manifest["search"]["method"] == "optuna_tpe"
    assert manifest["bootstrap_samples"] == 500 and manifest["figures"]


@needs_results
def test_cross_cohort_outer_folds_each_hold_one_complete_cohort():
    oof = pd.read_csv(METRICS / "oof_cross-cohort.csv")
    one_model = oof[oof.model == oof.model.iloc[0]]
    held_out = one_model.groupby("outer_id").cohort.unique()
    assert all(len(cohorts) == 1 for cohorts in held_out)
    assert {cohorts[0] for cohorts in held_out} == {"Cleveland", "Hungarian", "Switzerland", "VA"}


@needs_results
def test_ieee_tables_trace_to_saved_summaries():
    summary = pd.read_csv(METRICS / "summary_pooled.csv")
    table = (PROJECT_ROOT / "reports/phase2/tables/pooled_results.tex").read_text()
    for row in summary[summary.cohort == "ALL"].itertuples():
        assert f"{row.roc_auc:.3f}" in table
    assert "Run the Phase 2 pilot" not in table
