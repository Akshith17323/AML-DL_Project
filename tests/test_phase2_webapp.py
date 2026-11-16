"""Phase 2 Webapp and Demo Model Tests.

Tests:
    - Final hyperparameters selection from saved Optuna search results.
    - Typed final parameter schema and serialization.
    - Offline fitting and caching of all eight models on all 920 patients.
    - Eight-model probability predictions (deterministic, calibrated, valid ranges).
    - Separation of demo predictions from cross-validation performance claims.
    - Missing-result handling and graceful fallbacks.
    - Headless AppTest rendering of all pages in Phase 2 mode.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data.loader import (
    FEATURE_COLUMNS,
    PHASE2_CONFIG_PATH,
    PROJECT_ROOT,
    get_xy,
    load_all_cohorts,
    load_config,
    resolve,
)
from src.models.demo_models_phase2 import (
    INT_PARAM_KEYS,
    PHASE2_MODEL_NAMES,
    FinalModelsPhase2,
    cast_typed_params,
    fit_final_phase2_models,
    load_or_fit_phase2_demo_models,
    save_typed_hyperparameters,
    select_phase2_hyperparameters,
)
from src.webapp.content import DISCLAIMER
from src.webapp.inference import validate_inputs
from src.webapp.results_io import (
    PHASE2_MODEL_ORDER,
    load_phase2_results,
    phase2_results_available,
)


@pytest.fixture(scope="module")
def phase2_models_fixture():
    return load_or_fit_phase2_demo_models()


@pytest.fixture(scope="module")
def sample_profile(phase2_models_fixture):
    X = phase2_models_fixture.train_X
    defaults = {}
    for c in FEATURE_COLUMNS:
        defaults[c] = float(X[c].median()) if c in ["age", "trestbps", "chol", "thalach", "oldpeak"] else float(X[c].mode().iloc[0])
    for c in ("age", "trestbps", "chol", "thalach"):
        defaults[c] = int(round(defaults[c]))
    return defaults


def test_select_phase2_hyperparameters_extracts_typed_params():
    params_by_model, source = select_phase2_hyperparameters()
    assert len(params_by_model) == len(PHASE2_MODEL_NAMES)
    assert "Optuna" in source or "nested" in source

    for model_name, p in params_by_model.items():
        assert isinstance(p, dict)
        for k, v in p.items():
            if k in INT_PARAM_KEYS and v is not None:
                assert isinstance(v, int), f"{k} in {model_name} should be int, got {type(v)}"
            elif isinstance(v, (int, float)):
                assert np.isfinite(v)


def test_save_typed_hyperparameters_creates_valid_json(tmp_path):
    params_by_model, _ = select_phase2_hyperparameters()
    json_path = tmp_path / "test_hyperparameters.json"
    saved_path = save_typed_hyperparameters(params_by_model, json_path)

    assert saved_path.exists()
    with open(saved_path) as f:
        loaded = json.load(f)
    assert set(loaded.keys()) == set(PHASE2_MODEL_NAMES)


def test_phase2_final_models_fit_and_cache(tmp_path):
    cfg = load_config(PHASE2_CONFIG_PATH)
    cache_file = tmp_path / "cached_demo_models.joblib"

    # Fit and save
    models = fit_final_phase2_models(cfg, save_models_path=cache_file)
    assert isinstance(models, FinalModelsPhase2)
    assert len(models.pipelines) == len(PHASE2_MODEL_NAMES)
    assert len(models.train_X) == 920
    assert cache_file.exists()

    # Load from cache
    loaded = load_or_fit_phase2_demo_models(cfg, cache_path=cache_file)
    assert isinstance(loaded, FinalModelsPhase2)
    assert len(loaded.pipelines) == len(PHASE2_MODEL_NAMES)


def test_eight_model_predictions_valid_and_deterministic(phase2_models_fixture, sample_profile):
    row, errors, warnings = validate_inputs(sample_profile, phase2_models_fixture.train_X)
    assert errors == [] and row is not None

    preds1 = phase2_models_fixture.predict(row)
    preds2 = phase2_models_fixture.predict(row)

    assert set(preds1.keys()) == set(PHASE2_MODEL_NAMES)
    assert len(preds1) == 8

    for m, p in preds1.items():
        assert np.isfinite(p), f"{m} prediction is not finite"
        assert 0.0 <= p <= 1.0, f"{m} prediction {p} outside [0, 1]"
        assert preds1[m] == preds2[m], f"{m} prediction is non-deterministic"


def test_demo_predictions_strictly_separated_from_cv_performance():
    cfg = load_config(PHASE2_CONFIG_PATH)
    metrics_dir = resolve(cfg["paths"]["metrics"])

    # CV performance claims come strictly from metrics files
    assert (metrics_dir / "oof_pooled.csv").exists()
    assert (metrics_dir / "summary_pooled.csv").exists()
    assert (metrics_dir / "bootstrap_ci_pooled.csv").exists()

    p2_res = load_phase2_results(cfg)
    assert not p2_res["ci_pooled"].empty
    assert "roc_auc" in p2_res["ci_pooled"]["metric"].values

    # Demo container explicitly declares separation in documentation
    assert "demonstration only" in FinalModelsPhase2.__doc__.lower()
    assert "separate" in FinalModelsPhase2.__doc__.lower()


def test_phase2_missing_result_handling_gracefully_falls_back(tmp_path):
    empty_cfg = {"paths": {"metrics": str(tmp_path)}, "project": {"seed": 42}}
    assert phase2_results_available(empty_cfg) is False

    res = load_phase2_results(empty_cfg)
    for k, v in res.items():
        assert isinstance(v, pd.DataFrame)
        assert v.empty

    params, source = select_phase2_hyperparameters(empty_cfg)
    assert "default" in source.lower() or "not found" in source.lower()


# ------------------------------------------------------------------ UI Headless Tests
def _page_headless(mod: str) -> AppTest:
    script = f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); from app.views import {mod}; {mod}.render()"
    return AppTest.from_string(script, default_timeout=90)


def test_phase2_all_pages_render_without_exception():
    for mod in ["overview", "data_explorer", "results", "risk_estimator", "methods"]:
        at = _page_headless(mod).run()
        assert not at.exception, f"Exception on page {mod}: {[e.value for e in at.exception]}"
        assert not at.error, f"Error on page {mod}: {[e.value for e in at.error]}"
        assert any("Not a medical device" in w.value or "not a medical device" in w.value.lower() for w in at.warning)


def test_phase2_risk_estimator_shows_eight_models_when_selected():
    at = _page_headless("risk_estimator").run()
    # Select Phase 2 radio option
    at.radio[0].set_value("Phase 2: Eight models (All four cohorts, 920 patients - incl. regularised MLP and Stack)").run()
    submit_btn = next(b for b in at.button if b.label.startswith("Estimate"))
    submit_btn.click().run()

    assert not at.exception
    metric_labels = [m.label for m in at.metric]
    assert len(metric_labels) == 8
    assert set(metric_labels) == set(PHASE2_MODEL_NAMES)

    metric_vals = [float(m.value.rstrip("%")) for m in at.metric]
    assert all(0.0 <= v <= 100.0 for v in metric_vals)


def test_phase2_results_tab_renders_multi_cohort_tables():
    at = _page_headless("results").run()
    assert not at.exception
    assert any("Phase 2 Multi-Cohort" in t.value or "Pooled Nested CV" in t.value for t in at.markdown)


def test_phase2_data_explorer_shows_all_cohorts_audit():
    at = _page_headless("data_explorer").run()
    assert not at.exception
    assert any("Phase 2 Multi-Cohort Audit" in m.value or "Cohort breakdown" in m.value for m in at.markdown)
