"""Tests for SHAP attribution and nonlinear interaction analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import FEATURE_COLUMNS, PHASE2_CONFIG_PATH, get_xy, load_all_cohorts, load_config, resolve
from src.evaluation.shap_analysis import compute_mlp_shap, plot_interaction_dependence, plot_shap_summary
from src.models.demo_models_phase2 import load_or_fit_phase2_demo_models


@pytest.fixture(scope="module")
def shap_artifacts():
    cfg = load_config(PHASE2_CONFIG_PATH)
    metrics_dir = resolve(cfg["paths"]["metrics"])
    manifest_path = metrics_dir / "mlp_shap_manifest.json"
    values_path = metrics_dir / "mlp_shap_values.csv"
    importance_path = metrics_dir / "mlp_shap_importance.csv"
    return manifest_path, values_path, importance_path


def test_shap_artifacts_exist_and_match_schema(shap_artifacts):
    manifest_path, values_path, importance_path = shap_artifacts
    assert manifest_path.exists(), "mlp_shap_manifest.json missing"
    assert values_path.exists(), "mlp_shap_values.csv missing"
    assert importance_path.exists(), "mlp_shap_importance.csv missing"

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert "n_samples_explained" in manifest
    assert "key_interactions_evaluated" in manifest
    assert len(manifest["key_interactions_evaluated"]) >= 2
    assert "age" in manifest["key_interactions_evaluated"][0]["features"]
    assert "trestbps" in manifest["key_interactions_evaluated"][1]["features"]

    df_vals = pd.read_csv(values_path)
    assert list(df_vals.columns) == FEATURE_COLUMNS
    assert len(df_vals) == manifest["n_samples_explained"]
    assert np.isfinite(df_vals.to_numpy()).all()

    df_imp = pd.read_csv(importance_path)
    assert set(df_imp["feature"]) == set(FEATURE_COLUMNS)
    assert (df_imp["mean_abs_shap"] >= 0).all()


def test_shap_computation_produces_finite_values_and_efficiency():
    df = load_all_cohorts()
    X, _ = get_xy(df)
    models = load_or_fit_phase2_demo_models()
    mlp = models.pipelines["MLP"]

    shap_vals, sample_df, base_val = compute_mlp_shap(
        mlp,
        X,
        sample_size=15,
        background_size=10,
        n_samples=50,
        seed=123,
    )
    assert shap_vals.shape == (15, len(FEATURE_COLUMNS))
    assert np.isfinite(shap_vals).all()
    assert 0.0 <= base_val <= 1.0


def test_shap_figures_exist():
    cfg = load_config(PHASE2_CONFIG_PATH)
    fig_dir = resolve(cfg["paths"]["figures"])
    expected_figures = [
        "mlp_shap_beeswarm.png",
        "mlp_shap_bar_importance.png",
        "mlp_shap_interaction_age_thalach.png",
        "mlp_shap_interaction_trestbps_cp.png",
        "mlp_vs_bn_interaction_comparison.png",
    ]
    for fig_name in expected_figures:
        fig_path = fig_dir / fig_name
        assert fig_path.exists(), f"Figure {fig_name} missing from {fig_dir}"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_name} is empty or corrupted"
