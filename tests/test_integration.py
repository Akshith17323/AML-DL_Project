"""Integration and reproducibility tests: analysis pipeline, saved-result consistency, repo hygiene."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.data.loader import PROJECT_ROOT, get_xy, load_cleveland, load_config
from src.evaluation.cv import evaluate_model, fold_hash
from src.evaluation.run_evaluation import MODELS, analyse
from src.evaluation.splits import outer_folds

METRICS = PROJECT_ROOT / "results" / "metrics"
needs_results = pytest.mark.skipif(not (METRICS / "oof_predictions.csv").exists(),
                                   reason="run `python -m src.evaluation.run_evaluation` first")


def _synthetic_oof(y, outer, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name in MODELS:
        for f, (_, va) in enumerate(outer):
            yv = y.iloc[va].to_numpy()
            p = np.clip(0.5 + (yv - 0.5) * 0.5 + rng.normal(0, 0.25, len(va)), 0.01, 0.99)
            rows.append(pd.DataFrame({"model": name, "repeat": f // 5, "fold": f % 5, "outer_id": f, "patient": va,
                                      "y_true": yv, "p_pos": p, "p_neg": 1 - p, "youden_threshold": 0.5}))
    return pd.concat(rows, ignore_index=True)


def test_analysis_pipeline_end_to_end_on_synthetic_predictions(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["paths"].update({"figures": str(tmp_path / "fig"), "metrics": str(tmp_path / "met")})
    cfg["evaluation"]["n_bootstrap"] = 20
    (tmp_path / "met").mkdir()
    X, y = get_xy(load_cleveland(config=cfg)); outer = outer_folds(y, cfg)
    oof = _synthetic_oof(y, outer)
    res = analyse(cfg, X, y, outer, oof, pd.DataFrame())
    for f in ["fold_metrics.csv", "pooled_metrics_by_repeat.csv", "summary_fold_level.csv", "summary_pooled_repeats.csv",
              "bootstrap_ci.csv", "bootstrap_paired_differences.csv", "evaluation_manifest.json"]:
        assert (tmp_path / "met" / f).stat().st_size > 0, f
    for f in ["roc_curves.png", "pr_curves.png", "confusion_matrices.png", "calibration_curves.png"]:
        assert (tmp_path / "fig" / f).stat().st_size > 0, f
    assert (tmp_path / "met" / "model_comparison_fold_level.csv").exists()
    fm = res["fold"]
    assert len(fm) == len(MODELS) * 25 * 2  # models x outer folds x threshold schemes
    row = fm[(fm.model == "SVM") & (fm.outer_id == 3) & (fm.threshold_scheme == "fixed_0.5")].iloc[0]
    d = oof[(oof.model == "SVM") & (oof.outer_id == 3)]
    assert row["roc_auc"] == pytest.approx(roc_auc_score(d.y_true, d.p_pos))
    assert res["checks"]["identical_folds_across_models"]
    assert json.loads((tmp_path / "met" / "evaluation_manifest.json").read_text())["outer"]["n_outer_fits_per_model"] == 25


def test_tuning_grids_match_pipeline_parameters():
    cfg = load_config()
    for key, builder in MODELS.values():
        pipe = builder(cfg)
        for param, values in cfg["tuning"][key].items():
            assert param in pipe.get_params(), (key, param)
            assert len(values) >= 2


@needs_results
def test_saved_folds_match_configuration():
    cfg = load_config(); _, y = get_xy(load_cleveland(config=cfg))
    manifest = json.loads((METRICS / "evaluation_manifest.json").read_text())
    assert manifest["outer"]["fold_hash"] == fold_hash(outer_folds(y, cfg))
    assert manifest["seed"] == cfg["project"]["seed"]


@needs_results
def test_saved_predictions_are_complete_and_valid():
    oof = pd.read_csv(METRICS / "oof_predictions.csv")
    assert set(oof.model) == set(MODELS) and len(oof) == len(MODELS) * 25 * 303 // 5
    assert oof.p_pos.between(0, 1).all() and np.isfinite(oof.p_pos).all()
    assert (oof.groupby(["model", "repeat"]).patient.nunique() == 303).all()


@needs_results
@pytest.mark.parametrize("name", list(MODELS))
def test_rerunning_a_fold_reproduces_saved_predictions(name):
    """Re-run nested CV (tuning, threshold, refit) for outer fold 7; must equal the stored predictions."""
    cfg = load_config(); X, y = get_xy(load_cleveland(config=cfg)); outer = outer_folds(y, cfg)
    key, builder = MODELS[name]
    f = 7
    o, _ = evaluate_model(name, builder(cfg), cfg["tuning"][key], X, y, [outer[f]], cfg, 5, verbose=False, fold_offset=f)
    saved = pd.read_csv(METRICS / "oof_predictions.csv")
    saved = saved[(saved.model == name) & (saved.outer_id == f)].sort_values("patient")
    o = o.sort_values("patient")
    assert o["patient"].tolist() == saved["patient"].tolist()
    assert np.allclose(o["p_pos"].to_numpy(), saved["p_pos"].to_numpy(), atol=1e-9)
    assert np.allclose(o["youden_threshold"].to_numpy(), saved["youden_threshold"].to_numpy(), atol=1e-9)


def test_gitignore_excludes_environments_caches_secrets_and_raw_data():
    lines = {ln.strip() for ln in (PROJECT_ROOT / ".gitignore").read_text().splitlines()}
    for pat in [".venv/", "__pycache__/", ".pytest_cache/", ".env", "data/raw/*", "!data/raw/README.md", "*.log", ".ipynb_checkpoints/"]:
        assert pat in lines, pat
