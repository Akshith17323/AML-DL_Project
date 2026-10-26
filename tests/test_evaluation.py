import numpy as np
import pandas as pd
import pytest

from src.data.loader import get_xy, load_cleveland, load_config
from src.evaluation.cv import evaluate_model, expand_grid, fold_hash
from src.evaluation.metrics import (calibration_slope_intercept, check_probabilities, compute_metrics,
                                    expected_calibration_error, youden_threshold)
from src.evaluation.run_evaluation import validate_oof
from src.evaluation.splits import outer_folds
from src.models.pipelines import (build_bn_pipeline, build_gb_pipeline, build_lr_pipeline, build_nb_pipeline,
                                  build_rf_pipeline, build_svm_pipeline)


def test_compute_metrics_known_values():
    y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([0.1, 0.4, 0.6, 0.35, 0.8, 0.9])
    m = compute_metrics(y, p, 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 2)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3) and m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(7 / 9)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_perfect_and_constant_predictors():
    y = np.array([0, 0, 1, 1])
    assert compute_metrics(y, np.array([0, 0, 1, 1.0]))["roc_auc"] == 1.0
    m = compute_metrics(y, np.zeros(4))
    assert m["sensitivity"] == 0 and m["specificity"] == 1 and np.isnan(m["precision"]) and m["f1"] == 0


def test_per_row_thresholds():
    y = np.array([0, 1, 1, 0]); p = np.array([0.3, 0.3, 0.6, 0.6])
    m = compute_metrics(y, p, np.array([0.5, 0.2, 0.5, 0.7]))
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 2, 0, 0)


@pytest.mark.parametrize("bad", [[0.1, 1.2], [0.1, -0.1], [0.1, np.nan], [0.1, np.inf]])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        check_probabilities(bad)


def test_metric_input_errors():
    with pytest.raises(ValueError, match="Both classes"):
        compute_metrics([1, 1], [0.2, 0.9])
    with pytest.raises(ValueError, match="different lengths"):
        compute_metrics([0, 1, 1], [0.2, 0.9])


def test_ece_and_calibration_slope():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=4000); y = (rng.uniform(size=4000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.04
    slope, _ = calibration_slope_intercept(y, p)
    assert 0.85 < slope < 1.15
    over = np.clip(0.5 + (p - 0.5) * 1.8, 0.001, 0.999)  # over-confident
    assert calibration_slope_intercept(y, over)[0] < slope


def test_youden_uses_given_data_only():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1]); p = np.array([.1, .2, .3, .45, .4, .6, .7, .9])
    t = youden_threshold(y, p)
    pred = (p >= t).astype(int)
    assert t == pytest.approx(0.4) and pred.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]


def test_expand_grid():
    assert len(expand_grid({"a": [1, 2], "b": [3, 4, 5]})) == 6
    assert expand_grid({}) == [{}]


def _mini_oof(model_names=("A", "B")):
    X, y = get_xy(load_cleveland()); outer = outer_folds(y)[:5]
    rows = []
    rng = np.random.default_rng(0)
    for m in model_names:
        for f, (tr, va) in enumerate(outer):
            p = rng.uniform(size=len(va))
            rows.append(pd.DataFrame({"model": m, "repeat": 0, "fold": f, "outer_id": f, "patient": va,
                                      "y_true": y.iloc[va].to_numpy(), "p_pos": p, "p_neg": 1 - p, "youden_threshold": .5}))
    return pd.concat(rows, ignore_index=True), y, outer


def test_validate_oof_accepts_good_and_detects_problems():
    oof, y, outer = _mini_oof()
    assert validate_oof(oof, y, outer)["identical_folds_across_models"]
    with pytest.raises(AssertionError, match="missing or duplicated"):
        validate_oof(oof.drop(index=oof.index[0]), y, outer)
    bad = oof.copy(); bad.loc[3, "p_pos"] = 1.5
    with pytest.raises(ValueError):
        validate_oof(bad, y, outer)
    bad = oof.copy(); bad.loc[bad.model == "B", "outer_id"] = (bad.loc[bad.model == "B", "outer_id"] + 1) % 5
    with pytest.raises(AssertionError):
        validate_oof(bad, y, outer)
    bad = oof.copy(); bad.loc[0, "y_true"] = 1 - bad.loc[0, "y_true"]
    with pytest.raises(AssertionError, match="misaligned"):
        validate_oof(bad, y, outer)


@pytest.mark.parametrize("builder,grid", [(build_nb_pipeline, {"model__alpha": [0.5, 2.0]}),
                                          (build_bn_pipeline, {"model__pseudo_count": [0.5, 2.0]}),
                                          (build_lr_pipeline, {"model__C": [0.1, 1.0]}),
                                          (build_svm_pipeline, {"model__C": [0.1, 1.0]}),
                                          (build_rf_pipeline, {"model__max_depth": [3, None]}),
                                          (build_gb_pipeline, {"model__max_depth": [2, 3]})])
def test_outer_validation_labels_never_influence_predictions(builder, grid):
    """Shuffling the labels of the outer validation rows must not change any prediction."""
    cfg = load_config(); X, y = get_xy(load_cleveland(config=cfg)); outer = outer_folds(y, cfg)[:2]
    # a row validating in fold 0 trains fold 1, so alter only each fold's own validation labels
    for f, (tr, va) in enumerate(outer):
        y_alt = y.copy(); y_alt.iloc[va] = 1 - y.iloc[va].to_numpy()
        o1, s1 = evaluate_model("m", builder(cfg), grid, X, y, [(tr, va)], cfg, 5, verbose=False)
        o2, s2 = evaluate_model("m", builder(cfg), grid, X, y_alt, [(tr, va)], cfg, 5, verbose=False)
        assert np.array_equal(o1["p_pos"], o2["p_pos"])
        assert o1["youden_threshold"].iloc[0] == o2["youden_threshold"].iloc[0]
        assert s1.drop(columns=["outer_id"]).equals(s2.drop(columns=["outer_id"]))
        assert not np.array_equal(o1["y_true"], o2["y_true"])


def test_evaluate_model_output_shape_and_class_semantics():
    cfg = load_config(); X, y = get_xy(load_cleveland(config=cfg)); outer = outer_folds(y, cfg)[:3]
    o, s = evaluate_model("NB", build_nb_pipeline(cfg), {"model__alpha": [1.0]}, X, y, outer, cfg, 5, verbose=False)
    assert len(o) == sum(len(va) for _, va in outer) and len(s) == 3
    assert np.allclose(o["p_pos"] + o["p_neg"], 1) and o["p_pos"].between(0, 1).all()
    assert o[o.y_true == 1]["p_pos"].mean() > o[o.y_true == 0]["p_pos"].mean()


def test_fold_hash_changes_with_folds():
    _, y = get_xy(load_cleveland())
    f = outer_folds(y)
    assert fold_hash(f) == fold_hash(outer_folds(y)) and fold_hash(f) != fold_hash(f[::-1])
