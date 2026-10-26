"""Tests for the Streamlit interface: logic layer (src/webapp) and headless UI runs (AppTest)."""
import copy

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data.loader import FEATURE_COLUMNS, PROJECT_ROOT, load_config
from src.webapp import inference as inf
from src.webapp.content import DISCLAIMER
from src.webapp.results_io import MODEL_ORDER, calibration_points, curve_points, metrics_at_threshold

HAS_RESULTS = (PROJECT_ROOT / "results/metrics/oof_predictions.csv").exists()
needs_results = pytest.mark.skipif(not HAS_RESULTS, reason="run the evaluation first")


@pytest.fixture(scope="module")
def models():
    return inf.train_final_models()


@pytest.fixture(scope="module")
def defaults(models):
    return inf.default_values(models.train_X)


# ------------------------------------------------------------------ validation
def test_default_profile_is_valid(models, defaults):
    row, errors, warnings = inf.validate_inputs(defaults, models.train_X)
    assert errors == [] and warnings == [] and list(row.columns) == FEATURE_COLUMNS and len(row) == 1


def test_optional_fields_may_be_missing_others_may_not(defaults):
    row, errors, _ = inf.validate_inputs({**defaults, "ca": None, "thal": None})
    assert errors == [] and row[["ca", "thal"]].isna().all(axis=None)
    row, errors, _ = inf.validate_inputs({**defaults, "age": None})
    assert row is None and any("required" in e for e in errors)


@pytest.mark.parametrize("bad,why", [({"cp": 9}, "not one of"), ({"sex": 2}, "not one of"), ({"thal": 5}, "not one of"),
                                     ({"age": "abc"}, "not a number"), ({"chol": float("nan")}, "finite"),
                                     ({"age": 5}, "plausible"), ({"trestbps": 900}, "plausible"), ({"oldpeak": -1}, "plausible")])
def test_invalid_inputs_are_rejected_with_message(defaults, bad, why):
    row, errors, _ = inf.validate_inputs({**defaults, **bad})
    assert row is None and any(why in e for e in errors)


def test_out_of_training_range_warns_but_predicts(models, defaults):
    row, errors, warnings = inf.validate_inputs({**defaults, "age": 95}, models.train_X)
    assert errors == [] and any("extrapolating" in w for w in warnings)
    probs = models.predict(row)
    assert all(0 <= p <= 1 for p in probs.values())


# ------------------------------------------------------------------ models
def test_final_hyperparameters_use_joint_mode(tmp_path):
    cfg = copy.deepcopy(load_config()); cfg["paths"]["metrics"] = str(tmp_path)
    rows = ([{"model": "MLP", "model__weight_decay": 0.01, "model__dropout": 0.5}] * 3 +
            [{"model": "MLP", "model__weight_decay": 0.001, "model__dropout": 0.5}] * 2 +
            [{"model": "Naive Bayes", "model__alpha": 1.0}] * 2 + [{"model": "Naive Bayes", "model__alpha": 2.0}] +
            [{"model": "Bayesian Network", "model__pseudo_count": 0.5}])
    pd.DataFrame(rows).to_csv(tmp_path / "selected_hyperparameters.csv", index=False)
    hp, src = inf.final_hyperparameters(cfg)
    assert hp["MLP"] == {"model__weight_decay": 0.01, "model__dropout": 0.5}
    assert hp["Naive Bayes"] == {"model__alpha": 1.0} and hp["Bayesian Network"] == {"model__pseudo_count": 0.5}
    assert "nested" in src


def test_final_hyperparameters_fall_back_to_config_when_results_missing(tmp_path):
    cfg = copy.deepcopy(load_config()); cfg["paths"]["metrics"] = str(tmp_path)
    hp, src = inf.final_hyperparameters(cfg)
    assert hp == {m: {} for m in inf.MODEL_NAMES} and "defaults" in src


def test_predictions_are_valid_and_deterministic(models, defaults):
    row, _, _ = inf.validate_inputs(defaults)
    a = models.predict(row)
    assert list(a) == MODEL_ORDER and all(np.isfinite(p) and 0 <= p <= 1 for p in a.values())
    b = inf.train_final_models().predict(row)
    assert a == b  # same seed, same data -> identical


def test_missing_ca_thal_equals_training_mode_imputation(models, defaults):
    mode_ca, mode_thal = models.train_X["ca"].mode().iloc[0], models.train_X["thal"].mode().iloc[0]
    r1, _, _ = inf.validate_inputs({**defaults, "ca": None, "thal": None})
    r2, _, _ = inf.validate_inputs({**defaults, "ca": mode_ca, "thal": mode_thal})
    assert models.predict(r1) == models.predict(r2)


def test_sensible_direction_for_extreme_profiles(models, defaults):
    low = {**defaults, "ca": 0, "thal": 3, "cp": 2, "exang": 0, "oldpeak": 0.0, "slope": 1, "age": 40, "thalach": 180}
    high = {**defaults, "ca": 3, "thal": 7, "cp": 4, "exang": 1, "oldpeak": 3.0, "slope": 2, "age": 65, "thalach": 110}
    pl, ph = (models.predict(inf.validate_inputs(v)[0]) for v in (low, high))
    assert all(ph[m] > pl[m] for m in MODEL_ORDER)


def test_explanation_shape_and_additivity_sign(models, defaults):
    row, _, _ = inf.validate_inputs(defaults)
    s = inf.explain_mlp(models, row, n_background=10, nsamples=100)
    assert list(s.index) == FEATURE_COLUMNS and np.isfinite(s.values).all() and 0 <= s.attrs["base_value"] <= 1
    again = inf.explain_mlp(models, row, n_background=10, nsamples=100)
    assert np.allclose(s.values, again.values)  # seeded -> reproducible


def test_default_values_are_typical_not_extreme(models):
    d = inf.default_values(models.train_X)
    X = models.train_X
    assert all(X[c].min() <= d[c] <= X[c].max() for c in FEATURE_COLUMNS)


# ------------------------------------------------------------------ results_io
@needs_results
def test_results_helpers():
    oof = pd.read_csv(PROJECT_ROOT / "results/metrics/oof_predictions.csv")
    roc = curve_points(oof, "roc", 0)
    assert set(roc.model) == set(MODEL_ORDER) and roc[["x", "y"]].apply(lambda c: c.between(0, 1).all()).all()
    cal = calibration_points(oof)
    assert set(cal.model) == set(MODEL_ORDER) and cal[["predicted", "observed"]].apply(lambda c: c.between(0, 1).all()).all()
    m5 = metrics_at_threshold(oof, 0.5)
    assert list(m5.index) == MODEL_ORDER
    assert (m5[["tn", "fp", "fn", "tp"]].sum(axis=1).round(6) == 303).all()
    lo, hi = metrics_at_threshold(oof, 0.1), metrics_at_threshold(oof, 0.9)
    assert (lo.sensitivity >= m5.sensitivity).all() and (hi.sensitivity <= m5.sensitivity).all()  # monotone in threshold
    fold = pd.read_csv(PROJECT_ROOT / "results/metrics/summary_pooled_repeats.csv")
    ref = fold[(fold.threshold_scheme == "fixed_0.5")].set_index("model")
    assert np.allclose(m5.sensitivity.to_numpy(), ref.loc[MODEL_ORDER, "sensitivity_mean"].to_numpy())  # matches the saved summary


# ------------------------------------------------------------------ UI (headless)
def _page(mod: str) -> AppTest:
    script = f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); from app.views import {mod}; {mod}.render()"
    return AppTest.from_string(script, default_timeout=90)


@needs_results
@pytest.mark.parametrize("mod", ["overview", "data_explorer", "results", "risk_estimator", "methods"])
def test_every_page_renders_with_disclaimer_and_no_exception(mod):
    at = _page(mod).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("Not a medical device" in w.value or "not a medical device" in w.value.lower() for w in at.warning)
    assert DISCLAIMER in [w.value for w in at.warning]


@needs_results
def test_entry_point_runs_and_shows_overview():
    at = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py"), default_timeout=90).run()
    assert not at.exception and [t.value for t in at.title] == ["Heart Disease Risk Modelling"]
    assert len(at.metric) == 3


@needs_results
def test_overview_numbers_come_from_saved_results():
    at = _page("overview").run()
    ci = pd.read_csv(PROJECT_ROOT / "results/metrics/bootstrap_ci.csv")
    expected = [f"{ci[(ci.model == m) & (ci.metric == 'roc_auc')].estimate.iloc[0]:.3f}" for m in MODEL_ORDER]
    assert [m.value for m in at.metric] == expected


def test_estimator_shows_nothing_until_submitted():
    at = _page("risk_estimator").run()
    assert not at.exception and len(at.metric) == 0 and "profile_values" not in at.session_state


def test_estimator_submit_shows_three_valid_probabilities():
    at = _page("risk_estimator").run()
    _submit(at).click().run()
    assert not at.exception and len(at.metric) == 3
    vals = [float(m.value.rstrip("%")) for m in at.metric]
    assert all(0 <= v <= 100 for v in vals) and [m.label for m in at.metric] == MODEL_ORDER


def test_estimator_unknown_ca_thal_is_noted_and_predicts():
    at = _page("risk_estimator").run()
    at.selectbox(key="in_ca").select(inf_unknown()).run()
    at.selectbox(key="in_thal").select(inf_unknown()).run()
    _submit(at).click().run()
    assert not at.exception and len(at.metric) == 3
    assert any("training-data mode" in c.value for c in at.caption)


def test_estimator_changes_with_inputs_and_explanation_button_works():
    at = _page("risk_estimator").run()
    _submit(at).click().run()
    base = [m.value for m in at.metric]
    at.selectbox(key="in_ca").select(3).run()
    at.selectbox(key="in_thal").select(7).run()
    _submit(at).click().run()
    assert [m.value for m in at.metric] != base
    at.button(key="explain").click().run()
    assert not at.exception
    assert any("SHAP" in i.value for i in at.info)


def _snapshot():
    skip = {".venv", "__pycache__", ".pytest_cache", ".git"}
    return {str(p.relative_to(PROJECT_ROOT)): (p.stat().st_size, p.stat().st_mtime_ns) for p in PROJECT_ROOT.rglob("*")
            if p.is_file() and not (skip & set(p.relative_to(PROJECT_ROOT).parts))}


def test_nothing_entered_is_written_to_disk_or_kept_beyond_session_state():
    before = _snapshot()
    at = _page("risk_estimator").run()
    at.number_input(key="in_age").set_value(61).run()
    _submit(at).click().run()
    at.button(key="explain").click().run()
    assert not at.exception
    assert _snapshot() == before  # no file created, modified or removed by using the estimator
    allowed = {"profile_values", "explain"} | {f"in_{c}" for c in FEATURE_COLUMNS}
    assert {k for k in at.session_state.filtered_state if not k.startswith("FormSubmitter:")} <= allowed


def _submit(at):
    """The form's submit button (in Streamlit 1.50 AppTest it is listed under `button`)."""
    return next(b for b in at.button if b.label.startswith("Estimate"))


def inf_unknown():
    from app.views.risk_estimator import UNKNOWN
    return UNKNOWN


# ------------------------------------------------------------------ regression tests for UI polish
def test_data_explorer_shows_level_counts_as_a_table_not_raw_json():
    at = _page("data_explorer").run()
    assert not at.exception and len(at.json) == 0
    assert any("Patients per level" in m.value for m in at.markdown)
    assert at.metric[1].value == "139 (45.9%)"


def test_estimator_tells_the_user_results_belong_to_the_last_submission():
    at = _page("risk_estimator").run()
    _submit(at).click().run()
    assert any("last submitted" in c.value for c in at.caption)


def test_toolbar_is_minimal_and_telemetry_is_off():
    cfg = (PROJECT_ROOT / ".streamlit/config.toml").read_text()  # no tomllib on Python 3.9
    assert 'toolbarMode = "minimal"' in cfg and "gatherUsageStats = false" in cfg
