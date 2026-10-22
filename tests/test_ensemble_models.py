"""Unit tests for Random Forest and Gradient Boosting ensemble pipelines."""
import numpy as np
import pytest
from sklearn.base import clone

from src.data.loader import get_xy, load_cleveland
from src.evaluation.splits import outer_folds
from src.models.pipelines import build_gb_pipeline, build_rf_pipeline


@pytest.fixture(scope="module")
def data():
    return get_xy(load_cleveland())


@pytest.fixture(scope="module")
def fold0(data):
    X, y = data
    tr, va = outer_folds(y)[0]
    return X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]


@pytest.fixture(scope="module")
def fitted_rf(fold0):
    Xtr, ytr, *_ = fold0
    return build_rf_pipeline().fit(Xtr, ytr)


@pytest.fixture(scope="module")
def fitted_gb(fold0):
    Xtr, ytr, *_ = fold0
    return build_gb_pipeline().fit(Xtr, ytr)


# --- Random Forest tests ---
def test_rf_probability_shape_and_range(fitted_rf, fold0):
    *_, Xva, yva = fold0
    p = fitted_rf.predict_proba(Xva)
    assert p.shape == (len(Xva), 2)
    assert np.isfinite(p).all()
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert list(fitted_rf.named_steps["model"].classes_) == [0, 1]
    y = yva.to_numpy()
    assert p[y == 1, 1].mean() > p[y == 0, 1].mean()


def test_rf_predict_output(fitted_rf, fold0):
    *_, Xva, _ = fold0
    preds = fitted_rf.predict(Xva)
    assert preds.shape == (len(Xva),)
    assert set(preds) <= {0, 1}


def test_rf_reproducibility(fold0):
    Xtr, ytr, Xva, _ = fold0
    p1 = clone(build_rf_pipeline()).fit(Xtr, ytr).predict_proba(Xva)
    p2 = clone(build_rf_pipeline()).fit(Xtr, ytr).predict_proba(Xva)
    assert np.array_equal(p1, p2)


# --- Gradient Boosting tests ---
def test_gb_probability_shape_and_range(fitted_gb, fold0):
    *_, Xva, yva = fold0
    p = fitted_gb.predict_proba(Xva)
    assert p.shape == (len(Xva), 2)
    assert np.isfinite(p).all()
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert list(fitted_gb.named_steps["model"].classes_) == [0, 1]
    y = yva.to_numpy()
    assert p[y == 1, 1].mean() > p[y == 0, 1].mean()


def test_gb_predict_output(fitted_gb, fold0):
    *_, Xva, _ = fold0
    preds = fitted_gb.predict(Xva)
    assert preds.shape == (len(Xva),)
    assert set(preds) <= {0, 1}


def test_gb_reproducibility(fold0):
    Xtr, ytr, Xva, _ = fold0
    p1 = clone(build_gb_pipeline()).fit(Xtr, ytr).predict_proba(Xva)
    p2 = clone(build_gb_pipeline()).fit(Xtr, ytr).predict_proba(Xva)
    assert np.array_equal(p1, p2)
