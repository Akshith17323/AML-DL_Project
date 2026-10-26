"""Unit tests for Logistic Regression and SVM pipelines (Hardik).

Both models use the continuous + one-hot numeric preprocessor and must expose
``predict_proba`` (Platt scaling for SVM).  Each test mirrors the ensemble
model tests added by Akshith so the test suite is symmetric.
"""
import numpy as np
import pytest
from sklearn.base import clone

from src.data.loader import get_xy, load_cleveland
from src.evaluation.splits import outer_folds
from src.models.pipelines import build_lr_pipeline, build_svm_pipeline


@pytest.fixture(scope="module")
def data():
    return get_xy(load_cleveland())


@pytest.fixture(scope="module")
def fold0(data):
    X, y = data
    tr, va = outer_folds(y)[0]
    return X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]


@pytest.fixture(scope="module")
def fitted_lr(fold0):
    Xtr, ytr, *_ = fold0
    return build_lr_pipeline().fit(Xtr, ytr)


@pytest.fixture(scope="module")
def fitted_svm(fold0):
    Xtr, ytr, *_ = fold0
    return build_svm_pipeline().fit(Xtr, ytr)


# ------------------------------------------------------------------ LR tests

def test_lr_probability_shape_and_range(fitted_lr, fold0):
    """predict_proba returns (n, 2) array with valid probabilities."""
    *_, Xva, yva = fold0
    p = fitted_lr.predict_proba(Xva)
    assert p.shape == (len(Xva), 2)
    assert np.isfinite(p).all(), "LR predicted probabilities contain non-finite values"
    assert ((p >= 0.0) & (p <= 1.0)).all(), "LR probabilities outside [0, 1]"
    assert np.allclose(p.sum(axis=1), 1.0), "LR class probabilities do not sum to 1"
    assert list(fitted_lr.named_steps["model"].classes_) == [0, 1]
    # Disease-positive patients should have higher mean predicted risk
    y = yva.to_numpy()
    assert p[y == 1, 1].mean() > p[y == 0, 1].mean(), (
        "LR assigns higher mean risk to healthy than diseased patients"
    )


def test_lr_predict_output(fitted_lr, fold0):
    """predict returns binary labels only."""
    *_, Xva, _ = fold0
    preds = fitted_lr.predict(Xva)
    assert preds.shape == (len(Xva),)
    assert set(preds) <= {0, 1}, f"LR predicted unexpected classes: {set(preds)}"


def test_lr_reproducibility(fold0):
    """Two pipeline instances with same seed give identical predictions."""
    Xtr, ytr, Xva, _ = fold0
    p1 = clone(build_lr_pipeline()).fit(Xtr, ytr).predict_proba(Xva)
    p2 = clone(build_lr_pipeline()).fit(Xtr, ytr).predict_proba(Xva)
    assert np.array_equal(p1, p2), "LR is not reproducible with the same random_state"


def test_lr_uses_regularisation(fold0):
    """Changing C changes predictions, confirming the L2 penalty is active."""
    Xtr, ytr, Xva, _ = fold0
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from src.features.preprocessing import build_numeric_preprocessor
    pipe_strong = Pipeline([("preprocess", build_numeric_preprocessor()),
                             ("model", LogisticRegression(C=0.001, solver="lbfgs", max_iter=2000))])
    pipe_weak = Pipeline([("preprocess", build_numeric_preprocessor()),
                          ("model", LogisticRegression(C=100.0, solver="lbfgs", max_iter=2000))])
    p_strong = pipe_strong.fit(Xtr, ytr).predict_proba(Xva)
    p_weak = pipe_weak.fit(Xtr, ytr).predict_proba(Xva)
    assert not np.allclose(p_strong, p_weak), (
        "C=0.001 and C=100 give identical predictions — regularisation may not be active"
    )


# ------------------------------------------------------------------ SVM tests

def test_svm_probability_shape_and_range(fitted_svm, fold0):
    """SVM with Platt scaling returns valid probability estimates."""
    *_, Xva, yva = fold0
    p = fitted_svm.predict_proba(Xva)
    assert p.shape == (len(Xva), 2)
    assert np.isfinite(p).all(), "SVM predicted probabilities contain non-finite values"
    assert ((p >= 0.0) & (p <= 1.0)).all(), "SVM probabilities outside [0, 1]"
    assert np.allclose(p.sum(axis=1), 1.0), "SVM class probabilities do not sum to 1"
    assert list(fitted_svm.named_steps["model"].classes_) == [0, 1]
    y = yva.to_numpy()
    assert p[y == 1, 1].mean() > p[y == 0, 1].mean(), (
        "SVM assigns higher mean risk to healthy than diseased patients"
    )


def test_svm_predict_output(fitted_svm, fold0):
    """predict returns binary labels only."""
    *_, Xva, _ = fold0
    preds = fitted_svm.predict(Xva)
    assert preds.shape == (len(Xva),)
    assert set(preds) <= {0, 1}, f"SVM predicted unexpected classes: {set(preds)}"


def test_svm_reproducibility(fold0):
    """Two SVM pipelines with the same seed give identical predictions."""
    Xtr, ytr, Xva, _ = fold0
    p1 = clone(build_svm_pipeline()).fit(Xtr, ytr).predict_proba(Xva)
    p2 = clone(build_svm_pipeline()).fit(Xtr, ytr).predict_proba(Xva)
    assert np.array_equal(p1, p2), "SVM is not reproducible with the same random_state"


def test_svm_kernel_is_rbf(fitted_svm):
    """Pipeline uses an RBF kernel (not linear/poly)."""
    svc = fitted_svm.named_steps["model"]
    assert svc.kernel == "rbf", f"Expected rbf kernel, got {svc.kernel}"


def test_svm_probability_enabled(fitted_svm):
    """SVC is initialised with probability=True (Platt scaling)."""
    svc = fitted_svm.named_steps["model"]
    assert svc.probability is True, "SVC must have probability=True for predict_proba / calibration"
