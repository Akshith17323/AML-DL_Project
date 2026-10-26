import numpy as np
import pytest
from sklearn.base import clone

from src.data.loader import get_xy, load_cleveland
from src.evaluation.curves import plot_learning_curves
from src.evaluation.splits import outer_folds
from src.models.mlp import MLPClassifier
from src.models.pipelines import build_mlp_pipeline


@pytest.fixture(scope="module")
def data():
    return get_xy(load_cleveland())


@pytest.fixture(scope="module")
def fold0(data):
    X, y = data
    tr, va = outer_folds(y)[0]
    return X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]


@pytest.fixture(scope="module")
def fitted(fold0):
    Xtr, ytr, *_ = fold0
    return build_mlp_pipeline().fit(Xtr, ytr)


def test_fit_and_history(fitted):
    m = fitted.named_steps["model"]
    assert len(m.history_["train_loss"]) == len(m.history_["val_loss"]) == m.n_epochs_run_
    assert 0 <= m.best_epoch_ < m.n_epochs_run_
    assert m.history_["val_loss"][m.best_epoch_] == min(m.history_["val_loss"])
    assert m.n_features_in_ == 20 and m.n_parameters_ == 20 * 16 + 16 + 16 * 8 + 8 + 8 + 1
    assert m.history_["train_loss"][-1] < m.history_["train_loss"][0]  # it learns something


def test_probability_range_shape_and_semantics(fitted, fold0):
    *_, Xva, yva = fold0
    p = fitted.predict_proba(Xva)
    assert p.shape == (len(Xva), 2) and p.dtype == np.float64
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert np.allclose(p.sum(axis=1), 1)
    assert list(fitted.named_steps["model"].classes_) == [0, 1]
    y = yva.to_numpy()
    assert p[y == 1, 1].mean() > p[y == 0, 1].mean()  # column 1 = positive class
    assert fitted.predict(Xva).shape == (len(Xva),) and set(fitted.predict(Xva)) <= {0, 1}


def test_reproducible_same_seed(fold0):
    Xtr, ytr, Xva, _ = fold0
    a = clone(build_mlp_pipeline()).fit(Xtr, ytr).predict_proba(Xva)
    b = clone(build_mlp_pipeline()).fit(Xtr, ytr).predict_proba(Xva)
    assert np.array_equal(a, b)


def test_different_seed_differs(fold0):
    Xtr, ytr, Xva, _ = fold0
    a = build_mlp_pipeline().fit(Xtr, ytr).predict_proba(Xva)
    p = build_mlp_pipeline(); p.set_params(model__random_state=7)
    assert not np.allclose(a, p.fit(Xtr, ytr).predict_proba(Xva))


def test_early_stopping_triggers_and_restores_best():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4)); y = (X[:, 0] + 0.3 * rng.normal(size=120) > 0).astype(int)
    m = MLPClassifier(hidden_sizes=(16,), patience=3, max_epochs=200, learning_rate=0.05, dropout=0.0).fit(X, y)
    assert m.stopped_early_ and m.n_epochs_run_ < 200
    assert m.n_epochs_run_ == m.best_epoch_ + 1 + 3  # stopped exactly `patience` epochs after the best


def test_inner_validation_is_subset_of_training_data_only(fold0, monkeypatch):
    """fit() must only ever see the rows it is given (no access to the outer validation fold)."""
    Xtr, ytr, Xva, _ = fold0
    seen = {}
    import src.models.mlp as mod
    real = mod.train_test_split
    def spy(idx, **kw):
        seen["n"] = len(idx); return real(idx, **kw)
    monkeypatch.setattr(mod, "train_test_split", spy)
    build_mlp_pipeline().fit(Xtr, ytr)
    assert seen["n"] == len(Xtr)


def test_preprocessing_fitted_on_training_fold_only(fold0, fitted):
    Xtr, *_ = fold0
    sc = fitted.named_steps["preprocess"].named_steps["columns"].named_transformers_["continuous"].named_steps["scale"]
    assert np.allclose(sc.mean_, Xtr[["age", "trestbps", "chol", "thalach", "oldpeak"]].mean().to_numpy())


@pytest.mark.parametrize("kw", [
    {"hidden_sizes": ()}, {"hidden_sizes": (0,)}, {"activation": "swish"}, {"dropout": 1.0},
    {"dropout": -0.1}, {"weight_decay": -1}, {"learning_rate": 0}, {"batch_size": 0},
    {"patience": 0}, {"val_fraction": 0.0}, {"val_fraction": 0.6}])
def test_invalid_hyperparameters(kw):
    X = np.random.default_rng(0).normal(size=(40, 3)); y = np.arange(40) % 2
    with pytest.raises(ValueError):
        MLPClassifier(**kw).fit(X, y)


def test_invalid_inputs():
    X = np.random.default_rng(0).normal(size=(40, 3)); y = np.arange(40) % 2
    with pytest.raises(ValueError, match="NaN"):
        MLPClassifier().fit(np.vstack([X, [np.nan] * 3]), np.append(y, 0))
    with pytest.raises(ValueError, match="2-D"):
        MLPClassifier().fit(X[:, 0], y)
    with pytest.raises(ValueError, match="different lengths"):
        MLPClassifier().fit(X, y[:-1])
    with pytest.raises(ValueError, match="both classes"):
        MLPClassifier().fit(X, np.zeros(40, dtype=int))
    with pytest.raises(ValueError, match="only the values 0 and 1"):
        MLPClassifier().fit(X, y + 1)
    with pytest.raises(ValueError, match="numeric"):
        MLPClassifier().fit(np.array([["a", "b", "c"]] * 40), y)
    with pytest.raises(RuntimeError, match="not fitted"):
        MLPClassifier().predict_proba(X)
    m = MLPClassifier(max_epochs=2).fit(X, y)
    with pytest.raises(ValueError, match="Expected 3 features"):
        m.predict_proba(X[:, :2])
    with pytest.raises(ValueError, match="NaN"):
        m.predict_proba(np.full((2, 3), np.nan))


def test_pipeline_rejects_target_column(data):
    X, y = data
    with pytest.raises(ValueError, match="must not be passed"):
        build_mlp_pipeline().fit(X.assign(target=y), y)


def test_pipeline_handles_missing_ca_thal(fitted, data):
    X, _ = data
    rows = X[X[["ca", "thal"]].isna().any(axis=1)]
    p = fitted.predict_proba(rows)
    assert p.shape == (len(rows), 2) and np.isfinite(p).all()


def test_learning_curve_plot(fitted, tmp_path):
    m = fitted.named_steps["model"]
    out = plot_learning_curves([m.history_, m.history_], [m.best_epoch_] * 2, tmp_path / "c.png")
    assert out.stat().st_size > 0
