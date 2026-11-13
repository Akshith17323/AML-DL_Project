import numpy as np
from sklearn.base import clone

from src.data.loader import get_xy, load_all_cohorts
from src.models.hybrid import BNMLPAverageClassifier, BNMLPStackingClassifier
from src.models.mlp import TabularMLPClassifier, parse_hidden_layout


def _data(n=180):
    X, y = get_xy(load_all_cohorts())
    return X.iloc[:n], y.iloc[:n]


def test_mlp_fits_predicts_and_keeps_stopping_rows_out_of_preprocessing():
    X, y = _data()
    model = TabularMLPClassifier(max_epochs=6, patience=2, batch_size=32, random_state=7).fit(X, y)
    p = model.predict_proba(X.iloc[:11])
    assert p.shape == (11, 2) and np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
    assert not set(model.weight_train_indices_) & set(model.stopping_indices_)
    age_mean = X.iloc[model.weight_train_indices_]["age"].mean()
    scaler = model.preprocessor_.named_steps["columns"].named_transformers_["continuous"].named_steps["scale"]
    assert scaler.mean_[0] == age_mean
    assert model.n_parameters_ > 0 and len(model.history_["train_loss"]) == model.n_epochs_run_


def test_mlp_is_cloneable_and_reproducible():
    X, y = _data(140)
    base = TabularMLPClassifier(hidden_layout="16", max_epochs=4, patience=2, random_state=5)
    p1 = clone(base).fit(X, y).predict_proba(X.iloc[:10])
    p2 = clone(base).fit(X, y).predict_proba(X.iloc[:10])
    assert np.array_equal(p1, p2) and parse_hidden_layout("16-8") == (16, 8)


def test_hybrid_and_average_return_valid_probabilities():
    X, y = _data(120)
    kwargs = dict(mlp_max_epochs=3, mlp_patience=2, stacking_folds=2, random_state=3)
    stack = BNMLPStackingClassifier(**kwargs).fit(X, y)
    average = BNMLPAverageClassifier(**kwargs).fit(X, y)
    for model in (stack, average):
        p = model.predict_proba(X.iloc[:8])
        assert p.shape == (8, 2) and np.allclose(p.sum(axis=1), 1)
    assert stack.meta_features_.shape == (len(X), 2)
    assert np.isfinite(stack.meta_features_).all()
