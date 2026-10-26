import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.data.loader import FEATURE_COLUMNS, get_xy, load_cleveland
from src.features.preprocessing import (DiscretePreprocessor, build_discrete_preprocessor,
                                        build_numeric_preprocessor, feature_groups)


@pytest.fixture(scope="module")
def data():
    df = load_cleveland()
    return df, *get_xy(df)


def test_feature_groups_partition_predictors():
    g = feature_groups()
    assert sorted(sum(g.values(), [])) == sorted(FEATURE_COLUMNS)


def test_numeric_no_nan_and_dimensions(data):
    _, X, y = data
    out = build_numeric_preprocessor().fit_transform(X)
    assert not out.isna().any().any()
    # 5 continuous + 3 binary + 2 ordinal + onehot(cp 4 + restecg 3 + thal 3)
    assert out.shape == (303, 5 + 3 + 2 + 4 + 3 + 3)
    assert "cp_4.0" in out.columns and "thal_7.0" in out.columns


def test_discrete_no_nan_and_codes(data):
    _, X, _ = data
    pre = build_discrete_preprocessor().fit(X)
    out = pre.transform(X)
    assert out.shape == (303, 13) and not out.isna().any().any()
    assert list(out.columns) == FEATURE_COLUMNS
    n_states = pre.named_steps["discretize"].n_states()
    for c in FEATURE_COLUMNS:
        assert out[c].min() == 0 and out[c].max() == n_states[c] - 1
    assert all(n_states[c] <= 3 for c in ["age", "trestbps", "chol", "thalach", "oldpeak"])


def test_missing_values_imputed_with_train_mode(data):
    _, X, _ = data
    train = X.iloc[:200]
    mlp = build_numeric_preprocessor().fit(train)
    row = X.iloc[[0]].copy()
    row[["ca", "thal"]] = np.nan
    out = mlp.transform(row)
    assert not out.isna().any().any()
    mode_thal = train["thal"].mode().iloc[0]
    assert out[f"thal_{mode_thal}"].iloc[0] == 1.0
    disc = build_discrete_preprocessor().fit(train)
    d = disc.transform(row)
    dp = disc.named_steps["discretize"]
    assert d["ca"].iloc[0] == dp.mode_code_["ca"] and d["thal"].iloc[0] == dp.mode_code_["thal"]


def test_unseen_category_handled(data):
    _, X, _ = data
    mlp = build_numeric_preprocessor().fit(X)
    row = X.iloc[[0]].copy(); row["cp"] = 99.0
    out = mlp.transform(row)
    assert out[[c for c in out.columns if c.startswith("cp_")]].to_numpy().sum() == 0
    disc = build_discrete_preprocessor().fit(X)
    d = disc.transform(row)
    assert d["cp"].iloc[0] == disc.named_steps["discretize"].mode_code_["cp"]


def test_extreme_values_clip_to_end_bins(data):
    _, X, _ = data
    disc = build_discrete_preprocessor().fit(X)
    row = X.iloc[[0]].copy(); row["age"] = 500.0; row["chol"] = -5.0
    d = disc.transform(row)
    n = disc.named_steps["discretize"].n_states()
    assert d["age"].iloc[0] == n["age"] - 1 and d["chol"].iloc[0] == 0


@pytest.mark.parametrize("bad", ["num", "target"])
@pytest.mark.parametrize("builder", [build_numeric_preprocessor, build_discrete_preprocessor])
def test_target_column_rejected(data, bad, builder):
    df, X, y = data
    X2 = X.copy(); X2[bad] = df[bad]
    with pytest.raises(ValueError, match="must not be passed"):
        builder().fit(X2)
    with pytest.raises(ValueError, match="must not be passed"):
        builder().fit(X).transform(X2)


def test_output_has_no_target_columns(data):
    _, X, _ = data
    for pre in (build_numeric_preprocessor(), build_discrete_preprocessor()):
        out = pre.fit_transform(X)
        assert not {"num", "target"} & set(out.columns)


def test_statistics_come_from_training_data_only(data):
    _, X, _ = data
    train = X.iloc[:200]
    mlp = build_numeric_preprocessor().fit(train)
    scaler = mlp.named_steps["columns"].named_transformers_["continuous"].named_steps["scale"]
    assert np.allclose(scaler.mean_, train[["age", "trestbps", "chol", "thalach", "oldpeak"]].mean().to_numpy())
    assert not np.allclose(scaler.mean_, X[["age", "trestbps", "chol", "thalach", "oldpeak"]].mean().to_numpy())
    disc = build_discrete_preprocessor().fit(train).named_steps["discretize"]
    ref = DiscretePreprocessor(["age"], 3).fit(train[FEATURE_COLUMNS])
    assert np.array_equal(disc.bin_edges_["age"], ref.bin_edges_["age"])


def test_transforming_validation_does_not_change_fit(data):
    _, X, _ = data
    train, val = X.iloc[:200], X.iloc[200:].copy()
    val["chol"] = 10_000.0
    pre = build_numeric_preprocessor().fit(train)
    before = pre.named_steps["columns"].named_transformers_["continuous"].named_steps["scale"].mean_.copy()
    pre.transform(val)
    after = pre.named_steps["columns"].named_transformers_["continuous"].named_steps["scale"].mean_
    assert np.array_equal(before, after)


def test_refit_in_each_fold_differs_and_clone_is_unfitted(data):
    _, X, y = data
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    means = []
    base = build_numeric_preprocessor()
    for tr, va in skf.split(X, y):
        pre = clone(base).fit(X.iloc[tr])
        assert not hasattr(clone(base).named_steps["columns"], "transformers_")
        means.append(pre.named_steps["columns"].named_transformers_["continuous"].named_steps["scale"].mean_[0])
        assert pre.transform(X.iloc[va]).shape[0] == len(va)
    assert len(set(np.round(means, 6))) > 1


def test_feature_names_preserved(data):
    _, X, _ = data
    out = build_numeric_preprocessor().fit_transform(X)
    assert list(out.columns[:5]) == ["age", "trestbps", "chol", "thalach", "oldpeak"]
    assert list(build_numeric_preprocessor().fit(X).get_feature_names_out()) == list(out.columns)
