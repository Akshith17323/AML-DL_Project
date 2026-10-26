"""Leakage-safe preprocessing pipelines.

Every data-dependent step (imputation values, scaling statistics, one-hot
categories, quantile bin edges, category vocabularies) is learned in ``fit``.
Build a pipeline with a builder below and fit it on a training fold only,
then call ``transform`` on validation/test data.

* ``build_numeric_preprocessor``  -> scaled numeric + one-hot features for Logistic Regression,
                                     SVM, Random Forest and Gradient Boosting.
* ``build_discrete_preprocessor`` -> integer-coded discrete states for the Bayesian
                                     Network and Naive Bayes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import BINARY_TARGET, FEATURE_COLUMNS, ORIGINAL_TARGET, load_config

TARGET_COLUMNS = (ORIGINAL_TARGET, BINARY_TARGET)


def feature_groups(config: dict | None = None) -> dict:
    """Feature-type groups from config.yaml, validated against the 13 predictors."""
    f = (config or load_config())["features"]
    groups = {k: list(f[k]) for k in ("continuous", "binary", "ordinal", "categorical")}
    flat = [c for g in groups.values() for c in g]
    if sorted(flat) != sorted(FEATURE_COLUMNS):
        raise ValueError("Feature groups must partition the 13 predictors exactly once")
    return groups


class FeatureGuard(BaseEstimator, TransformerMixin):
    """Rejects any target column and returns the 13 predictors in canonical order."""

    def fit(self, X, y=None):
        self._check(X)
        self.feature_names_in_ = np.array(FEATURE_COLUMNS)
        return self

    def transform(self, X):
        self._check(X)
        return X[FEATURE_COLUMNS]

    @staticmethod
    def _check(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Preprocessing expects a pandas DataFrame")
        leaked = [c for c in TARGET_COLUMNS if c in X.columns]
        if leaked:
            raise ValueError(f"Target column(s) {leaked} must not be passed to the preprocessor")
        missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing predictor columns: {missing}")

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_COLUMNS)


class DiscretePreprocessor(BaseEstimator, TransformerMixin):
    """Integer-coded discrete states for BN / Naive Bayes.

    Fit (training data only):
      * continuous columns: quantile bin edges (duplicate edges collapsed, so a
        column with many ties gets fewer bins);
      * all other columns: the set of observed values (vocabulary);
      * the training mode of each column (imputation value).
    Transform: missing values and values outside a column's training vocabulary
    are replaced by the training mode; continuous values are placed into the
    training bins (values beyond the edges fall into the first/last bin).
    Output codes are 0..k-1 per column; ``state_labels_`` documents each code.
    """

    def __init__(self, continuous=None, n_bins=3):
        self.continuous = continuous
        self.n_bins = n_bins

    def fit(self, X, y=None):
        X = self._frame(X)
        cont = list(self.continuous or [])
        self.columns_ = list(X.columns)
        self.bin_edges_, self.vocab_, self.mode_code_, self.state_labels_ = {}, {}, {}, {}
        for c in self.columns_:
            col = X[c]
            if c in cont:
                qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
                edges = np.unique(np.nanquantile(col.to_numpy(float), qs))
                self.bin_edges_[c] = edges
                codes = pd.Series(np.digitize(col.to_numpy(float), edges, right=True), index=col.index)
                codes[col.isna()] = np.nan
                self.state_labels_[c] = self._bin_labels(edges)
            else:
                vocab = np.sort(col.dropna().unique())
                self.vocab_[c] = vocab
                codes = col.map({v: i for i, v in enumerate(vocab)})
                self.state_labels_[c] = [str(v) for v in vocab]
            self.mode_code_[c] = int(codes.mode(dropna=True).iloc[0])
        return self

    def transform(self, X):
        X = self._frame(X)
        out = {}
        for c in self.columns_:
            col = X[c]
            if c in self.bin_edges_:
                codes = pd.Series(np.digitize(col.to_numpy(float), self.bin_edges_[c], right=True),
                                  index=col.index, dtype=float)
                codes[col.isna()] = np.nan
            else:
                codes = col.map({v: i for i, v in enumerate(self.vocab_[c])}).astype(float)
            out[c] = codes.fillna(self.mode_code_[c]).astype(int)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)

    @staticmethod
    def _frame(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DiscretePreprocessor expects a pandas DataFrame")
        return X

    @staticmethod
    def _bin_labels(edges):
        if len(edges) == 0:
            return ["all"]
        labels = [f"<= {edges[0]:g}"]
        labels += [f"({a:g}, {b:g}]" for a, b in zip(edges[:-1], edges[1:])]
        labels.append(f"> {edges[-1]:g}")
        return labels

    def n_states(self) -> dict:
        return {c: len(v) for c, v in self.state_labels_.items()}


def build_numeric_preprocessor(config: dict | None = None) -> Pipeline:
    """Guard -> per-type imputation/scaling/encoding. Output is a dense float array.

    continuous : median impute, standardise
    binary     : most-frequent impute, passthrough 0/1
    ordinal    : most-frequent impute, standardise (ca, slope kept as ordered numbers)
    categorical: most-frequent impute, one-hot (train categories; unseen -> all zeros)
    """
    g = feature_groups(config)
    ct = ColumnTransformer(
        [("continuous", Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), g["continuous"]),
         ("binary", SimpleImputer(strategy="most_frequent"), g["binary"]),
         ("ordinal", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("scale", StandardScaler())]), g["ordinal"]),
         ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                   ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                            sparse_output=False))]),
          g["categorical"])],
        remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("guard", FeatureGuard()), ("columns", ct)]).set_output(transform="pandas")


def build_discrete_preprocessor(config: dict | None = None) -> Pipeline:
    """Guard -> discretise continuous features and integer-code everything (BN / NB)."""
    cfg = config or load_config()
    d = cfg["features"]["discretization"]
    feature_groups(cfg)  # validates config
    return Pipeline([("guard", FeatureGuard()),
                     ("discretize", DiscretePreprocessor(continuous=d["columns"], n_bins=d["n_bins"]))])
