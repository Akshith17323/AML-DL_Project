"""Categorical Naive Bayes baseline.

Model: P(y | x) is proportional to P(y) * prod_j P(x_j | y), i.e. all 13
predictors are assumed conditionally independent given the class. This is
violated here (e.g. oldpeak/slope, cp/exang, thalach/age), which tends to make
posterior probabilities over-confident even when ranking is fine; calibration
is therefore evaluated explicitly later. Inputs are the same integer-coded
discrete states as the Bayesian network, with Laplace smoothing ``alpha``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import CategoricalNB

from src.data.loader import BINARY_TARGET, FEATURE_COLUMNS


class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    """predict_proba -> (n, 2), columns ordered by ``classes_`` = [0, 1]."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = self._check_frame(X)
        if X.isna().any().any():
            raise ValueError("Training data must not contain missing values")
        y = np.asarray(y)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError(f"y must be binary 0/1, got {sorted(np.unique(y))}")
        if len(np.unique(y)) != 2:
            raise ValueError("Both classes must be present in the training data")
        self.n_states_ = {c: int(X[c].max()) + 1 for c in FEATURE_COLUMNS}
        self.model_ = CategoricalNB(alpha=self.alpha, min_categories=[self.n_states_[c] for c in FEATURE_COLUMNS])
        self.model_.fit(X[FEATURE_COLUMNS].astype(int).to_numpy(), y)
        self.classes_ = self.model_.classes_
        if list(self.classes_) != [0, 1]:
            raise RuntimeError(f"Unexpected class order {self.classes_}")
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted")
        X = self._check_frame(X)
        if X.isna().any().any():
            raise ValueError("Predictors must not contain missing values (impute upstream)")
        arr = X[FEATURE_COLUMNS].to_numpy()
        for j, c in enumerate(FEATURE_COLUMNS):
            bad = (arr[:, j] != arr[:, j].astype(int)) | (arr[:, j] < 0) | (arr[:, j] >= self.n_states_[c])
            if bad.any():
                raise ValueError(f"Unseen state(s) for '{c}': {np.unique(arr[bad, j])}; trained states 0..{self.n_states_[c] - 1}")
        return self.model_.predict_proba(arr.astype(int))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @staticmethod
    def _check_frame(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NaiveBayesClassifier expects a pandas DataFrame of coded predictors")
        absent = [c for c in FEATURE_COLUMNS if c not in X.columns]
        if absent:
            raise ValueError(f"Missing predictor columns: {absent}")
        if BINARY_TARGET in X.columns or "num" in X.columns:
            raise ValueError("Target column(s) must not be passed as predictors")
        return X
