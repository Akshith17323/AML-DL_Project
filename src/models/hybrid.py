"""Cross-fitted Bayesian-network and MLP probability stack."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.models.mlp import TabularMLPClassifier
from src.models.pipelines import build_bn_pipeline


class BNMLPStackingClassifier(ClassifierMixin, BaseEstimator):
    """Combine out-of-fold BN/MLP probabilities with regularised logistic regression."""

    def __init__(self, bn_pseudo_count=2.0, mlp_hidden_layout="16-8", mlp_dropout=0.3,
                 mlp_weight_decay=1e-3, mlp_learning_rate=1e-3, mlp_batch_size=32,
                 mlp_max_epochs=150, mlp_patience=15, mlp_val_fraction=0.2,
                 meta_C=1.0, stacking_folds=3, random_state=42):
        self.bn_pseudo_count = bn_pseudo_count
        self.mlp_hidden_layout = mlp_hidden_layout
        self.mlp_dropout = mlp_dropout
        self.mlp_weight_decay = mlp_weight_decay
        self.mlp_learning_rate = mlp_learning_rate
        self.mlp_batch_size = mlp_batch_size
        self.mlp_max_epochs = mlp_max_epochs
        self.mlp_patience = mlp_patience
        self.mlp_val_fraction = mlp_val_fraction
        self.meta_C = meta_C
        self.stacking_folds = stacking_folds
        self.random_state = random_state

    def _bn(self):
        return build_bn_pipeline().set_params(model__pseudo_count=self.bn_pseudo_count)

    def _mlp(self):
        return TabularMLPClassifier(
            hidden_layout=self.mlp_hidden_layout, dropout=self.mlp_dropout,
            weight_decay=self.mlp_weight_decay, learning_rate=self.mlp_learning_rate,
            batch_size=self.mlp_batch_size, max_epochs=self.mlp_max_epochs,
            patience=self.mlp_patience, val_fraction=self.mlp_val_fraction,
            random_state=self.random_state)

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BNMLPStackingClassifier expects a pandas DataFrame")
        y = np.asarray(y, dtype=int)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("y must contain classes 0 and 1")
        if self.stacking_folds < 2:
            raise ValueError("stacking_folds must be at least 2")
        cv = StratifiedKFold(self.stacking_folds, shuffle=True, random_state=self.random_state)
        bn_oof = cross_val_predict(self._bn(), X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        mlp_oof = cross_val_predict(self._mlp(), X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        self.meta_features_ = np.column_stack([bn_oof, mlp_oof])
        if not np.isfinite(self.meta_features_).all():
            raise RuntimeError("Cross-fitted base probabilities are incomplete")
        self.meta_model_ = LogisticRegression(C=self.meta_C, solver="lbfgs", random_state=self.random_state)
        self.meta_model_.fit(self.meta_features_, y)
        self.bn_, self.mlp_ = clone(self._bn()).fit(X, y), clone(self._mlp()).fit(X, y)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = self.mlp_.n_features_in_
        return self

    def predict_base_proba(self, X) -> np.ndarray:
        if not hasattr(self, "meta_model_"):
            raise RuntimeError("Model is not fitted")
        return np.column_stack([self.bn_.predict_proba(X)[:, 1], self.mlp_.predict_proba(X)[:, 1]])

    def predict_proba(self, X) -> np.ndarray:
        return self.meta_model_.predict_proba(self.predict_base_proba(X))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class BNMLPAverageClassifier(BNMLPStackingClassifier):
    """Unlearned probability average used as the integration control."""

    def fit(self, X, y):
        self.bn_, self.mlp_ = clone(self._bn()).fit(X, y), clone(self._mlp()).fit(X, y)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = self.mlp_.n_features_in_
        return self

    def predict_base_proba(self, X) -> np.ndarray:
        if not hasattr(self, "bn_"):
            raise RuntimeError("Model is not fitted")
        return np.column_stack([self.bn_.predict_proba(X)[:, 1], self.mlp_.predict_proba(X)[:, 1]])

    def predict_proba(self, X) -> np.ndarray:
        p1 = self.predict_base_proba(X).mean(axis=1)
        return np.column_stack([1 - p1, p1])
