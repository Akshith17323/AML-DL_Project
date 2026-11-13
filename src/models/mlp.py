"""Small regularised PyTorch MLP that owns leakage-safe raw-row preprocessing."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn

from src.data.loader import FEATURE_COLUMNS
from src.features.preprocessing import build_numeric_preprocessor


def parse_hidden_layout(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(v) for v in value.split("-") if v)
    except (AttributeError, ValueError) as exc:
        raise ValueError("hidden_layout must look like '16' or '16-8'") from exc
    if not sizes or any(v < 1 for v in sizes):
        raise ValueError("hidden_layout must contain positive layer sizes")
    return sizes


def build_network(n_inputs: int, hidden_layout: str, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = n_inputs
    for size in parse_hidden_layout(hidden_layout):
        layers.extend([nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)])
        width = size
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class TabularMLPClassifier(ClassifierMixin, BaseEstimator):
    """Binary MLP with preprocessing fitted before neither stopping nor test rows.

    Raw rows are split first. The preprocessor is fitted only on weight-training
    rows, then applied to the early-stopping rows.
    """

    def __init__(self, hidden_layout="16-8", dropout=0.3, weight_decay=1e-3,
                 learning_rate=1e-3, batch_size=32, max_epochs=150, patience=15,
                 min_delta=1e-4, val_fraction=0.2, early_stopping=True,
                 random_state=42):
        self.hidden_layout = hidden_layout
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.val_fraction = val_fraction
        self.early_stopping = early_stopping
        self.random_state = random_state

    def _validate_params(self):
        parse_hidden_layout(self.hidden_layout)
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.weight_decay < 0 or self.learning_rate <= 0:
            raise ValueError("weight_decay must be >= 0 and learning_rate must be > 0")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs and patience must be positive")
        if not 0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction must be between 0 and 0.5")

    @staticmethod
    def _frame(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TabularMLPClassifier expects a pandas DataFrame")
        missing = [c for c in FEATURE_COLUMNS if c not in X]
        if missing:
            raise ValueError(f"Missing predictor columns: {missing}")
        if {"num", "target"} & set(X):
            raise ValueError("Target column(s) must not be passed as predictors")
        return X[FEATURE_COLUMNS]

    def fit(self, X, y):
        self._validate_params()
        X = self._frame(X)
        y = np.asarray(y, dtype=int)
        if len(X) != len(y) or set(np.unique(y)) != {0, 1}:
            raise ValueError("X and y must have equal length and y must contain classes 0 and 1")

        indices = np.arange(len(X))
        train_idx, stop_idx = train_test_split(indices, test_size=self.val_fraction,
                                                stratify=y, random_state=self.random_state)
        self.preprocessor_ = build_numeric_preprocessor().fit(X.iloc[train_idx])
        x_train = self.preprocessor_.transform(X.iloc[train_idx]).to_numpy(dtype=np.float32)
        x_stop = self.preprocessor_.transform(X.iloc[stop_idx]).to_numpy(dtype=np.float32)
        y_train = y[train_idx].astype(np.float32)
        y_stop = y[stop_idx].astype(np.float32)

        torch.manual_seed(int(self.random_state))
        model = build_network(x_train.shape[1], self.hidden_layout, self.dropout)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(int(self.random_state))
        xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train)
        xv, yv = torch.from_numpy(x_stop), torch.from_numpy(y_stop)

        def loss_on(features, labels) -> float:
            model.eval()
            with torch.no_grad():
                return float(loss_fn(model(features).squeeze(1), labels))

        history = {"train_loss": [], "val_loss": []}
        best_loss, best_epoch = np.inf, -1
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
        for epoch in range(int(self.max_epochs)):
            model.train()
            order = torch.randperm(len(xt), generator=generator)
            for start in range(0, len(order), int(self.batch_size)):
                batch = order[start:start + int(self.batch_size)]
                optimiser.zero_grad()
                loss = loss_fn(model(xt[batch]).squeeze(1), yt[batch])
                loss.backward()
                optimiser.step()
            train_loss, val_loss = loss_on(xt, yt), loss_on(xv, yv)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if not self.early_stopping:
                best_state, best_epoch = copy.deepcopy(model.state_dict()), epoch
                continue
            if val_loss < best_loss - self.min_delta:
                best_loss, best_epoch = val_loss, epoch
                best_state, wait = copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.array(FEATURE_COLUMNS)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.n_transformed_features_ = x_train.shape[1]
        self.n_parameters_ = sum(p.numel() for p in model.parameters())
        self.n_weight_train_, self.n_stopping_ = len(train_idx), len(stop_idx)
        self.weight_train_indices_, self.stopping_indices_ = train_idx, stop_idx
        self.history_, self.best_epoch_ = history, best_epoch
        self.n_epochs_run_ = len(history["train_loss"])
        self.stopped_early_ = bool(self.early_stopping and self.n_epochs_run_ < self.max_epochs)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted")
        arr = self.preprocessor_.transform(self._frame(X)).to_numpy(dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            p1 = torch.sigmoid(self.model_(torch.from_numpy(arr)).squeeze(1)).numpy().astype(float)
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
