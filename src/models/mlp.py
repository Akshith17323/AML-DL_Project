"""Small feed-forward network for binary classification (PyTorch).

Architecture: inputs -> [Linear -> activation -> Dropout] x len(hidden_sizes) -> Linear(1).
One output logit; loss = BCEWithLogitsLoss; optimiser = Adam with L2 weight decay.
Positive class = 1: ``predict_proba`` returns (n, 2) with columns [P(0), P(1)].

Early stopping: a stratified ``val_fraction`` of the data passed to ``fit`` (the
outer *training* fold, never the outer validation/test fold) is held out; training
stops when its loss has not improved by ``min_delta`` for ``patience`` epochs and
the best-epoch weights are restored. Preprocessing is fitted upstream on the whole
training fold, so the inner validation rows contributed to scaling statistics
(not to any weights); the outer evaluation fold never touches fit.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}


def build_network(n_inputs: int, hidden_sizes, activation: str, dropout: float) -> nn.Sequential:
    layers, width = [], n_inputs
    for h in hidden_sizes:
        layers += [nn.Linear(width, h), ACTIVATIONS[activation](), nn.Dropout(dropout)]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_sizes=(16, 8), activation="relu", dropout=0.3, weight_decay=1e-3,
                 learning_rate=1e-3, batch_size=32, max_epochs=300, patience=20,
                 min_delta=1e-4, val_fraction=0.2, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.val_fraction = val_fraction
        self.random_state = random_state

    # ---- validation ----
    def _check_params(self):
        if not len(self.hidden_sizes) or any(int(h) < 1 for h in self.hidden_sizes):
            raise ValueError("hidden_sizes must be a non-empty sequence of positive ints")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.weight_decay < 0 or self.learning_rate <= 0:
            raise ValueError("weight_decay must be >= 0 and learning_rate > 0")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs and patience must be >= 1")
        if not 0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction must be in (0, 0.5)")

    @staticmethod
    def _as_array(X, n_features=None) -> np.ndarray:
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {arr.shape}")
        try:
            arr = arr.astype(np.float32)
        except (TypeError, ValueError) as e:
            raise ValueError("X must be numeric") from e
        if not np.isfinite(arr).all():
            raise ValueError("X contains NaN or infinite values; impute upstream")
        if n_features is not None and arr.shape[1] != n_features:
            raise ValueError(f"Expected {n_features} features, got {arr.shape[1]}")
        return arr

    # ---- fit / predict ----
    def fit(self, X, y):
        self._check_params()
        X = self._as_array(X)
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError("X and y have different lengths")
        if set(np.unique(y)) - {0, 1} or len(np.unique(y)) != 2:
            raise ValueError("y must contain both classes and only the values 0 and 1")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        seed = int(self.random_state)
        torch.manual_seed(seed)

        tr, va = train_test_split(np.arange(len(X)), test_size=self.val_fraction,
                                  stratify=y, random_state=seed)
        Xtr, ytr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr].astype(np.float32))
        Xva, yva = torch.from_numpy(X[va]), torch.from_numpy(y[va].astype(np.float32))

        net = build_network(X.shape[1], self.hidden_sizes, self.activation, self.dropout)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(seed)

        def eval_loss(Xt, yt):
            net.eval()
            with torch.no_grad():
                return float(loss_fn(net(Xt).squeeze(1), yt))

        history = {"train_loss": [], "val_loss": []}
        best, best_state, best_epoch, wait = np.inf, None, -1, 0
        for epoch in range(self.max_epochs):
            net.train()
            order = torch.randperm(len(Xtr), generator=gen)
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xtr[idx]).squeeze(1), ytr[idx])
                loss.backward()
                opt.step()
            history["train_loss"].append(eval_loss(Xtr, ytr))
            v = eval_loss(Xva, yva)
            history["val_loss"].append(v)
            if v < best - self.min_delta:
                best, best_state, best_epoch, wait = v, copy.deepcopy(net.state_dict()), epoch, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.model_ = net
        self.history_ = history
        self.best_epoch_ = best_epoch
        self.n_epochs_run_ = len(history["val_loss"])
        self.stopped_early_ = self.n_epochs_run_ < self.max_epochs
        self.n_parameters_ = sum(p.numel() for p in net.parameters())
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted")
        arr = self._as_array(X, self.n_features_in_)
        self.model_.eval()
        with torch.no_grad():
            p1 = torch.sigmoid(self.model_(torch.from_numpy(arr)).squeeze(1)).numpy().astype(np.float64)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
