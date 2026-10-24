"""Domain-informed Bayesian network classifier (pgmpy).

The structure below is a *domain-informed assumption*, written down before any
model was evaluated. Edges are NOT learned from data and are NOT evidence of
causation; only the conditional probability tables (CPDs) are estimated from the
training fold. The structure was kept small because n = 303: a node with many
parents has many sparsely populated CPD cells.

Direction convention: risk factors -> disease -> manifestations (test findings /
symptoms), i.e. a causal-style description of how findings arise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn.base import BaseEstimator, ClassifierMixin

from src.data.loader import BINARY_TARGET, FEATURE_COLUMNS

TARGET_NODE = BINARY_TARGET

# (parent, child, rationale, kind). kind: "domain" = textbook physiology/clinical
# reasoning taken as an assumption; "weak" = plausible but weakly motivated, kept
# so the node is connected; "modelling" = chosen to relax Naive Bayes independence.
EDGES = [
    ("age", "target", "Age is an established cardiovascular risk factor.", "domain"),
    ("sex", "target", "Sex is an established risk factor for coronary disease.", "domain"),
    ("chol", "target", "Serum cholesterol is an established risk factor.", "domain"),
    ("age", "trestbps", "Resting blood pressure tends to rise with age (blood pressure is not made a direct parent of target: too many CPD cells for n = 303).", "weak"),
    ("age", "fbs", "Fasting glucose tends to drift upward with age (weak; links fbs without enlarging the target CPD).", "weak"),
    ("target", "cp", "Chest pain type is a symptom manifestation of disease.", "domain"),
    ("target", "exang", "Exercise-induced angina is a manifestation of ischaemic disease.", "domain"),
    ("cp", "exang", "Both describe angina symptoms; relaxes the Naive Bayes independence assumption.", "modelling"),
    ("target", "restecg", "Resting ECG abnormalities can accompany disease.", "domain"),
    ("target", "thal", "Perfusion defects on the scan reflect disease.", "domain"),
    ("target", "ca", "Number of vessels with visible stenosis reflects disease extent.", "domain"),
    ("target", "thalach", "Reduced peak heart rate is a sign of reduced exercise capacity.", "domain"),
    ("age", "thalach", "Peak heart rate falls with age.", "domain"),
    ("target", "oldpeak", "Exercise ST depression is a diagnostic sign of ischaemia.", "domain"),
    ("target", "slope", "Peak-exercise ST slope is a diagnostic sign of ischaemia.", "domain"),
    ("oldpeak", "slope", "Both measure the exercise ST response (strongest predictor pair in EDA, Spearman 0.6).", "modelling"),
]
NODES = FEATURE_COLUMNS + [TARGET_NODE]


def build_structure() -> BayesianNetwork:
    model = BayesianNetwork([(a, b) for a, b, *_ in EDGES])
    model.add_nodes_from(NODES)
    return model


class BayesianNetworkClassifier(BaseEstimator, ClassifierMixin):
    """Fixed-structure BN. Input: integer-coded discrete DataFrame (see
    ``src.features.preprocessing.build_discrete_preprocessor``).

    CPDs are estimated with a Dirichlet prior (``pseudo_count`` per cell), from
    the data passed to ``fit`` only. ``predict_proba`` returns an (n, 2) array
    with columns ordered by ``classes_`` = [0, 1] (column 1 = positive class).

    missing:
      "error"      -> NaN in a predictor raises ValueError (default).
      "marginalize"-> NaN predictors are dropped from the evidence and summed out
                      exactly by variable elimination.
    Values outside a predictor's training states always raise ValueError; the
    model does not guess a replacement. Inference errors are not suppressed.
    """

    def __init__(self, pseudo_count: float = 1.0, missing: str = "error"):
        self.pseudo_count = pseudo_count
        self.missing = missing

    def fit(self, X, y):
        if self.missing not in ("error", "marginalize"):
            raise ValueError("missing must be 'error' or 'marginalize'")
        if self.pseudo_count <= 0:
            raise ValueError("pseudo_count must be > 0 so that every CPD cell is defined")
        X = self._check_frame(X)
        y = pd.Series(np.asarray(y), index=X.index, name=TARGET_NODE)
        if set(y.unique()) - {0, 1}:
            raise ValueError(f"y must be binary 0/1, got {sorted(y.unique())}")
        if X.isna().any().any():
            raise ValueError("Training data must not contain missing values")
        self.classes_ = np.array([0, 1])
        self.n_states_ = {c: int(X[c].max()) + 1 for c in FEATURE_COLUMNS}
        state_names = {c: list(range(k)) for c, k in self.n_states_.items()}
        state_names[TARGET_NODE] = [0, 1]
        data = X[FEATURE_COLUMNS].astype(int).assign(**{TARGET_NODE: y.astype(int)})
        model = build_structure()
        model.fit(data, estimator=BayesianEstimator, prior_type="dirichlet",
                  pseudo_counts=float(self.pseudo_count), state_names=state_names)
        if not model.check_model():
            raise RuntimeError("pgmpy check_model() failed after fitting")
        self.model_ = model
        self._infer = VariableElimination(model)
        return self

    def predict_proba(self, X) -> np.ndarray:
        self._require_fitted()
        X = self._check_frame(X)
        out = np.empty((len(X), 2))
        for i, (_, row) in enumerate(X[FEATURE_COLUMNS].iterrows()):
            evidence = {}
            for c, v in row.items():
                if pd.isna(v):
                    if self.missing == "error":
                        raise ValueError(f"Missing value in '{c}' (row {i}); use missing='marginalize' to sum it out")
                    continue
                if v != int(v) or not 0 <= int(v) < self.n_states_[c]:
                    raise ValueError(f"Unseen state {v!r} for '{c}' (row {i}); trained states: 0..{self.n_states_[c] - 1}")
                evidence[c] = int(v)
            factor = self._infer.query([TARGET_NODE], evidence=evidence, show_progress=False)
            p = np.array([factor.get_value(**{TARGET_NODE: s}) for s in (0, 1)], dtype=float)
            if not np.isfinite(p).all() or abs(p.sum() - 1) > 1e-8:
                raise RuntimeError(f"Posterior not a valid distribution for row {i}: {p}")
            out[i] = p
        return out

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def _check_frame(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BayesianNetworkClassifier expects a pandas DataFrame of coded predictors")
        absent = [c for c in FEATURE_COLUMNS if c not in X.columns]
        if absent:
            raise ValueError(f"Missing predictor columns: {absent}")
        if TARGET_NODE in X.columns or "num" in X.columns:
            raise ValueError("Target column(s) must not be passed as predictors")
        return X

    def _require_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted")


def edge_table() -> pd.DataFrame:
    return pd.DataFrame(EDGES, columns=["parent", "child", "rationale", "kind"])
