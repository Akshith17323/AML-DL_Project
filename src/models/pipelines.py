"""Full preprocessing + model pipelines. Fit on a training fold only."""
from __future__ import annotations

import math

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data.loader import load_config
from src.features.preprocessing import build_discrete_preprocessor, build_numeric_preprocessor
from src.models.bayesian_network import BayesianNetworkClassifier
from src.models.naive_bayes import NaiveBayesClassifier


def build_bn_pipeline(config: dict | None = None) -> Pipeline:
    cfg = config or load_config()
    m = cfg["models"]["bayesian_network"]
    return Pipeline([("preprocess", build_discrete_preprocessor(cfg)),
                     ("model", BayesianNetworkClassifier(pseudo_count=m["pseudo_count"], missing=m["missing"]))])


def build_nb_pipeline(config: dict | None = None) -> Pipeline:
    cfg = config or load_config()
    return Pipeline([("preprocess", build_discrete_preprocessor(cfg)),
                     ("model", NaiveBayesClassifier(alpha=cfg["models"]["naive_bayes"]["alpha"]))])


def build_lr_pipeline(config: dict | None = None) -> Pipeline:
    """Logistic Regression with L2 regularisation (strength 1/C) on standardised, one-hot features."""
    cfg = config or load_config()
    p = dict(cfg["models"]["logistic_regression"])
    return Pipeline([("preprocess", build_numeric_preprocessor(cfg)),
                     ("model", LogisticRegression(random_state=cfg["project"]["seed"], solver="lbfgs", max_iter=2000, **p))])


def build_svm_pipeline(config: dict | None = None) -> Pipeline:
    """RBF-kernel SVM. ``probability=True`` adds Platt scaling so the model outputs probabilities."""
    cfg = config or load_config()
    p = dict(cfg["models"]["svm"])
    return Pipeline([("preprocess", build_numeric_preprocessor(cfg)),
                     ("model", SVC(kernel="rbf", probability=True, random_state=cfg["project"]["seed"], **p))])


def build_rf_pipeline(config: dict | None = None) -> Pipeline:
    cfg = config or load_config()
    p = dict(cfg["models"]["random_forest"])
    return Pipeline([("preprocess", build_numeric_preprocessor(cfg)),
                     ("model", RandomForestClassifier(random_state=cfg["project"]["seed"], **p))])


def build_gb_pipeline(config: dict | None = None) -> Pipeline:
    cfg = config or load_config()
    p = dict(cfg["models"]["gradient_boosting"])
    return Pipeline([("preprocess", build_numeric_preprocessor(cfg)),
                     ("model", GradientBoostingClassifier(random_state=cfg["project"]["seed"], **p))])


# display name -> (config key, builder); the order used in every table and figure
MODELS = {"Bayesian Network": ("bayesian_network", build_bn_pipeline),
          "Naive Bayes": ("naive_bayes", build_nb_pipeline),
          "Logistic Regression": ("logistic_regression", build_lr_pipeline),
          "SVM": ("svm", build_svm_pipeline),
          "Random Forest": ("random_forest", build_rf_pipeline),
          "Gradient Boosting": ("gradient_boosting", build_gb_pipeline)}


def _canonical(v) -> str:
    """Comparable form of a grid value or of the same value after a CSV round trip ("0.01", 200.0, NaN...)."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v) in ("nan", "None"):
        return "None"
    try:
        return repr(float(v))
    except (TypeError, ValueError):
        return str(v)


def params_from_selection(row, grid: dict) -> dict:
    """Turn one saved row of selected_hyperparameters.csv back into the original grid values.

    CSV round-trips lose types (200 -> 200.0, None -> NaN, 0.01 -> "0.01" in a column that also holds "scale"),
    so each value is matched to the grid entry it came from."""
    out = {}
    for k, options in grid.items():
        match = [o for o in options if _canonical(o) == _canonical(row[k])]
        if not match:
            raise ValueError(f"{k}={row[k]!r} is not in the grid {options}")
        out[k] = match[0]
    return out
