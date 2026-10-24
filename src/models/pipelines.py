"""Full preprocessing + model pipelines. Fit on a training fold only."""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data.loader import load_config
from src.features.preprocessing import build_discrete_preprocessor, build_mlp_preprocessor
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


def build_mlp_pipeline(config: dict | None = None) -> Pipeline:
    from src.models.mlp import MLPClassifier
    cfg = config or load_config()
    p = dict(cfg["models"]["mlp"])
    p["hidden_sizes"] = tuple(p["hidden_sizes"])
    return Pipeline([("preprocess", build_mlp_preprocessor(cfg)),
                     ("model", MLPClassifier(random_state=cfg["project"]["seed"], **p))])


def build_rf_pipeline(config: dict | None = None) -> Pipeline:
    """Random Forest pipeline with continuous standardisation and one-hot encoding."""
    cfg = config or load_config()
    p = dict(cfg.get("models", {}).get("random_forest", {}))
    return Pipeline([("preprocess", build_mlp_preprocessor(cfg)),
                     ("model", RandomForestClassifier(random_state=cfg["project"]["seed"], **p))])


def build_gb_pipeline(config: dict | None = None) -> Pipeline:
    """Gradient Boosting pipeline with continuous standardisation and one-hot encoding."""
    cfg = config or load_config()
    p = dict(cfg.get("models", {}).get("gradient_boosting", {}))
    return Pipeline([("preprocess", build_mlp_preprocessor(cfg)),
                     ("model", GradientBoostingClassifier(random_state=cfg["project"]["seed"], **p))])


def build_lr_pipeline(config: dict | None = None) -> Pipeline:
    """Logistic Regression pipeline with standardisation and one-hot encoding.

    Uses L2 regularisation (penalty='l2') via the ``C`` parameter read from
    config.yaml (``models.logistic_regression.C``).  The solver is 'lbfgs'
    which supports L2 and multi-class natively; ``max_iter`` is set high so the
    solver always converges on the small Cleveland dataset.
    """
    cfg = config or load_config()
    p = dict(cfg.get("models", {}).get("logistic_regression", {}))
    return Pipeline([("preprocess", build_mlp_preprocessor(cfg)),
                     ("model", LogisticRegression(
                         random_state=cfg["project"]["seed"],
                         solver="lbfgs",
                         max_iter=2000,
                         **p))])


def build_svm_pipeline(config: dict | None = None) -> Pipeline:
    """SVM pipeline with standardisation and one-hot encoding.

    Uses an RBF kernel with Platt scaling (``probability=True``) so that
    ``predict_proba`` / calibration metrics work.  ``C`` and ``gamma`` are
    read from config.yaml (``models.svm.*``).
    """
    cfg = config or load_config()
    p = dict(cfg.get("models", {}).get("svm", {}))
    return Pipeline([("preprocess", build_mlp_preprocessor(cfg)),
                     ("model", SVC(
                         kernel="rbf",
                         probability=True,  # Platt scaling: needed for predict_proba / calibration
                         random_state=cfg["project"]["seed"],
                         **p))])
