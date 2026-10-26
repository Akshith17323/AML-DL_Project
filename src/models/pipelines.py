"""Full preprocessing + model pipelines. Fit on a training fold only."""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.data.loader import load_config
from src.features.preprocessing import build_discrete_preprocessor
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
    from src.features.preprocessing import build_mlp_preprocessor
    from src.models.mlp import MLPClassifier
    cfg = config or load_config()
    p = dict(cfg["models"]["mlp"])
    p["hidden_sizes"] = tuple(p["hidden_sizes"])
    return Pipeline([("preprocess", build_mlp_preprocessor(cfg)),
                     ("model", MLPClassifier(random_state=cfg["project"]["seed"], **p))])
