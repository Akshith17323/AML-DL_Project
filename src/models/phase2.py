"""Phase 2 model registry: six ML references, MLP, and BN–MLP integration."""
from __future__ import annotations

from src.models.hybrid import BNMLPStackingClassifier
from src.models.mlp import TabularMLPClassifier
from src.models.pipelines import (build_bn_pipeline, build_gb_pipeline, build_lr_pipeline,
                                  build_nb_pipeline, build_rf_pipeline, build_svm_pipeline)


def build_mlp(cfg: dict) -> TabularMLPClassifier:
    m = cfg["models"]["mlp"]
    return TabularMLPClassifier(**m, random_state=cfg["project"]["seed"])


def build_hybrid(cfg: dict) -> BNMLPStackingClassifier:
    m, h = cfg["models"]["mlp"], cfg["models"]["hybrid"]
    return BNMLPStackingClassifier(
        bn_pseudo_count=h["bn_pseudo_count"], meta_C=h["meta_C"],
        mlp_hidden_layout=m["hidden_layout"], mlp_dropout=m["dropout"],
        mlp_weight_decay=m["weight_decay"], mlp_learning_rate=m["learning_rate"],
        mlp_batch_size=m["batch_size"], mlp_max_epochs=m["max_epochs"],
        mlp_patience=m["patience"], mlp_val_fraction=m["val_fraction"],
        stacking_folds=cfg["cv"]["stacking_folds"], random_state=cfg["project"]["seed"])


def phase2_models(cfg: dict) -> dict:
    """Display name -> (search-space key, unfitted estimator)."""
    return {
        "Bayesian Network": ("bayesian_network", build_bn_pipeline(cfg)),
        "Naive Bayes": ("naive_bayes", build_nb_pipeline(cfg)),
        "Logistic Regression": ("logistic_regression", build_lr_pipeline(cfg)),
        "SVM": ("svm", build_svm_pipeline(cfg)),
        "Random Forest": ("random_forest", build_rf_pipeline(cfg)),
        "Gradient Boosting": ("gradient_boosting", build_gb_pipeline(cfg)),
        "MLP": ("mlp", build_mlp(cfg)),
        "BN-MLP Stack": ("hybrid", build_hybrid(cfg)),
    }
