"""Library-managed Phase 2 hyperparameter search (no hand-written candidate loop)."""
from __future__ import annotations

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna_integration.sklearn import OptunaSearchCV
from sklearn.model_selection import StratifiedKFold


def search_space(model_key: str) -> dict:
    spaces = {
        "bayesian_network": {"model__pseudo_count": FloatDistribution(0.5, 50.0, log=True)},
        "naive_bayes": {"model__alpha": FloatDistribution(0.5, 50.0, log=True)},
        "logistic_regression": {"model__C": FloatDistribution(1e-2, 10.0, log=True)},
        "svm": {"model__C": FloatDistribution(0.1, 10.0, log=True),
                "model__gamma": CategoricalDistribution(["scale", 0.01, 0.1])},
        "random_forest": {"model__n_estimators": IntDistribution(100, 400, step=100),
                          "model__max_depth": CategoricalDistribution([3, 5, None]),
                          "model__min_samples_leaf": IntDistribution(1, 4)},
        "gradient_boosting": {"model__learning_rate": FloatDistribution(0.02, 0.2, log=True),
                              "model__n_estimators": IntDistribution(50, 200, step=50),
                              "model__max_depth": IntDistribution(1, 3)},
        "mlp": {"hidden_layout": CategoricalDistribution(["16", "16-8", "32-16"]),
                "dropout": FloatDistribution(0.1, 0.5),
                "learning_rate": FloatDistribution(1e-4, 3e-3, log=True),
                "weight_decay": FloatDistribution(1e-5, 1e-2, log=True),
                "batch_size": CategoricalDistribution([16, 32])},
        "hybrid": {"bn_pseudo_count": FloatDistribution(0.5, 50.0, log=True),
                   "mlp_hidden_layout": CategoricalDistribution(["16", "16-8", "32-16"]),
                   "mlp_dropout": FloatDistribution(0.1, 0.5),
                   "mlp_learning_rate": FloatDistribution(1e-4, 3e-3, log=True),
                   "mlp_weight_decay": FloatDistribution(1e-5, 1e-2, log=True),
                   "meta_C": FloatDistribution(1e-2, 10.0, log=True)},
    }
    if model_key not in spaces:
        raise KeyError(f"No Phase 2 search space for {model_key!r}")
    return spaces[model_key]


def make_search(estimator, model_key: str, cfg: dict, *, seed: int | None = None,
                n_trials: int | None = None, inner_folds: int | None = None) -> OptunaSearchCV:
    """Create a TPE OptunaSearchCV; the library owns candidate iteration and CV."""
    seed = cfg["project"]["seed"] if seed is None else seed
    params = search_space(model_key)
    absent = sorted(set(params) - set(estimator.get_params(deep=True)))
    if absent:
        raise ValueError(f"Search parameters absent from {type(estimator).__name__}: {absent}")
    cv = StratifiedKFold(inner_folds or cfg["cv"]["inner_folds"], shuffle=True, random_state=seed)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    return OptunaSearchCV(
        estimator, params, cv=cv, scoring=cfg["search"]["scoring"],
        n_trials=n_trials or cfg["search"]["n_trials"], study=study,
        n_jobs=cfg["search"]["n_jobs"], refit=True, error_score="raise",
        return_train_score=False, verbose=0)
