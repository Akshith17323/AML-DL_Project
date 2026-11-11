import copy

from optuna_integration.sklearn import OptunaSearchCV

from src.data.loader import PHASE2_CONFIG_PATH, get_xy, load_all_cohorts, load_config
from src.evaluation.phase2_search import make_search, search_space
from src.models.pipelines import build_lr_pipeline


def test_every_search_parameter_exists_on_its_estimator():
    from src.models.phase2 import phase2_models
    cfg = load_config(PHASE2_CONFIG_PATH)
    for _, (key, estimator) in phase2_models(cfg).items():
        assert set(search_space(key)) <= set(estimator.get_params(deep=True))


def test_optuna_search_cv_runs_without_project_candidate_loop():
    cfg = copy.deepcopy(load_config(PHASE2_CONFIG_PATH)); cfg["cv"]["inner_folds"] = 2
    X, y = get_xy(load_all_cohorts(config=cfg)); X, y = X.iloc[:180], y.iloc[:180]
    search = make_search(build_lr_pipeline(cfg), "logistic_regression", cfg, n_trials=2, inner_folds=2)
    assert isinstance(search, OptunaSearchCV)
    search.fit(X, y)
    assert set(search.best_params_) == {"model__C"}
    assert len(search.study_.trials) == 2
    assert search.predict_proba(X.iloc[:5]).shape == (5, 2)
