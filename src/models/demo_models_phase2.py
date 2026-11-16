"""Phase 2 demo model selection, offline fitting, and caching.

Final models are fitted on all 920 rows (Cleveland, Hungarian, Switzerland, Long Beach VA)
for interactive demonstration only. Hyperparameters are selected from the saved inner Optuna
TPE search results in `results/phase2/main/metrics/selected_hyperparameters_pooled.csv`.

Demo predictions are strictly separated from cross-validation performance claims.
Performance claims come exclusively from `results/phase2/main/metrics/`.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.data.loader import (
    FEATURE_COLUMNS,
    PHASE2_CONFIG_PATH,
    get_xy,
    load_all_cohorts,
    load_config,
    resolve,
)
from src.models.phase2 import phase2_models

PHASE2_MODEL_NAMES = [
    "Bayesian Network",
    "Naive Bayes",
    "Logistic Regression",
    "SVM",
    "Random Forest",
    "Gradient Boosting",
    "MLP",
    "BN-MLP Stack",
]

# Integer hyperparameter keys for explicit typing
INT_PARAM_KEYS = {
    "model__n_estimators",
    "model__max_depth",
    "model__min_samples_leaf",
    "batch_size",
    "max_epochs",
    "patience",
    "stacking_folds",
}


def cast_typed_params(params: dict[str, Any]) -> dict[str, Any]:
    """Ensure parameters match their expected Python types."""
    typed = {}
    for k, v in params.items():
        if v is None:
            typed[k] = None
        elif k in INT_PARAM_KEYS:
            typed[k] = int(v) if v is not None else None
        elif isinstance(v, (int, float)):
            typed[k] = float(v)
        else:
            typed[k] = str(v)
    return typed


def select_phase2_hyperparameters(cfg: dict | None = None) -> tuple[dict[str, dict[str, Any]], str]:
    """Select the best hyperparameters for each Phase 2 model from saved Optuna results.

    Returns:
        (params_by_model, source_description)
    """
    cfg = cfg or load_config(PHASE2_CONFIG_PATH)
    metrics_dir = resolve(cfg["paths"]["metrics"])
    sel_path = metrics_dir / "selected_hyperparameters_pooled.csv"

    out = {m: {} for m in PHASE2_MODEL_NAMES}
    if not sel_path.exists():
        return out, "default config (saved Optuna selections not found)"

    df_sel = pd.read_csv(sel_path)
    for model_name in PHASE2_MODEL_NAMES:
        group = df_sel[df_sel["model"] == model_name]
        if group.empty:
            continue
        best_row = group.loc[group["best_inner_neg_log_loss"].idxmax()]
        raw_params = json.loads(best_row["best_params"])
        out[model_name] = cast_typed_params(raw_params)

    return out, "best inner-validation score from Optuna TPE search (pooled nested CV)"


def save_typed_hyperparameters(params: dict[str, dict[str, Any]], out_path: Path | str | None = None) -> Path:
    """Save typed final hyperparameters to a JSON file."""
    if out_path is None:
        cfg = load_config(PHASE2_CONFIG_PATH)
        out_path = resolve(cfg["paths"]["metrics"]) / "final_hyperparameters.json"
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    return out_path


@dataclass
class FinalModelsPhase2:
    """Container for the eight fitted Phase 2 demo models.

    Demo predictions are for educational demonstration only. Cross-validation performance
    estimates are strictly separate and come from results/phase2/main/metrics/.
    """
    pipelines: dict[str, Any]
    train_X: pd.DataFrame
    train_y: pd.Series
    hyperparameters: dict[str, dict[str, Any]]
    hyperparameter_source: str
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def predict(self, row: pd.DataFrame) -> dict[str, float]:
        """Return P(heart disease) for all eight models."""
        with self._lock:
            return {m: float(p.predict_proba(row)[0, 1]) for m, p in self.pipelines.items()}


def fit_final_phase2_models(
    cfg: dict | None = None,
    save_models_path: Path | str | None = None,
    save_params_path: Path | str | None = None,
) -> FinalModelsPhase2:
    """Fit all eight Phase 2 models offline on the complete four-cohort dataset (920 rows)."""
    cfg = cfg or load_config(PHASE2_CONFIG_PATH)
    df = load_all_cohorts(config=cfg)
    X, y = get_xy(df)

    params_by_model, source = select_phase2_hyperparameters(cfg)
    save_typed_hyperparameters(params_by_model, save_params_path)

    base_models = phase2_models(cfg)
    fitted_pipelines = {}

    for model_name in PHASE2_MODEL_NAMES:
        if model_name not in base_models:
            continue
        _, base_est = base_models[model_name]
        pipe = clone(base_est)
        if params_by_model.get(model_name):
            pipe.set_params(**params_by_model[model_name])
        pipe.fit(X, y)
        fitted_pipelines[model_name] = pipe

    final_models = FinalModelsPhase2(
        pipelines=fitted_pipelines,
        train_X=X,
        train_y=y,
        hyperparameters=params_by_model,
        hyperparameter_source=source,
    )

    if save_models_path is None:
        save_models_path = resolve(cfg["paths"]["metrics"]) / "final_demo_models.joblib"
    else:
        save_models_path = Path(save_models_path)

    save_models_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_models, save_models_path)
    return final_models


def load_or_fit_phase2_demo_models(
    cfg: dict | None = None,
    cache_path: Path | str | None = None,
) -> FinalModelsPhase2:
    """Load cached final models from disk; if absent, fit and cache them."""
    cfg = cfg or load_config(PHASE2_CONFIG_PATH)
    if cache_path is None:
        cache_path = resolve(cfg["paths"]["metrics"]) / "final_demo_models.joblib"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists():
        try:
            loaded = joblib.load(cache_path)
            if isinstance(loaded, FinalModelsPhase2) and len(loaded.pipelines) == len(PHASE2_MODEL_NAMES):
                return loaded
        except Exception:
            pass

    return fit_final_phase2_models(cfg, save_models_path=cache_path)


if __name__ == "__main__":
    print("Selecting final hyperparameters and fitting Phase 2 demo models offline...")
    final = fit_final_phase2_models()
    print(f"Successfully fitted {len(final.pipelines)} models on {len(final.train_X)} patients.")
    print("Demo predictions are ready and cached.")
