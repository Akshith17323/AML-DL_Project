"""Input validation, final-model training and prediction for the web interface.

The "final" models are fitted on all 303 rows for demonstration only. Their hyper-parameters are the
combination selected most often by the inner cross-validation of the nested evaluation (no new tuning).
Nothing here evaluates performance; performance claims come from results/metrics/ only.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.data.eda import DOMAINS, PLAUSIBLE
from src.data.loader import FEATURE_COLUMNS, get_xy, load_cleveland, load_config, resolve
from src.models.pipelines import build_bn_pipeline, build_mlp_pipeline, build_nb_pipeline

MODEL_NAMES = ["Bayesian Network", "Naive Bayes", "MLP"]
_CONFIG_KEY = {"Bayesian Network": "bayesian_network", "Naive Bayes": "naive_bayes", "MLP": "mlp"}
_BUILDERS = {"Bayesian Network": build_bn_pipeline, "Naive Bayes": build_nb_pipeline, "MLP": build_mlp_pipeline}

LABELS = {
    "age": "Age (years)", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)", "fbs": "Fasting blood sugar > 120 mg/dl", "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved", "exang": "Exercise-induced angina", "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak-exercise ST segment", "ca": "Major vessels coloured by fluoroscopy (0-3)", "thal": "Thallium scan result",
}
OPTIONS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "No (<= 120 mg/dl)", 1: "Yes (> 120 mg/dl)"},
    "restecg": {0: "Normal", 1: "ST-T wave abnormality", 2: "Probable/definite LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}
OPTIONAL = ("ca", "thal")  # the only fields with missing values in the study data; filled with the training mode
CONTINUOUS = list(PLAUSIBLE)
STEP = {"age": 1, "trestbps": 1, "chol": 1, "thalach": 1, "oldpeak": 0.1}


def validate_inputs(values: dict, train_X: pd.DataFrame | None = None) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Return (one-row DataFrame or None, errors, warnings). Errors block prediction; warnings do not.

    ``None`` is accepted only for ca and thal (treated as missing, as in the study).
    Warnings flag values inside the plausible range but outside the training data's observed range.
    """
    errors, warnings, row = [], [], {}
    for c in FEATURE_COLUMNS:
        v = values.get(c)
        if v is None:
            if c in OPTIONAL:
                row[c] = np.nan
            else:
                errors.append(f"{LABELS[c]}: a value is required.")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{LABELS[c]}: '{v}' is not a number.")
            continue
        if not np.isfinite(v):
            errors.append(f"{LABELS[c]}: value must be finite.")
            continue
        if c in DOMAINS and v not in DOMAINS[c]:
            errors.append(f"{LABELS[c]}: {v:g} is not one of {sorted(DOMAINS[c])}.")
            continue
        if c in PLAUSIBLE and not PLAUSIBLE[c][0] <= v <= PLAUSIBLE[c][1]:
            errors.append(f"{LABELS[c]}: {v:g} is outside the plausible range {PLAUSIBLE[c][0]}-{PLAUSIBLE[c][1]}.")
            continue
        row[c] = v
        if train_X is not None and c in CONTINUOUS and not train_X[c].min() <= v <= train_X[c].max():
            warnings.append(f"{LABELS[c]} = {v:g} is outside the range seen in the training data "
                            f"({train_X[c].min():g}-{train_X[c].max():g}); the models are extrapolating.")
    if errors:
        return None, errors, warnings
    return pd.DataFrame([row], columns=FEATURE_COLUMNS), errors, warnings


def default_values(train_X: pd.DataFrame) -> dict:
    """Typical profile: training median for continuous features, most frequent value otherwise."""
    out = {}
    for c in FEATURE_COLUMNS:
        out[c] = float(train_X[c].median()) if c in CONTINUOUS else float(train_X[c].mode().iloc[0])
    for c in ("age", "trestbps", "chol", "thalach"):
        out[c] = int(round(out[c]))
    return out


def final_hyperparameters(cfg: dict | None = None) -> tuple[dict, str]:
    """Most frequently selected (joint) inner-CV setting per model; falls back to the config defaults."""
    cfg = cfg or load_config()
    path = resolve(cfg["paths"]["metrics"]) / "selected_hyperparameters.csv"
    out = {m: {} for m in MODEL_NAMES}
    if not path.exists():
        return out, "config defaults (nested-CV selections not found)"
    sel = pd.read_csv(path)
    for m in MODEL_NAMES:
        s = sel[sel.model == m]
        cols = [c for c in s.columns if c.startswith("model__") and s[c].notna().any()]
        combo = s.groupby(cols).size().sort_values(ascending=False).index[0]
        combo = (combo,) if not isinstance(combo, tuple) else combo
        out[m] = {c: float(v) for c, v in zip(cols, combo)}
    return out, "most frequently selected setting in the nested cross-validation"


@dataclass
class FinalModels:
    pipelines: dict
    train_X: pd.DataFrame
    train_y: pd.Series
    hyperparameters: dict
    hyperparameter_source: str
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def predict(self, row: pd.DataFrame) -> dict[str, float]:
        """P(disease) per model. The lock guards objects shared across Streamlit sessions."""
        with self._lock:
            return {m: float(p.predict_proba(row)[0, 1]) for m, p in self.pipelines.items()}


def train_final_models(cfg: dict | None = None) -> FinalModels:
    cfg = cfg or load_config()
    X, y = get_xy(load_cleveland(config=cfg))
    hp, source = final_hyperparameters(cfg)
    pipes = {}
    for m in MODEL_NAMES:
        pipe = clone(_BUILDERS[m](cfg)).set_params(**hp[m])
        pipes[m] = pipe.fit(X, y)
    return FinalModels(pipes, X, y, hp, source)


def explain_mlp(models: FinalModels, row: pd.DataFrame, n_background: int = 30, nsamples: int = 300, seed: int = 42) -> pd.Series:
    """Kernel-SHAP values (change in P(disease)) per raw predictor for one row, plus a ``base value`` entry."""
    import logging
    import shap  # imported lazily: heavy, and only needed when the user asks for an explanation
    logging.getLogger("shap").setLevel(logging.WARNING)
    pipe = models.pipelines["MLP"]
    f = lambda a: pipe.predict_proba(pd.DataFrame(a, columns=FEATURE_COLUMNS))[:, 1]
    bg = models.train_X.sample(n_background, random_state=seed).to_numpy(dtype=float)
    with models._lock:
        np.random.seed(seed)
        expl = shap.KernelExplainer(f, bg)
        sv = np.asarray(expl.shap_values(row[FEATURE_COLUMNS].to_numpy(dtype=float), nsamples=nsamples, silent=True))[0]
        base = float(np.ravel(expl.expected_value)[0])
    s = pd.Series(sv, index=FEATURE_COLUMNS)
    s.attrs["base_value"] = base
    return s
