"""Shared helpers for all pages: disclaimer banner, cached loaders, palette."""
from __future__ import annotations

import altair as alt
import streamlit as st

from src.data.loader import (
    PHASE2_CONFIG_PATH,
    get_xy,
    load_all_cohorts,
    load_cleveland,
    load_config,
    resolve,
)
from src.models.demo_models_phase2 import (
    FinalModelsPhase2,
    load_or_fit_phase2_demo_models,
)
from src.webapp.content import ATTRIBUTION, DATA_MISSING_HELP, DISCLAIMER
from src.webapp.inference import FinalModels, train_final_models
from src.webapp.results_io import (
    MODEL_ORDER,
    PHASE2_MODEL_ORDER,
    load_phase2_results,
    load_results,
    phase2_results_available,
    results_available,
)

# Okabe-Ito colour-blind-safe palette; model identity is also given by labels, never by colour alone.
PALETTE = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161"]  # same as the figures (seaborn "colorblind")
MODEL_COLOR = alt.Scale(domain=MODEL_ORDER, range=PALETTE)

PHASE2_PALETTE = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161", "#56b4e9", "#e69f00"]
PHASE2_MODEL_COLOR = alt.Scale(domain=PHASE2_MODEL_ORDER, range=PHASE2_PALETTE)


def banner() -> None:
    st.warning(DISCLAIMER)


def footer() -> None:
    st.divider()
    st.caption(ATTRIBUTION)


@st.cache_data(show_spinner=False)
def get_data():
    """Cleveland data (features, binary target, full frame). Extracts the archive member if needed."""
    cfg = load_config()
    raw = resolve(cfg["data"]["raw_file"])
    if not raw.exists():
        from src.data.loader import extract_raw_file
        extract_raw_file(cfg["data"]["archive"], raw.parent)
    df = load_cleveland(config=cfg)
    X, y = get_xy(df)
    return X, y, df


def require_data():
    try:
        return get_data()
    except (FileNotFoundError, OSError):
        st.error("The Cleveland dataset was not found.")
        st.markdown(DATA_MISSING_HELP)
        st.stop()


@st.cache_data(show_spinner=False)
def get_phase2_data():
    """All 4 UCI cohorts (920 rows). Extracts members if needed."""
    cfg = load_config(PHASE2_CONFIG_PATH)
    df = load_all_cohorts(config=cfg)
    X, y = get_xy(df)
    return X, y, df


def require_phase2_data():
    try:
        return get_phase2_data()
    except Exception as exc:
        st.error(f"Phase 2 datasets could not be loaded: {exc}")
        st.markdown(DATA_MISSING_HELP)
        st.stop()


@st.cache_resource(show_spinner="Training the six demonstration models on the full dataset (a few seconds)...")
def get_models() -> FinalModels:
    require_data()
    return train_final_models()


@st.cache_resource(show_spinner="Loading the eight Phase 2 demonstration models...")
def get_phase2_models() -> FinalModelsPhase2:
    require_phase2_data()
    return load_or_fit_phase2_demo_models()


@st.cache_data(show_spinner=False)
def get_results():
    return load_results()


def require_results():
    if not results_available():
        st.error("Saved evaluation results were not found.")
        st.markdown("Run `python -m src.evaluation.run_evaluation` (about 5 minutes) to create them.")
        st.stop()
    return get_results()


@st.cache_data(show_spinner=False)
def get_phase2_results():
    return load_phase2_results()


def require_phase2_results():
    if not phase2_results_available():
        st.error("Saved Phase 2 evaluation results were not found.")
        st.markdown("Run `python -m src.evaluation.run_phase2 --protocol pooled` to generate them.")
        st.stop()
    return get_phase2_results()
