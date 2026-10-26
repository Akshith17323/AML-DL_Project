"""Shared helpers for all pages: disclaimer banner, cached loaders, palette."""
from __future__ import annotations

import altair as alt
import streamlit as st

from src.data.loader import get_xy, load_cleveland, load_config, resolve
from src.webapp.content import ATTRIBUTION, DATA_MISSING_HELP, DISCLAIMER
from src.webapp.inference import FinalModels, train_final_models
from src.webapp.results_io import MODEL_ORDER, load_results, results_available

# Okabe-Ito colour-blind-safe palette; model identity is also given by labels, never by colour alone.
PALETTE = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161"]  # same as the figures (seaborn "colorblind")
MODEL_COLOR = alt.Scale(domain=MODEL_ORDER, range=PALETTE)


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


@st.cache_resource(show_spinner="Training the six demonstration models on the full dataset (a few seconds)...")
def get_models() -> FinalModels:
    require_data()
    return train_final_models()


@st.cache_data(show_spinner=False)
def get_results():
    return load_results()


def require_results():
    if not results_available():
        st.error("Saved evaluation results were not found.")
        st.markdown("Run `python -m src.evaluation.run_evaluation` (about 5 minutes) to create them.")
        st.stop()
    return get_results()
