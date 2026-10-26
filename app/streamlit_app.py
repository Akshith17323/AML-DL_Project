"""Streamlit entry point.  Run from the project root:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Heart Disease Risk Modelling", page_icon=None, layout="wide")

from app.views import data_explorer, methods, overview, results, risk_estimator  # noqa: E402

pages = [
    st.Page(overview.render, title="Overview", default=True),
    st.Page(data_explorer.render, title="Data explorer", url_path="data"),
    st.Page(results.render, title="Model results", url_path="results"),
    st.Page(risk_estimator.render, title="Risk estimator (demo)", url_path="estimator"),
    st.Page(methods.render, title="Methods and limitations", url_path="methods"),
]
with st.sidebar:
    st.caption("Academic demonstration, not a medical device. Nothing entered is stored.")
st.navigation(pages).run()
