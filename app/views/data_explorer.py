"""Descriptive data explorer (no inference)."""
import altair as alt
import pandas as pd
import streamlit as st

from app.views.common import banner, footer, require_data
from src.data.eda import CATEGORICAL, CONTINUOUS, DESCRIPTIONS
from src.data.loader import FEATURE_COLUMNS


def render() -> None:
    st.title("Data explorer")
    banner()
    X, y, df = require_data()
    st.caption("Descriptive views of the Cleveland data. Nothing here is a significance test, and associations are not causal.")
    c = st.columns(4)
    c[0].metric("Patients", len(df))
    c[1].metric("Disease (num 1-4)", f"{int(y.sum())} ({100 * y.mean():.1f}%)")
    c[2].metric("Predictors", len(FEATURE_COLUMNS))
    c[3].metric("Rows with missing values", int(X.isna().any(axis=1).sum()))
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Feature vs target", "Correlations", "Missing values and levels"])
    plot_df = X.assign(target=y.map({0: "no disease", 1: "disease"}))
    with tab1:
        feat = st.selectbox("Predictor", FEATURE_COLUMNS, key="dist_feat", format_func=lambda f: f"{f}: {DESCRIPTIONS[f]}")
        if feat in CONTINUOUS:
            bins = st.slider("Bins", 5, 40, 20, key="dist_bins")
            ch = alt.Chart(plot_df).mark_bar().encode(
                alt.X(f"{feat}:Q", bin=alt.Bin(maxbins=bins)), alt.Y("count():Q"),
                alt.Color("target:N", scale=alt.Scale(domain=["no disease", "disease"], range=["#0072B2", "#D55E00"])))
        else:
            ch = alt.Chart(plot_df.dropna(subset=[feat])).mark_bar().encode(
                alt.X(f"{feat}:O"), alt.Y("count():Q"),
                alt.Color("target:N", scale=alt.Scale(domain=["no disease", "disease"], range=["#0072B2", "#D55E00"])),
                xOffset="target:N")
        st.altair_chart(ch.properties(height=320), use_container_width=True)
    with tab2:
        cat = st.selectbox("Categorical predictor", CATEGORICAL, key="rate_feat", format_func=lambda f: f"{f}: {DESCRIPTIONS[f]}")
        g = plot_df.dropna(subset=[cat]).groupby(cat)["target"].agg(n="count", positive_rate=lambda s: (s == "disease").mean()).reset_index()
        st.altair_chart(alt.Chart(g).mark_bar(color="#D55E00").encode(
            alt.X(f"{cat}:O"), alt.Y("positive_rate:Q", scale=alt.Scale(domain=[0, 1]), title="Share with disease"),
            tooltip=[cat, "n", alt.Tooltip("positive_rate:Q", format=".3f")]).properties(height=300), use_container_width=True)
        st.dataframe(g, hide_index=True)
        st.caption("Small groups give unstable rates; compare with the group size n. A high rate in one group (for example asymptomatic "
                   "chest pain) probably reflects who was referred for angiography.")
    with tab3:
        corr = X.corr(method="spearman").reset_index(names="a").melt("a", var_name="b", value_name="rho")
        st.altair_chart(alt.Chart(corr).mark_rect().encode(
            alt.X("b:O", title=None), alt.Y("a:O", title=None),
            alt.Color("rho:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)),
            tooltip=["a", "b", alt.Tooltip("rho:Q", format=".2f")]).properties(height=420), use_container_width=True)
        st.caption("Spearman correlation between predictors. Nominal codes (cp, restecg, thal) have no natural order, so their coefficients are only a rough guide.")
    with tab4:
        miss = X.isna().sum()
        st.dataframe(pd.DataFrame({"missing": miss[miss > 0]}))
        st.caption("Only `ca` and `thal` have missing values. In the study they are filled with the training-fold mode.")
        st.markdown("**Patients per level** (sparse levels make probability estimates unstable)")
        lv = pd.concat([X[f].value_counts(dropna=False).sort_index().rename_axis("level").reset_index(name="patients").assign(feature=f)
                        for f in ["cp", "restecg", "slope", "ca", "thal"]])
        lv["level"] = lv["level"].map(lambda v: "missing" if pd.isna(v) else f"{v:g}")
        st.dataframe(lv[["feature", "level", "patients"]], hide_index=True)
    footer()
