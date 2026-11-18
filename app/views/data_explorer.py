"""Descriptive data explorer (no inference)."""
import altair as alt
import pandas as pd
import streamlit as st

from app.views.common import banner, footer, get_phase2_data, require_data
from src.data.eda import CATEGORICAL, CONTINUOUS, DESCRIPTIONS
from src.data.loader import FEATURE_COLUMNS, PHASE2_CONFIG_PATH, load_config, resolve


def render() -> None:
    st.title("Data explorer")
    banner()

    try:
        X_all, y_all, df_all = get_phase2_data()
        has_multi = True
    except Exception:
        X_all, y_all, df_all = None, None, None
        has_multi = False

    X, y, df = require_data()

    st.caption("Descriptive audit and statistical distributions across international clinical cohorts (920 multi-cohort patients, 303 Cleveland baseline).")
    c = st.columns(4)
    c[0].metric("Patients", len(df))
    c[1].metric("Disease (num 1-4)", f"{int(y.sum())} ({100 * y.mean():.1f}%)")
    c[2].metric("Predictors", len(FEATURE_COLUMNS))
    c[3].metric("Rows with missing values", int(X.isna().any(axis=1).sum()))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Phase 2 Multi-Cohort Audit (920 patients)", "Distributions", "Feature vs target", "Correlations", "Missing values and levels"
    ])

    with tab1:
        st.subheader("Phase 2 Multi-Cohort Audit (920 patients across 4 hospitals)")
        st.caption("Cross-cohort comparison of patient volume, disease prevalence, and data completeness across four international centres.")

        if has_multi and df_all is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown("**Total Patients**\n### 920\n4 Clinical Centres")
            col2.markdown("**Disease Cases**\n### 508 (55.2%)\nCAD (>=50% stenosis)")
            col3.markdown("**Clinical Predictors**\n### 13\nStandardised Features")
            col4.markdown("**Data Completeness**\n### Varied\nSwiss/VA Missing ca & thal")

            c_counts = df_all.groupby("cohort")["target"].agg(
                patients="count",
                positives=lambda s: int(s.sum()),
                prevalence=lambda s: f"{100 * s.mean():.1f}%"
            ).reset_index()
            c_counts["missing_ca_pct"] = df_all.groupby("cohort")["ca"].apply(lambda s: f"{100 * s.isna().mean():.1f}%").values
            c_counts["missing_thal_pct"] = df_all.groupby("cohort")["thal"].apply(lambda s: f"{100 * s.isna().mean():.1f}%").values
            c_counts.columns = ["Cohort", "Total Patients", "Disease Cases", "Prevalence", "Missing Fluoroscopy (ca)", "Missing Thallium (thal)"]

            st.markdown("#### Cohort breakdown & pre-test severity")
            st.markdown("**Cohort breakdown:**")
            st.dataframe(c_counts, hide_index=True, use_container_width=True)

            try:
                cfg2 = load_config(PHASE2_CONFIG_PATH)
                audit_img = resolve(cfg2["paths"]["figures"]) / "cohort_data_audit.png"
                if audit_img.exists():
                    st.image(str(audit_img), caption="Missingness and Pre-test Probability Across Cohorts (Cleveland, Hungarian, Long Beach VA, Switzerland)")
            except Exception:
                pass
        else:
            st.info("Multi-cohort dataset not available.")

    with tab2:
        st.markdown("**Predictor Distributions**")
        cohort_choice = "Cleveland (303 patients, Phase 1 baseline)"
        if has_multi and df_all is not None:
            cohort_choice = st.selectbox(
                "Select Cohort to Inspect",
                ["Cleveland (303 patients, Phase 1 baseline)", "All 4 Cohorts Combined (920 patients)", "Hungarian (294 patients)", "VA Long Beach (200 patients)", "Switzerland (123 patients)"],
                index=0,
                key="dist_cohort_select",
            )

        if cohort_choice.startswith("All 4"):
            active_df = df_all
        elif cohort_choice.startswith("Hungarian"):
            active_df = df_all[df_all["cohort"] == "Hungarian"]
        elif cohort_choice.startswith("VA"):
            active_df = df_all[df_all["cohort"] == "VA"]
        elif cohort_choice.startswith("Switzerland"):
            active_df = df_all[df_all["cohort"] == "Switzerland"]
        else:
            active_df = df

        plot_df = active_df[FEATURE_COLUMNS].assign(target=active_df["target"].map({0: "no disease", 1: "disease"}))
        feat = st.selectbox("Predictor", FEATURE_COLUMNS, key="dist_feat", format_func=lambda f: f"{f}: {DESCRIPTIONS[f]}")
        if feat in CONTINUOUS:
            bins = st.slider("Bins", 5, 40, 20, key="dist_bins")
            ch = alt.Chart(plot_df.dropna(subset=[feat])).mark_bar().encode(
                alt.X(f"{feat}:Q", bin=alt.Bin(maxbins=bins)), alt.Y("count():Q"),
                alt.Color("target:N", scale=alt.Scale(domain=["no disease", "disease"], range=["#0072B2", "#D55E00"])))
        else:
            ch = alt.Chart(plot_df.dropna(subset=[feat])).mark_bar().encode(
                alt.X(f"{feat}:O"), alt.Y("count():Q"),
                alt.Color("target:N", scale=alt.Scale(domain=["no disease", "disease"], range=["#0072B2", "#D55E00"])),
                xOffset="target:N")
        st.altair_chart(ch.properties(height=320), use_container_width=True)

    with tab3:
        st.markdown("**Feature vs Disease Target (Cleveland Cohort)**")
        plot_df_cleveland = X.assign(target=y.map({0: "no disease", 1: "disease"}))
        cat = st.selectbox("Categorical predictor", CATEGORICAL, key="rate_feat", format_func=lambda f: f"{f}: {DESCRIPTIONS[f]}")
        g = plot_df_cleveland.dropna(subset=[cat]).groupby(cat)["target"].agg(n="count", positive_rate=lambda s: (s == "disease").mean()).reset_index()
        st.altair_chart(alt.Chart(g).mark_bar(color="#D55E00").encode(
            alt.X(f"{cat}:O"), alt.Y("positive_rate:Q", scale=alt.Scale(domain=[0, 1]), title="Share with disease"),
            tooltip=[cat, "n", alt.Tooltip("positive_rate:Q", format=".3f")]).properties(height=300), use_container_width=True)
        st.dataframe(g, hide_index=True)
        st.caption("Small groups give unstable rates; compare with the group size n. A high rate in one group (for example asymptomatic "
                   "chest pain) probably reflects who was referred for angiography.")

    with tab4:
        st.markdown("**Predictor Correlations (Spearman Rank)**")
        corr = X.corr(method="spearman").reset_index(names="a").melt("a", var_name="b", value_name="rho")
        st.altair_chart(alt.Chart(corr).mark_rect().encode(
            alt.X("b:O", title=None), alt.Y("a:O", title=None),
            alt.Color("rho:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)),
            tooltip=["a", "b", alt.Tooltip("rho:Q", format=".2f")]).properties(height=420), use_container_width=True)
        st.caption("Spearman correlation between predictors. Nominal codes (cp, restecg, thal) have no natural order, so their coefficients are only a rough guide.")

    with tab5:
        st.markdown("**Missing Values & Level Counts**")
        miss = X.isna().sum()
        st.dataframe(pd.DataFrame({"missing": miss[miss > 0]}))
        st.caption("Only `ca` and `thal` have missing values in Cleveland. In multi-cohort Swiss and Long Beach, missingness is more severe.")
        st.markdown("**Patients per level** (sparse levels make probability estimates unstable)")
        lv = pd.concat([X[f].value_counts(dropna=False).sort_index().rename_axis("level").reset_index(name="patients").assign(feature=f)
                        for f in ["cp", "restecg", "slope", "ca", "thal"]])
        lv["level"] = lv["level"].map(lambda v: "missing" if pd.isna(v) else f"{v:g}")
        st.dataframe(lv[["feature", "level", "patients"]], hide_index=True)

    footer()
