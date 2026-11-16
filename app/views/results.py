"""Saved evaluation results: tables, curves, calibration, exploratory threshold view."""
import altair as alt
import pandas as pd
import streamlit as st

from app.views.common import MODEL_COLOR, banner, footer, require_results
from src.data.loader import PROJECT_ROOT
from src.webapp.results_io import (
    MODEL_ORDER,
    PHASE2_MODEL_ORDER,
    curve_points,
    load_phase2_results,
    metrics_at_threshold,
    phase2_results_available,
)


def _pm(df, model, col, d=3):
    return f"{df.loc[model, col + '_mean']:.{d}f} ± {df.loc[model, col + '_sd']:.{d}f}"


def render() -> None:
    st.title("Model results")
    banner()
    res = require_results()
    st.caption("Out-of-fold results from nested cross-validation: stratified 5-fold x 5 repeats = 25 identical outer folds per model, "
               "hyper-parameters tuned inside each training fold, threshold 0.5. Values are read from `results/metrics/`.")
    tabs = ["Summary tables", "ROC and PR curves", "Threshold explorer", "Paired differences"]
    has_p2 = phase2_results_available()
    if has_p2:
        tabs.append("Phase 2 Multi-Cohort (920 patients)")

    rendered_tabs = st.tabs(tabs)
    t_sum, t_curve, t_thr, t_diff = rendered_tabs[0], rendered_tabs[1], rendered_tabs[2], rendered_tabs[3]

    with t_sum:
        f = res["fold"]
        f = f[f.threshold_scheme == "fixed_0.5"].set_index("model").reindex(MODEL_ORDER)
        cols = [("accuracy", "Accuracy"), ("roc_auc", "ROC AUC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                ("precision", "Precision"), ("f1", "F1"), ("brier", "Brier")]
        st.markdown("**Fold-level mean ± SD over 25 outer folds (about 60 patients each)**")
        st.dataframe(pd.DataFrame({h: [_pm(f, m, c) for m in MODEL_ORDER] for c, h in cols}, index=MODEL_ORDER))
        ci = res["ci"]
        st.markdown("**Pooled out-of-fold estimate with 95% patient-level bootstrap interval**")
        rows = {}
        for m in MODEL_ORDER:
            r = ci[ci.model == m].set_index("metric")
            rows[m] = {h: f"{r.loc[k, 'estimate']:.3f} [{r.loc[k, 'ci_low']:.3f}, {r.loc[k, 'ci_high']:.3f}]"
                       for k, h in [("accuracy", "Accuracy"), ("roc_auc", "ROC AUC"), ("sensitivity", "Sensitivity"),
                                    ("specificity", "Specificity"), ("brier", "Brier")]}
        st.dataframe(pd.DataFrame(rows).T)
        st.caption("The fold-level SD understates uncertainty about new patients (training sets overlap); the bootstrap interval resamples patients, not the training procedure.")
        st.image(str(PROJECT_ROOT / "results/figures/confusion_matrices.png"), caption="Mean confusion counts per repeat at threshold 0.5")

    with t_curve:
        repeat = st.selectbox("Repeat of the 5x5 cross-validation", [0, 1, 2, 3, 4], key="curve_repeat")
        for kind, title, xl, yl in (("roc", "ROC curve", "False positive rate (1 - specificity)", "True positive rate (sensitivity)"),
                                    ("pr", "Precision-recall curve", "Recall (sensitivity)", "Precision")):
            pts = curve_points(res["oof"], kind, repeat)
            st.altair_chart(alt.Chart(pts).mark_line(interpolate="step-after").encode(
                alt.X("x:Q", title=xl, scale=alt.Scale(domain=[0, 1])), alt.Y("y:Q", title=yl, scale=alt.Scale(domain=[0, 1])),
                alt.Color("model:N", scale=MODEL_COLOR, title="Model"), tooltip=["model", alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("y:Q", format=".3f")]
            ).properties(title=title, height=330), use_container_width=True)
        st.caption("The curves overlap closely; see the Paired differences tab for which gaps are larger than the noise.")

    with t_thr:
        st.markdown("Move the threshold to see the sensitivity/specificity trade-off on the pooled out-of-fold predictions.")
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05, key="thr")
        m = metrics_at_threshold(res["oof"], thr)
        rate, count = st.column_config.NumberColumn(format="%.3f"), st.column_config.NumberColumn(format="%.1f")
        st.dataframe(m[["accuracy", "sensitivity", "specificity", "precision", "f1", "tn", "fp", "fn", "tp"]],
                     column_config={**{c: rate for c in ("accuracy", "sensitivity", "specificity", "precision", "f1")}, **{c: count for c in ("tn", "fp", "fn", "tp")}})
        st.caption("tn / fp / fn / tp are mean patient counts per repeat (303 patients in total).")
        st.warning("Exploratory only. Picking a threshold by looking at these labels is optimistic. The study fixed 0.5 in advance, and a "
                   "threshold chosen on training data (Youden) did not improve sensitivity for any model.")

    with t_diff:
        d = res["diff"]
        st.markdown("**Paired out-of-fold differences (2000-sample patient bootstrap)**")
        st.caption("A positive difference means the first model has a higher score. Intervals that do not include zero are highlighted in bold.")
        metric = st.selectbox("Metric", ["roc_auc", "accuracy", "brier"], format_func=lambda m: {"roc_auc": "ROC AUC", "accuracy": "Accuracy", "brier": "Brier score"}[m], key="diff_metric")
        sub = d[d.metric == metric].copy()
        sub["includes_zero"] = sub.ci_includes_zero.map({True: "yes", False: "no"})
        sub["ci"] = sub.apply(lambda r: f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
        sub["diff"] = sub.estimate.apply(lambda v: f"{v:+.3f}")
        st.dataframe(sub[["comparison", "diff", "ci", "includes_zero"]].rename(columns={"diff": "difference", "ci": "95% bootstrap CI", "includes_zero": "includes zero"}),
                     hide_index=True)

    if has_p2:
        t_phase2 = rendered_tabs[4]
        with t_phase2:
            st.subheader("Phase 2 Multi-Cohort Nested Cross-Validation (920 patients)")
            p2 = load_phase2_results()

            st.markdown("#### 1. Pooled Nested CV Results (8 Models)")
            st.caption("All eight models evaluated on identical outer folds across all four processed UCI cohorts. Hyperparameters selected by inner Optuna TPE search.")
            p2_ci = p2.get("ci_pooled", pd.DataFrame())
            p2_sum = p2.get("summary_pooled", pd.DataFrame())
            if not p2_ci.empty:
                rows = []
                for m in PHASE2_MODEL_ORDER:
                    sub_ci = p2_ci[p2_ci.model == m].set_index("metric")
                    sub_sum = p2_sum[(p2_sum.model == m) & (p2_sum.cohort == "ALL")].iloc[0] if not p2_sum.empty and ((p2_sum.model == m) & (p2_sum.cohort == "ALL")).any() else None
                    if "roc_auc" in sub_ci.index:
                        rows.append({
                            "Model": m,
                            "ROC AUC [95% CI]": f"{sub_ci.loc['roc_auc', 'estimate']:.3f} [{sub_ci.loc['roc_auc', 'ci_low']:.3f}, {sub_ci.loc['roc_auc', 'ci_high']:.3f}]",
                            "PR AUC": f"{sub_sum['pr_auc']:.3f}" if sub_sum is not None and "pr_auc" in sub_sum else "N/A",
                            "Brier Score": f"{sub_ci.loc['brier', 'estimate']:.3f}" if "brier" in sub_ci.index else "N/A",
                            "Log Loss": f"{sub_ci.loc['log_loss', 'estimate']:.3f}" if "log_loss" in sub_ci.index else "N/A",
                            "Accuracy": f"{sub_ci.loc['accuracy', 'estimate']:.3f}" if "accuracy" in sub_ci.index else "N/A",
                        })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            st.markdown("#### 2. Cross-Cohort Hospital Shift (Leave-One-Cohort-Out CV)")
            st.caption("Generalisation gap when each hospital is completely held out as an unseen test cohort.")
            p2_cohort_sum = p2.get("summary_cross_cohort", pd.DataFrame())
            if not p2_cohort_sum.empty:
                cohort_rows = p2_cohort_sum[p2_cohort_sum.cohort == "ALL"][["model", "roc_auc", "pr_auc", "brier", "log_loss", "accuracy"]].copy()
                cohort_rows.columns = ["Model", "ROC AUC (Macro)", "PR AUC", "Brier", "Log Loss", "Accuracy"]
                st.dataframe(cohort_rows.round(3), hide_index=True, use_container_width=True)

            st.markdown("#### 3. Controlled Ablation Studies")
            st.caption("Ablating MLP components (dropout, weight decay, early stopping, depth) and comparing integration schemes.")
            abl = p2.get("ablations", pd.DataFrame())
            if not abl.empty:
                st.dataframe(abl.round(3), hide_index=True, use_container_width=True)

            st.markdown("#### 4. Diagnostic & Training Figures")
            c1, c2 = st.columns(2)
            fig_pooled = PROJECT_ROOT / "results/phase2/main/figures/curves_pooled.png"
            fig_cohort = PROJECT_ROOT / "results/phase2/main/figures/curves_cross-cohort.png"
            fig_mlp = PROJECT_ROOT / "results/phase2/main/figures/mlp_learning_curve.png"

            with c1:
                if fig_pooled.exists():
                    st.image(str(fig_pooled), caption="Pooled Nested CV ROC/PR Curves")
                if fig_mlp.exists():
                    st.image(str(fig_mlp), caption="MLP Training & Early-Stopping Validation Curve")
            with c2:
                if fig_cohort.exists():
                    st.image(str(fig_cohort), caption="Leave-One-Cohort-Out Generalisation Curves")

            st.markdown("---")
            st.markdown("#### 5. Explainability & Nonlinear Interactions (SHAP Analysis)")
            st.caption("SHAP (SHapley Additive exPlanations) demonstrates how the MLP learns continuous nonlinear risk factor interactions that the fixed Bayesian Network misses.")

            col_s1, col_s2 = st.columns(2)
            fig_beeswarm = PROJECT_ROOT / "results/phase2/main/figures/mlp_shap_beeswarm.png"
            fig_bars = PROJECT_ROOT / "results/phase2/main/figures/mlp_shap_bar_importance.png"
            with col_s1:
                if fig_beeswarm.exists():
                    st.image(str(fig_beeswarm), caption="MLP Global SHAP Beeswarm (Marginal Risk Impact)")
            with col_s2:
                if fig_bars.exists():
                    st.image(str(fig_bars), caption="Mean |SHAP| Value per Predictor")

            st.markdown("##### Learned Nonlinear Interactions: Age × Heart Rate and Blood Pressure × Chest Pain")
            col_i1, col_i2 = st.columns(2)
            fig_at = PROJECT_ROOT / "results/phase2/main/figures/mlp_shap_interaction_age_thalach.png"
            fig_bc = PROJECT_ROOT / "results/phase2/main/figures/mlp_shap_interaction_trestbps_cp.png"
            with col_i1:
                if fig_at.exists():
                    st.image(str(fig_at), caption="Interaction 1: Age × Max Heart Rate (thalach)")
            with col_i2:
                if fig_bc.exists():
                    st.image(str(fig_bc), caption="Interaction 2: Resting BP (trestbps) × Chest Pain (cp)")

            fig_comp = PROJECT_ROOT / "results/phase2/main/figures/mlp_vs_bn_interaction_comparison.png"
            if fig_comp.exists():
                st.image(str(fig_comp), caption="MLP vs Fixed Bayesian Network: Learning Continuous Nonlinear Surfaces vs Rigid Independent Step Functions")

    footer()
