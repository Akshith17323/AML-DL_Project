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

    has_p2 = phase2_results_available()
    if has_p2:
        tabs = [
            "Phase 2 Multi-Cohort (920 patients)",
            "Phase 2 Hospital Distribution Shift (LOGO)",
            "Phase 2 Ablations & SHAP Interpretability",
            "Phase 1 Summary & Paired Differences",
            "Phase 1 Curves & Threshold Explorer",
        ]
        rendered_tabs = st.tabs(tabs)
        t_p2_pooled, t_p2_shift, t_p2_shap, t_p1_sum, t_p1_curves = rendered_tabs

        # ---------------------------------------------------- TAB 0: Phase 2 Multi-Cohort Pooled
        with t_p2_pooled:
            st.subheader("Phase 2 Multi-Cohort Nested Cross-Validation (920 patients)")
            st.markdown("#### 1. Pooled Nested CV Results (8 Models)")
            st.caption(
                "All eight models evaluated on identical outer folds across all four processed UCI cohorts "
                "(Cleveland, Hungarian, Long Beach VA, Switzerland). Hyperparameters selected strictly inside training folds by Optuna TPE search."
            )
            p2 = load_phase2_results()
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

            st.markdown("#### Diagnostic & Training Curves")
            c1, c2 = st.columns(2)
            fig_pooled = PROJECT_ROOT / "results/phase2/main/figures/curves_pooled.png"
            fig_mlp = PROJECT_ROOT / "results/phase2/main/figures/mlp_learning_curve.png"

            with c1:
                if fig_pooled.exists():
                    st.image(str(fig_pooled), caption="Pooled Nested CV ROC/PR Curves across all 8 models")
            with c2:
                if fig_mlp.exists():
                    st.image(str(fig_mlp), caption="Tabular MLP Training & Early-Stopping Validation Curve (Log Loss convergence)")

        # ---------------------------------------------------- TAB 1: Phase 2 Hospital Distribution Shift (LOGO)
        with t_p2_shift:
            st.subheader("Cross-Cohort Hospital Shift (Leave-One-Group-Out CV)")
            st.caption(
                "Leave-One-Group-Out cross-validation: in each round, 3 hospital centres are used for training and tuning, "
                "and the 4th held-out centre acts as an unseen external validation hospital. This directly measures vulnerability to real-world clinical distribution shift."
            )
            p2 = load_phase2_results()
            p2_cohort_sum = p2.get("summary_cross_cohort", pd.DataFrame())
            if not p2_cohort_sum.empty:
                cohort_rows = p2_cohort_sum[p2_cohort_sum.cohort == "ALL"][["model", "roc_auc", "pr_auc", "brier", "log_loss", "accuracy"]].copy()
                cohort_rows.columns = ["Model", "ROC AUC (Macro)", "PR AUC", "Brier", "Log Loss", "Accuracy"]
                st.dataframe(cohort_rows.round(3), hide_index=True, use_container_width=True)

            fig_cohort = PROJECT_ROOT / "results/phase2/main/figures/curves_cross-cohort.png"
            if fig_cohort.exists():
                st.image(str(fig_cohort), caption="Leave-One-Cohort-Out Generalisation Curves (Cleveland, Hungarian, Long Beach VA, Switzerland)")

            st.warning(
                "**Clinical Implication:** In-distribution pooled CV overstates performance on external centres. "
                "On Switzerland, discrimination drops across all models (ROC AUC 0.70–0.78) due to referral severity bias and missing fluoroscopy/thal measures."
            )

        # ---------------------------------------------------- TAB 2: Phase 2 Ablations & SHAP
        with t_p2_shap:
            st.subheader("Controlled Ablations & SHAP Nonlinear Interaction Analysis")
            st.caption("Ablating MLP components (dropout, weight decay, early stopping, depth) and isolating learned feature interactions.")
            p2 = load_phase2_results()
            abl = p2.get("ablations", pd.DataFrame())
            if not abl.empty:
                st.markdown("#### 1. Ablation Study Results")
                st.dataframe(abl.round(3), hide_index=True, use_container_width=True)

            st.markdown("#### 2. Global SHAP Feature Importance (Tabular MLP)")
            st.caption("SHAP (SHapley Additive exPlanations) values quantify each predictor's marginal contribution to predicted cardiovascular risk.")

            col_s1, col_s2 = st.columns(2)
            fig_beeswarm = PROJECT_ROOT / "results/phase2/main/figures/mlp_shap_beeswarm.png"
            fig_bars = PROJECT_ROOT / "results/phase2/main/figures/mlp_shap_bar_importance.png"
            with col_s1:
                if fig_beeswarm.exists():
                    st.image(str(fig_beeswarm), caption="MLP Global SHAP Beeswarm (Marginal Risk Impact)")
            with col_s2:
                if fig_bars.exists():
                    st.image(str(fig_bars), caption="Mean |SHAP| Value per Predictor")

            st.markdown("#### 3. Learned Nonlinear Interactions")
            st.caption("Continuous nonlinear surfaces captured by the Tabular MLP compared to rigid step functions in discrete Bayesian networks.")
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

        # ---------------------------------------------------- TAB 3: Phase 1 Summary & Paired Differences
        with t_p1_sum:
            st.subheader("Phase 1 Cleveland Classical Benchmark (303 patients)")
            res = require_results()
            st.caption("Out-of-fold results from nested cross-validation: stratified 5-fold x 5 repeats = 25 identical outer folds per model.")
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
            st.image(str(PROJECT_ROOT / "results/figures/confusion_matrices.png"), caption="Mean confusion counts per repeat at threshold 0.5")

            st.markdown("---")
            st.markdown("**Paired out-of-fold differences (2000-sample patient bootstrap)**")
            d = res["diff"]
            metric = st.selectbox("Metric", ["roc_auc", "accuracy", "brier"], format_func=lambda m: {"roc_auc": "ROC AUC", "accuracy": "Accuracy", "brier": "Brier score"}[m], key="diff_metric")
            sub = d[d.metric == metric].copy()
            sub["includes_zero"] = sub.ci_includes_zero.map({True: "yes", False: "no"})
            sub["ci"] = sub.apply(lambda r: f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
            sub["diff"] = sub.estimate.apply(lambda v: f"{v:+.3f}")
            st.dataframe(sub[["comparison", "diff", "ci", "includes_zero"]].rename(columns={"diff": "difference", "ci": "95% bootstrap CI", "includes_zero": "includes zero"}),
                         hide_index=True)

        # ---------------------------------------------------- TAB 4: Phase 1 Curves & Threshold Explorer
        with t_p1_curves:
            st.subheader("Phase 1 Diagnostic Curves & Decision Threshold Explorer")
            res = require_results()
            repeat = st.selectbox("Repeat of the 5x5 cross-validation", [0, 1, 2, 3, 4], key="curve_repeat")
            for kind, title, xl, yl in (("roc", "ROC curve", "False positive rate (1 - specificity)", "True positive rate (sensitivity)"),
                                        ("pr", "Precision-recall curve", "Recall (sensitivity)", "Precision")):
                pts = curve_points(res["oof"], kind, repeat)
                st.altair_chart(alt.Chart(pts).mark_line(interpolate="step-after").encode(
                    alt.X("x:Q", title=xl, scale=alt.Scale(domain=[0, 1])), alt.Y("y:Q", title=yl, scale=alt.Scale(domain=[0, 1])),
                    alt.Color("model:N", scale=MODEL_COLOR, title="Model"), tooltip=["model", alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("y:Q", format=".3f")]
                ).properties(title=title, height=330), use_container_width=True)

            st.markdown("---")
            st.markdown("#### Decision Threshold Explorer")
            st.markdown("Move the threshold to see the sensitivity/specificity trade-off on the pooled out-of-fold predictions.")
            thr = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05, key="thr")
            m = metrics_at_threshold(res["oof"], thr)
            rate, count = st.column_config.NumberColumn(format="%.3f"), st.column_config.NumberColumn(format="%.1f")
            st.dataframe(m[["accuracy", "sensitivity", "specificity", "precision", "f1", "tn", "fp", "fn", "tp"]],
                         column_config={**{c: rate for c in ("accuracy", "sensitivity", "specificity", "precision", "f1")}, **{c: count for c in ("tn", "fp", "fn", "tp")}})
            st.caption("tn / fp / fn / tp are mean patient counts per repeat (303 patients in total).")

    else:
        # Fallback when Phase 2 results are not present
        tabs = ["Summary tables", "ROC and PR curves", "Threshold explorer", "Paired differences"]
        rendered_tabs = st.tabs(tabs)
        t_sum, t_curve, t_thr, t_diff = rendered_tabs

        with t_sum:
            res = require_results()
            f = res["fold"]
            f = f[f.threshold_scheme == "fixed_0.5"].set_index("model").reindex(MODEL_ORDER)
            cols = [("accuracy", "Accuracy"), ("roc_auc", "ROC AUC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                    ("precision", "Precision"), ("f1", "F1"), ("brier", "Brier")]
            st.dataframe(pd.DataFrame({h: [_pm(f, m, c) for m in MODEL_ORDER] for c, h in cols}, index=MODEL_ORDER))
        with t_curve:
            res = require_results()
            pts = curve_points(res["oof"], "roc", 0)
            st.altair_chart(alt.Chart(pts).mark_line().encode(x="x:Q", y="y:Q", color="model:N"))
        with t_thr:
            res = require_results()
            thr = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)
            st.dataframe(metrics_at_threshold(res["oof"], thr))
        with t_diff:
            res = require_results()
            st.dataframe(res["diff"])

    footer()
