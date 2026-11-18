"""Overview page."""
import pandas as pd
import streamlit as st

from app.views.common import banner, footer, require_results
from src.data.loader import PROJECT_ROOT
from src.webapp.results_io import MODEL_ORDER, PHASE2_MODEL_ORDER, load_phase2_results, phase2_results_available


def render() -> None:
    st.title("Heart Disease Risk Modelling")
    st.subheader("Phase 2: Multi-Cohort Benchmarking Across 4 International Centres (920 Patients)")
    banner()
    st.markdown(
        "This web application presents our comprehensive multi-cohort heart disease risk modelling study. "
        "Phase 2 expands upon the initial single-centre Cleveland benchmark (303 patients) to **all four processed UCI cohorts "
        "(Cleveland, Hungarian, Long Beach VA, and Switzerland; 920 total patients)**. "
        "We benchmark **eight distinct model families** spanning Bayesian probabilistic networks, classical machine learning, "
        "deep neural networks (regularised Tabular MLP), and an integrated Bayesian-neural stacking hybrid under automated "
        "Optuna TPE hyperparameter optimisation."
    )

    has_p2 = phase2_results_available()
    if has_p2:
        st.markdown("### Phase 2 Multi-Cohort Headline Results")
        st.caption("Pooled nested cross-validation across all 4 cohorts (920 patients) with 95% patient-level bootstrap intervals.")

        # Top 4 summary highlights
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**Patients & Centres**\n### 920 Patients\n4 International Cohorts")
        c2.markdown("**Top ROC AUC**\n### 0.913\nGradient Boosting / Stacking")
        c3.markdown("**Optimal Brier Score**\n### 0.118\nTabular MLP / Stacking")
        c4.markdown("**Model Architectures**\n### 8 Models\nBayesian, Trees, MLP, Stack")

        p2_res = load_phase2_results()
        p2_ci = p2_res.get("ci_pooled", None)
        p2_sum = p2_res.get("summary_pooled", None)

        if p2_ci is not None and not p2_ci.empty:
            p2_summary = []
            for m in PHASE2_MODEL_ORDER:
                sub_ci = p2_ci[p2_ci.model == m].set_index("metric")
                sub_sum = p2_sum[(p2_sum.model == m) & (p2_sum.cohort == "ALL")].iloc[0] if p2_sum is not None and not p2_sum.empty and ((p2_sum.model == m) & (p2_sum.cohort == "ALL")).any() else None
                if "roc_auc" in sub_ci.index and "accuracy" in sub_ci.index:
                    p2_summary.append({
                        "Model Architecture": m,
                        "ROC AUC [95% CI]": f"{sub_ci.loc['roc_auc', 'estimate']:.3f} [{sub_ci.loc['roc_auc', 'ci_low']:.3f}, {sub_ci.loc['roc_auc', 'ci_high']:.3f}]",
                        "Accuracy [95% CI]": f"{sub_ci.loc['accuracy', 'estimate']:.3f} [{sub_ci.loc['accuracy', 'ci_low']:.3f}, {sub_ci.loc['accuracy', 'ci_high']:.3f}]",
                        "PR AUC": f"{sub_sum['pr_auc']:.3f}" if sub_sum is not None and "pr_auc" in sub_sum else "N/A",
                        "Brier Score": f"{sub_ci.loc['brier', 'estimate']:.3f}" if "brier" in sub_ci.index else "N/A",
                        "Log Loss": f"{sub_ci.loc['log_loss', 'estimate']:.3f}" if "log_loss" in sub_ci.index else "N/A",
                    })
            if p2_summary:
                st.dataframe(pd.DataFrame(p2_summary), hide_index=True, use_container_width=True)

        st.info(
            "**Key Findings:** While tree-based ensembles and neural architectures achieve top discrimination (ROC AUC 0.90–0.91), "
            "Leave-One-Group-Out (LOGO) cross-cohort hospital shift reveals significant performance degradation across centres "
            "(e.g., Switzerland ROC AUC drops to 0.70–0.78), demonstrating the clinical challenge of hospital distribution shift. "
            "Explore the **Model results** tab for full hospital shift tables, SHAP feature interactions, and ablations."
        )

    st.markdown("---")
    st.markdown("#### Phase 1 Baseline Models (Cleveland Cohort, 303 patients)")
    st.caption("Nested cross-validation, 25 identical outer folds per model, stratified 5-fold x 5 repeats.")
    res = require_results()
    ci = res["ci"]
    for start in range(0, len(MODEL_ORDER), 3):
        for col, model in zip(st.columns(3), MODEL_ORDER[start:start + 3]):
            r = ci[ci.model == model].set_index("metric")
            with col:
                st.metric(f"{model}: ROC AUC", f"{r.loc['roc_auc', 'estimate']:.3f}",
                          help=f"95% bootstrap interval [{r.loc['roc_auc', 'ci_low']:.3f}, {r.loc['roc_auc', 'ci_high']:.3f}]")
                st.caption(f"95% CI [{r.loc['roc_auc', 'ci_low']:.3f}, {r.loc['roc_auc', 'ci_high']:.3f}] · "
                           f"accuracy {r.loc['accuracy', 'estimate']:.3f}")
    diff = res["diff"]
    auc = diff[diff.metric == "roc_auc"]
    separated = auc[~auc.ci_includes_zero].comparison.tolist()
    common = {m for m in MODEL_ORDER if separated and all(m in c for c in separated)}
    if not separated:
        ranking = "no paired ROC AUC difference is distinguishable from zero, so the models cannot be told apart."
    elif len(common) == 1:
        ranking = (f"{len(auc) - len(separated)} of {len(auc)} paired ROC AUC differences include zero. The "
                   f"{len(separated)} that do not all involve {common.pop()}, which is slightly behind; the other models cannot be told apart.")
    else:
        ranking = (f"{len(auc) - len(separated)} of {len(auc)} paired ROC AUC differences include zero "
                   "(see Model results for the rest).")
    st.markdown(
        f"- **Ranking:** {ranking}\n"
        "- **Limits:** one centre, 303 patients, internal validation only. These numbers do not show clinical validity.")

    footer()
