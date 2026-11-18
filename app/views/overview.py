"""Overview page."""
import streamlit as st

from app.views.common import banner, footer, require_results
from src.data.loader import PROJECT_ROOT
from src.webapp.results_io import MODEL_ORDER, PHASE2_MODEL_ORDER, load_phase2_results, phase2_results_available


def render() -> None:
    st.title("Heart Disease Risk Modelling")
    st.subheader("Phase 1: six machine-learning classifiers on the Cleveland heart disease data")
    banner()
    st.markdown(
        "This app accompanies an academic study that compares classifiers for angiographically defined heart disease "
        "on the UCI Heart Disease benchmark: initially six classical classifiers on the Cleveland subset (303 patients), "
        "extended in Phase 2 to all four processed cohorts (920 patients) with a regularised MLP and a Bayesian-network--MLP stack. "
        "Use the pages on the left to explore the data, inspect the evaluation results, try the models on a hypothetical profile, "
        "and read how the models were built and where they fall short.")
    res = require_results()
    st.markdown("#### Headline results (nested cross-validation, 25 identical outer folds per model)")
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

    if phase2_results_available():
        st.markdown("---")
        st.markdown("#### Phase 2 Multi-Cohort Extension (920 patients across 4 centres, 8 models)")
        st.markdown(
            "Phase 2 evaluates all eight models (Bayesian Network, Naive Bayes, Logistic Regression, SVM, Random Forest, "
            "Gradient Boosting, MLP, and BN-MLP Stack) across all four UCI cohorts under library-managed Optuna TPE tuning.")
        p2_res = load_phase2_results()
        p2_ci = p2_res.get("ci_pooled", None)
        if p2_ci is not None and not p2_ci.empty:
            p2_summary = []
            for m in PHASE2_MODEL_ORDER:
                sub = p2_ci[p2_ci.model == m].set_index("metric")
                if "roc_auc" in sub.index and "accuracy" in sub.index:
                    p2_summary.append({
                        "Model": m,
                        "ROC AUC": f"{sub.loc['roc_auc', 'estimate']:.3f} [{sub.loc['roc_auc', 'ci_low']:.3f}, {sub.loc['roc_auc', 'ci_high']:.3f}]",
                        "Accuracy": f"{sub.loc['accuracy', 'estimate']:.3f} [{sub.loc['accuracy', 'ci_low']:.3f}, {sub.loc['accuracy', 'ci_high']:.3f}]",
                        "Brier score": f"{sub.loc['brier', 'estimate']:.3f}" if "brier" in sub.index else "N/A",
                        "Log loss": f"{sub.loc['log_loss', 'estimate']:.3f}" if "log_loss" in sub.index else "N/A",
                    })
            if p2_summary:
                import pandas as pd
                st.dataframe(pd.DataFrame(p2_summary), hide_index=True, use_container_width=True)
        st.caption("Detailed pooled, cross-cohort hospital shift, and ablation tables are available on the Model Results page.")

    footer()
