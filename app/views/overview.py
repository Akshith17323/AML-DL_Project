"""Overview page."""
import streamlit as st

from app.views.common import banner, footer, require_results
from src.webapp.results_io import MODEL_ORDER


def render() -> None:
    st.title("Heart Disease Risk Modelling")
    st.subheader("Phase 1: six machine-learning classifiers on the Cleveland heart disease data")
    banner()
    st.markdown(
        "This app accompanies an academic study that compares six classifiers for angiographically defined heart disease "
        "on the Cleveland subset of the UCI Heart Disease data (303 patients): a Bayesian network, Naive Bayes, "
        "Logistic Regression, an SVM, a Random Forest and Gradient Boosting. Use the pages on the left to explore the "
        "data, inspect the evaluation results, try the models on a hypothetical profile, and read how the models were "
        "built and where they fall short.")
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
    footer()
