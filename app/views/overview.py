"""Overview page."""
import streamlit as st

from app.views.common import banner, footer, require_results
from src.webapp.results_io import MODEL_ORDER


def render() -> None:
    st.title("Heart Disease Risk Modelling")
    st.subheader("Bayesian Network vs Naive Bayes vs feed-forward neural network")
    banner()
    st.markdown(
        "This app accompanies an academic study that compares three probabilistic classifiers for angiographically "
        "defined heart disease on the Cleveland subset of the UCI Heart Disease data (303 patients). "
        "Use the pages on the left to explore the data, inspect the evaluation results, try the models on a "
        "hypothetical profile, and read how the models were built and where they fall short.")
    res = require_results()
    st.markdown("#### Headline results (nested cross-validation, 25 identical outer folds per model)")
    ci = res["ci"]
    cols = st.columns(3)
    for col, model in zip(cols, MODEL_ORDER):
        r = ci[ci.model == model].set_index("metric")
        with col:
            st.metric(f"{model}: ROC AUC", f"{r.loc['roc_auc', 'estimate']:.3f}",
                      help=f"95% bootstrap interval [{r.loc['roc_auc', 'ci_low']:.3f}, {r.loc['roc_auc', 'ci_high']:.3f}]")
            st.caption(f"95% CI [{r.loc['roc_auc', 'ci_low']:.3f}, {r.loc['roc_auc', 'ci_high']:.3f}] · "
                       f"Brier {r.loc['brier', 'estimate']:.3f}")
    diff = res["diff"]
    auc_all_zero = bool(diff[diff.metric.isin(["roc_auc", "pr_auc"])].ci_includes_zero.all())
    st.markdown(
        "- **Ranking:** " + ("the three models cannot be told apart; every paired ROC/PR AUC difference has an interval that includes zero."
                             if auc_all_zero else "see the Model results page for the paired differences.") + "\n"
        "- **Probability quality:** the MLP's probabilities were better calibrated; the Bayesian network and Naive Bayes "
        "were over-confident. This comparison is exploratory.\n"
        "- **Limits:** one centre, 303 patients, internal validation only. These numbers do not show clinical validity.")
    footer()
