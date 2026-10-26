"""Methodology, interpretation and limitations."""
import streamlit as st

from app.views.common import banner, footer
from src.data.loader import PROJECT_ROOT
from src.models.bayesian_network import edge_table


def _img(rel, caption):
    p = PROJECT_ROOT / rel
    if p.exists():
        st.image(str(p), caption=caption)
    else:
        st.info(f"Figure not found: {rel}")


def render() -> None:
    st.title("Methods, interpretation and limitations")
    banner()
    t1, t2, t3, t4, t5 = st.tabs(["Pipeline and models", "Bayesian network", "Evaluation design", "Interpretation", "Limitations and ethics"])
    with t1:
        st.markdown(
            "**Target.** `num = 0` -> 0, `num` in {1,2,3,4} -> 1 (164 negative, 139 positive).\n\n"
            "**Preprocessing (fitted on training folds only).** Logistic Regression, SVM, Random Forest and Gradient Boosting: median / most-frequent "
            "imputation, standardisation, one-hot nominal features (20 inputs). Bayesian network and Naive Bayes: train-fold tertile bins for continuous "
            "features, integer-coded states, missing or unseen values set to the training mode (13 inputs).\n\n"
            "**Bayesian network.** Fixed, domain-informed graph; probability tables learned with Dirichlet smoothing; exact inference.\n\n"
            "**Naive Bayes.** Class -> every predictor; assumes predictors are independent given the class, which is violated here "
            "(for example oldpeak/slope, cp/exang), so its probabilities tend to be too extreme.\n\n"
            "**Logistic Regression.** A weighted sum of the inputs passed through the logistic (sigmoid) function; L2 penalty, strength tuned (C).\n\n"
            "**SVM.** RBF-kernel support vector machine (C and gamma tuned); Platt scaling turns its scores into probabilities.\n\n"
            "**Random Forest.** Many shallow decision trees on bootstrap samples, averaged (number of trees and depth tuned).\n\n"
            "**Gradient Boosting.** Small trees added one at a time, each correcting the previous ones (learning rate, number of trees, depth tuned).")
    with t2:
        st.markdown("The structure is a **domain-informed assumption** fixed before any evaluation. It is not learned from data and is not evidence of causation; "
                    "only the probability tables are estimated from the training fold.")
        _img("results/figures/bn_structure.png", "Assumed Bayesian-network structure")
        et = edge_table()[["parent", "child", "kind", "rationale"]]
        st.dataframe(et, hide_index=True)
        st.caption("domain = clinical reasoning accepted as an assumption; modelling = added to relax Naive Bayes independence; weak = plausible but weakly motivated.")
    with t3:
        st.markdown(
            "- 5-fold stratified cross-validation repeated 5 times (seed 42): 25 identical outer folds for every model.\n"
            "- Nested tuning: inner 5-fold CV inside each training fold picks the hyper-parameters (lowest log loss) from a small grid.\n"
            "- Decision threshold fixed at 0.5.\n"
            "- Metrics: accuracy, ROC AUC, sensitivity (recall), specificity, precision, F1, confusion matrix and Brier score.\n"
            "- Uncertainty: fold SDs and a 2000-sample paired patient bootstrap.\n"
            "- Integrity checks: every patient predicted once per repeat, valid probabilities, identical folds across models.")
    with t4:
        st.markdown("**Permutation feature importance** for Logistic Regression, Random Forest and Gradient Boosting: how much ROC AUC drops "
                    "on held-out patients when one predictor's values are shuffled.")
        _img("results/figures/feature_importance.png", "Permutation feature importance")
        st.warning("Attribution describes what the fitted model uses, not what causes disease. Correlated predictors share credit, the folds are small, "
                   "and the leading features come from invasive or specialist tests.")
    with t5:
        st.markdown(
            "- 303 patients, one centre and period; no external validation. Internal cross-validation cannot support clinical validity or generalisability.\n"
            "- Referral and selection bias: prevalence (45.9%) is not a population rate.\n"
            "- Predictors come from exercise testing, fluoroscopy and perfusion imaging, so the models do not apply to first-contact triage.\n"
            "- The binary target discards severity; sparse categories make table estimates unstable; 206 men vs 97 women limits subgroup conclusions; no fairness analysis was done.\n"
            "- Differences between most models are within noise.\n"
            "- **Ethics:** the data are a public research dataset (CC BY 4.0) whose documentation says names and social-security numbers were removed. "
            "A model like this must not be used for patient care; it would be exposed to automation bias and has not been validated prospectively or externally.")
    footer()
