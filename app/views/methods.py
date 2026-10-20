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
            "**Preprocessing (fitted on training folds only).** MLP: mode/median imputation, standardisation, one-hot nominal features (20 inputs). "
            "Bayesian network and Naive Bayes: train-fold tertile bins for continuous features, integer-coded states, missing or unseen values set to the training mode.\n\n"
            "**Naive Bayes.** Class -> every predictor; assumes predictors are conditionally independent given the class, which is violated here "
            "(for example oldpeak/slope, cp/exang), so its probabilities tend to be over-confident.\n\n"
            "**MLP.** 20 -> 16 -> 8 -> 1, ReLU, dropout, 481 parameters, binary cross-entropy, Adam with weight decay, early stopping on a hold-out of the training fold.")
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
            "- Nested tuning: inner 5-fold CV inside each training fold, selected by log loss (small grids).\n"
            "- Hard-label threshold fixed at 0.5; a Youden threshold from training data was a secondary analysis.\n"
            "- Metrics: ROC AUC, PR AUC, sensitivity, specificity, precision, F1, confusion matrix, Brier, log loss, ECE, calibration slope/intercept.\n"
            "- Uncertainty: fold SDs and a 2000-sample paired patient bootstrap.\n"
            "- Integrity checks: every patient predicted once per repeat, valid probabilities, identical folds across models.")
    with t4:
        st.markdown("**MLP feature attribution** (SHAP and permutation importance on held-out rows). Both rank `ca`, `cp`, `thal` highest, then `oldpeak`.")
        _img("results/figures/mlp_attribution_bars.png", "SHAP and permutation importance")
        st.warning("Attribution describes what the fitted model uses, not what causes disease. Correlated predictors share credit, the folds are small, "
                   "and the leading features come from invasive or specialist tests.")
    with t5:
        st.markdown(
            "- 303 patients, one centre and period; no external validation. Internal cross-validation cannot support clinical validity or generalisability.\n"
            "- Referral and selection bias: prevalence (45.9%) is not a population rate.\n"
            "- Predictors come from exercise testing, fluoroscopy and perfusion imaging, so the models do not apply to first-contact triage.\n"
            "- The binary target discards severity; sparse categories make table estimates unstable; 206 men vs 97 women limits subgroup conclusions; no fairness analysis was done.\n"
            "- Differences between models are within noise; calibration differences are exploratory.\n"
            "- The smoothing grid for the Bayesian network and Naive Bayes was truncated at its upper end in every fold.\n"
            "- **Ethics:** the data are a public research dataset (CC BY 4.0) whose documentation says names and social-security numbers were removed. "
            "A model like this must not be used for patient care; it would be exposed to automation bias and has not been validated prospectively or externally.")
        pdf = PROJECT_ROOT / "reports/main.pdf"
        if pdf.exists():
            st.download_button("Download the IEEE-format report (PDF)", pdf.read_bytes(), file_name="heart_disease_risk_modelling_report.pdf", mime="application/pdf")
    footer()
