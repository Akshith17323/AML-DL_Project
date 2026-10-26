"""Saved evaluation results: tables, curves, calibration, exploratory threshold view."""
import altair as alt
import pandas as pd
import streamlit as st

from app.views.common import MODEL_COLOR, banner, footer, require_results
from src.data.loader import PROJECT_ROOT
from src.webapp.content import CALIBRATION_NOTE
from src.webapp.results_io import MODEL_ORDER, calibration_points, curve_points, metrics_at_threshold


def _pm(df, model, col, d=3):
    return f"{df.loc[model, col + '_mean']:.{d}f} ± {df.loc[model, col + '_sd']:.{d}f}"


def render() -> None:
    st.title("Model results")
    banner()
    res = require_results()
    st.caption("Out-of-fold results from nested cross-validation: stratified 5-fold x 5 repeats = 25 identical outer folds per model, "
               "hyper-parameters tuned inside each training fold, threshold 0.5. Values are read from `results/metrics/`.")
    t_sum, t_curve, t_cal, t_thr, t_diff = st.tabs(["Summary tables", "ROC and PR curves", "Calibration", "Threshold explorer", "Paired differences"])
    with t_sum:
        f = res["fold"]
        f = f[f.threshold_scheme == "fixed_0.5"].set_index("model").reindex(MODEL_ORDER)
        cols = [("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                ("precision", "Precision"), ("f1", "F1"), ("brier", "Brier"), ("log_loss", "Log loss")]
        st.markdown("**Fold-level mean ± SD over 25 outer folds (about 60 patients each)**")
        st.dataframe(pd.DataFrame({h: [_pm(f, m, c) for m in MODEL_ORDER] for c, h in cols}, index=MODEL_ORDER))
        ci = res["ci"]
        st.markdown("**Pooled out-of-fold estimate with 95% patient-level bootstrap interval**")
        rows = {}
        for m in MODEL_ORDER:
            r = ci[ci.model == m].set_index("metric")
            rows[m] = {h: f"{r.loc[k, 'estimate']:.3f} [{r.loc[k, 'ci_low']:.3f}, {r.loc[k, 'ci_high']:.3f}]"
                       for k, h in [("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("brier", "Brier"),
                                    ("sensitivity", "Sensitivity"), ("specificity", "Specificity")]}
        st.dataframe(pd.DataFrame(rows).T)
        st.caption("The fold-level SD understates uncertainty about new patients (training sets overlap); the bootstrap interval resamples patients, not the training procedure.")
        st.image(str(PROJECT_ROOT / "results/figures/confusion_matrices.png"), caption="Mean confusion counts per repeat at threshold 0.5 (top) and the training-fold Youden threshold (bottom)")
    with t_curve:
        repeat = st.selectbox("Repeat of the 5x5 cross-validation", [0, 1, 2, 3, 4], key="curve_repeat")
        for kind, title, xl, yl in (("roc", "ROC curve", "False positive rate (1 - specificity)", "True positive rate (sensitivity)"),
                                    ("pr", "Precision-recall curve", "Recall (sensitivity)", "Precision")):
            pts = curve_points(res["oof"], kind, repeat)
            st.altair_chart(alt.Chart(pts).mark_line(interpolate="step-after").encode(
                alt.X("x:Q", title=xl, scale=alt.Scale(domain=[0, 1])), alt.Y("y:Q", title=yl, scale=alt.Scale(domain=[0, 1])),
                alt.Color("model:N", scale=MODEL_COLOR, title="Model"), tooltip=["model", alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("y:Q", format=".3f")]
            ).properties(title=title, height=330), use_container_width=True)
        st.caption("The curves overlap closely; the paired bootstrap finds no AUC difference distinguishable from zero.")
    with t_cal:
        cal = calibration_points(res["oof"])
        diag = pd.DataFrame({"predicted": [0, 1], "observed": [0, 1]})
        line = alt.Chart(diag).mark_line(strokeDash=[4, 4], color="black").encode(x="predicted:Q", y="observed:Q")
        pts = alt.Chart(cal).mark_line(point=True).encode(
            alt.X("predicted:Q", title="Mean predicted probability (decile bins)", scale=alt.Scale(domain=[0, 1])),
            alt.Y("observed:Q", title="Observed fraction with disease", scale=alt.Scale(domain=[0, 1])),
            alt.Color("model:N", scale=MODEL_COLOR, title="Model"), tooltip=["model", alt.Tooltip("predicted:Q", format=".3f"), alt.Tooltip("observed:Q", format=".3f")])
        st.altair_chart((line + pts).properties(height=380), use_container_width=True)
        p = res["pooled"]
        p = p[p.threshold_scheme == "fixed_0.5"].set_index("model").reindex(MODEL_ORDER)
        st.dataframe(pd.DataFrame({"ECE": [_pm(p, m, "ece") for m in MODEL_ORDER],
                                   "Calibration slope": [_pm(p, m, "calibration_slope", 2) for m in MODEL_ORDER],
                                   "Calibration intercept": [_pm(p, m, "calibration_intercept", 2) for m in MODEL_ORDER]}, index=MODEL_ORDER))
        st.info(CALIBRATION_NOTE + " A slope below 1 means over-confident; 1 is ideal.")
    with t_thr:
        st.markdown("Move the threshold to see the sensitivity/specificity trade-off on the pooled out-of-fold predictions.")
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05, key="thr")
        m = metrics_at_threshold(res["oof"], thr)
        rate, count = st.column_config.NumberColumn(format="%.3f"), st.column_config.NumberColumn(format="%.1f")
        st.dataframe(m[["sensitivity", "specificity", "precision", "f1", "tn", "fp", "fn", "tp"]],
                     column_config={**{c: rate for c in ("sensitivity", "specificity", "precision", "f1")}, **{c: count for c in ("tn", "fp", "fn", "tp")}})
        st.caption("tn / fp / fn / tp are mean patient counts per repeat (303 patients in total).")
        st.warning("Exploratory only. Picking a threshold by looking at these labels is optimistic. The study fixed 0.5 in advance, and a "
                   "threshold chosen on training data (Youden) did not improve sensitivity for any model.")
    with t_diff:
        d = res["diff"]
        d = d[d.metric.isin(["roc_auc", "pr_auc", "brier", "log_loss"])].copy()
        d["95% interval"] = d.apply(lambda r: f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
        d["includes 0"] = d.ci_includes_zero.map({True: "yes", False: "no"})
        st.dataframe(d[["comparison", "metric", "estimate", "95% interval", "includes 0"]].round(3), hide_index=True)
        st.caption("First model minus second. Many comparisons were examined without multiplicity correction, so intervals that exclude zero "
                   "(log loss) are suggestive, not confirmatory.")
    footer()
