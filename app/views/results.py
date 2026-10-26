"""Saved evaluation results: tables, curves, calibration, exploratory threshold view."""
import altair as alt
import pandas as pd
import streamlit as st

from app.views.common import MODEL_COLOR, banner, footer, require_results
from src.data.loader import PROJECT_ROOT
from src.webapp.results_io import MODEL_ORDER, curve_points, metrics_at_threshold


def _pm(df, model, col, d=3):
    return f"{df.loc[model, col + '_mean']:.{d}f} ± {df.loc[model, col + '_sd']:.{d}f}"


def render() -> None:
    st.title("Model results")
    banner()
    res = require_results()
    st.caption("Out-of-fold results from nested cross-validation: stratified 5-fold x 5 repeats = 25 identical outer folds per model, "
               "hyper-parameters tuned inside each training fold, threshold 0.5. Values are read from `results/metrics/`.")
    t_sum, t_curve, t_thr, t_diff = st.tabs(["Summary tables", "ROC and PR curves", "Threshold explorer", "Paired differences"])
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
        d = d[d.metric.isin(["roc_auc", "accuracy"])].copy()
        d["95% interval"] = d.apply(lambda r: f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
        d["includes 0"] = d.ci_includes_zero.map({True: "yes", False: "no"})
        st.dataframe(d[["comparison", "metric", "estimate", "95% interval", "includes 0"]].round(3), hide_index=True)
        st.caption("First model minus second, 95% paired patient-level bootstrap interval. If the interval includes 0, the data cannot tell "
                   "the two models apart. Many pairs are compared, so a single interval that just excludes 0 is weak evidence.")
    footer()
