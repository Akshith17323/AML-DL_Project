"""Interactive demonstration: enter a hypothetical profile and see the six models' probabilities."""
import pandas as pd
import streamlit as st
import altair as alt

from app.views.common import MODEL_COLOR, banner, footer, get_models, require_data
from src.webapp.content import PROBABILITY_NOTE, POPULATION_NOTE
from src.webapp.inference import (CONTINUOUS, LABELS, OPTIONAL, OPTIONS, STEP, default_values, validate_inputs)
from src.data.eda import PLAUSIBLE
from src.data.loader import FEATURE_COLUMNS

UNKNOWN = "__unknown__"


def _widget(c, defaults):
    if c in CONTINUOUS:
        lo, hi = PLAUSIBLE[c]
        if c == "oldpeak":
            return st.number_input(LABELS[c], float(lo), float(hi), float(defaults[c]), STEP[c], format="%.1f", key=f"in_{c}")
        return st.number_input(LABELS[c], int(lo), int(hi), int(defaults[c]), int(STEP[c]), key=f"in_{c}")
    opts = list(OPTIONS[c])
    if c in OPTIONAL:
        opts = [UNKNOWN] + opts
        fmt = lambda v, c=c: "Unknown (use training mode)" if v == UNKNOWN else OPTIONS[c][v]
        idx = opts.index(int(defaults[c]))
    else:
        fmt = lambda v, c=c: OPTIONS[c][v]
        idx = opts.index(int(defaults[c]))
    v = st.selectbox(LABELS[c], opts, index=idx, format_func=fmt, key=f"in_{c}")
    return None if v == UNKNOWN else v


def render() -> None:
    st.title("Risk estimator (educational demo)")
    banner()
    require_data()
    models = get_models()
    st.markdown(POPULATION_NOTE)
    defaults = default_values(models.train_X)
    st.caption("Starting values are the training-data medians / most frequent categories (a typical patient in this cohort).")
    with st.form("profile"):
        cols = st.columns(3)
        values = {}
        for i, c in enumerate(FEATURE_COLUMNS):
            with cols[i % 3]:
                values[c] = _widget(c, defaults)
        submitted = st.form_submit_button("Estimate with the six models", type="primary")
    if submitted:
        st.session_state["profile_values"] = values
    values = st.session_state.get("profile_values")
    if values is None:
        st.info("Adjust the values and press **Estimate**. Nothing you enter is stored or sent anywhere; it is discarded when you close the page.")
        footer()
        return
    row, errors, warnings = validate_inputs(values, models.train_X)
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)
    if row is None:
        footer()
        return
    probs = models.predict(row)
    st.markdown("#### Estimated probability of disease (num 1-4) under each model")
    st.caption("For the values last submitted. After changing any input, press **Estimate** again.")
    items = list(probs.items())
    for start in range(0, len(items), 3):
        for col, (m, p) in zip(st.columns(3), items[start:start + 3]):
            col.metric(m, f"{100 * p:.1f}%")
    spread = max(probs.values()) - min(probs.values())
    st.altair_chart(alt.Chart(pd.DataFrame({"model": list(probs), "probability": list(probs.values())})).mark_bar().encode(
        alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="P(disease)"), alt.Y("model:N", title=None, sort=list(probs)),
        alt.Color("model:N", scale=MODEL_COLOR, legend=None), tooltip=["model", alt.Tooltip("probability:Q", format=".3f")]).properties(height=220),
        use_container_width=True)
    if spread >= 0.2:
        st.warning(f"The models disagree by {100 * spread:.0f} percentage points for this profile. In the study, differences in ranking between "
                   "models were within noise, so large disagreement signals that the profile is uncertain or unusual.")
    st.info(PROBABILITY_NOTE)
    unknown = [c for c in OPTIONAL if values.get(c) is None]
    if unknown:
        st.caption("Unknown " + " and ".join(f"`{c}`" for c in unknown) + " was replaced by the training-data mode in every model, "
                   "exactly as in the evaluation; this ignores uncertainty about the missing value.")
    with st.expander("How these numbers were produced"):
        st.markdown(
            "The six models are the study's pipelines (preprocessing + model) refitted on all 303 rows for this demonstration. "
            f"Hyper-parameters: the {models.hyperparameter_source}: `{models.hyperparameters}`. "
            "No performance claim is made for these refitted models; the performance figures come from the nested cross-validation on the "
            "*Model results* page. The refitted models were not evaluated on any held-out data.")
        st.dataframe(row.T.rename(columns={0: "value entered"}))
    footer()
