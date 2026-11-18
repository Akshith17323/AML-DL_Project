"""Interactive demonstration: enter a hypothetical profile and see predicted probabilities."""
import altair as alt
import pandas as pd
import streamlit as st

from app.views.common import (
    MODEL_COLOR,
    PHASE2_MODEL_COLOR,
    banner,
    footer,
    get_models,
    get_phase2_models,
    require_data,
)
from src.data.eda import PLAUSIBLE
from src.data.loader import FEATURE_COLUMNS
from src.webapp.content import POPULATION_NOTE, PROBABILITY_NOTE
from src.webapp.inference import (
    CONTINUOUS,
    LABELS,
    OPTIONAL,
    OPTIONS,
    STEP,
    default_values,
    validate_inputs,
)

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

    suite = st.radio(
        "Demonstration model suite",
        ["Phase 2: Eight models (All four cohorts, 920 patients - incl. regularised MLP and Stack)",
         "Phase 1: Six models (Cleveland cohort, 303 patients)"],
        index=0,
        horizontal=True,
    )

    is_phase2 = "Phase 2" in suite
    if is_phase2:
        models = get_phase2_models()
        palette_scale = PHASE2_MODEL_COLOR
        btn_label = "Estimate with the eight models"
    else:
        models = get_models()
        palette_scale = MODEL_COLOR
        btn_label = "Estimate with the six models"

    st.markdown(POPULATION_NOTE)
    defaults = default_values(models.train_X)
    st.caption("Starting values are the training-data medians / most frequent categories (a typical patient in this cohort).")

    with st.form("profile"):
        cols = st.columns(3)
        values = {}
        for i, c in enumerate(FEATURE_COLUMNS):
            with cols[i % 3]:
                values[c] = _widget(c, defaults)
        submitted = st.form_submit_button(btn_label, type="primary")

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
    st.caption("For the values last submitted. Models are cached offline and do not rerun Optuna on prediction. After changing any input, press **Estimate** again.")

    items = list(probs.items())
    for start in range(0, len(items), 3):
        for col, (m, p) in zip(st.columns(3), items[start:start + 3]):
            col.metric(m, f"{100 * p:.1f}%")

    spread = max(probs.values()) - min(probs.values())
    st.altair_chart(
        alt.Chart(pd.DataFrame({"model": list(probs), "probability": list(probs.values())}))
        .mark_bar()
        .encode(
            alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="P(disease)"),
            alt.Y("model:N", title=None, sort=list(probs)),
            alt.Color("model:N", scale=palette_scale, legend=None),
            tooltip=["model", alt.Tooltip("probability:Q", format=".3f")],
        )
        .properties(height=240 if is_phase2 else 200),
        use_container_width=True,
    )

    if spread >= 0.2:
        st.warning(
            f"The models disagree by {100 * spread:.0f} percentage points for this profile. In the study, differences in ranking between "
            "models were small, so large disagreement signals that the profile is uncertain, unusual, or affected by hospital shift."
        )

    st.info(PROBABILITY_NOTE)
    unknown = [c for c in OPTIONAL if values.get(c) is None]
    if unknown:
        st.caption(
            "Unknown " + " and ".join(f"`{c}`" for c in unknown) + " was replaced by the training-data mode in every model, "
            "exactly as in the evaluation; this ignores uncertainty about the missing value."
        )

    with st.expander("How these demo predictions were produced (separation from CV performance)"):
        st.markdown(
            f"**Demonstration cohort:** {len(models.train_X)} patients ({'all four cohorts' if is_phase2 else 'Cleveland cohort'}).\n\n"
            f"**Hyperparameters:** Selected from {models.hyperparameter_source}.\n\n"
            f"```json\n{pd.Series(models.hyperparameters).to_json(indent=2)}\n```\n\n"
            "> **Strict Separation Notice:** These demo predictions are produced by offline-fitted models cached in memory. "
            "No hyperparameter search (Optuna) is executed during user interaction. "
            "Crucially, these demo models were **not** evaluated on held-out test data; all reported performance figures and confidence intervals "
            "come strictly from the nested out-of-fold cross-validation on the **Model results** page."
        )
        st.dataframe(row.T.rename(columns={0: "value entered"}))

    footer()
