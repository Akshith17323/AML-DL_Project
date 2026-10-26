# Streamlit web interface

An educational front end for the study: explore the data, inspect the saved evaluation results, try the three models on a hypothetical profile, and read the methods and limitations. **It is an academic demonstration, not a medical device, and every page says so.**

## Run

```bash
# from the project root, with the environment from requirements-lock.txt and the dataset in data/raw/ (see data/raw/README.md)
python -m src.evaluation.run_evaluation        # once (about 30 min); the Results/Overview pages read results/metrics/
python -m src.evaluation.interpret             # once; adds the MLP attribution table and BN figure
streamlit run app/streamlit_app.py             # opens http://localhost:8501
```

Without the dataset the pages that need it show instructions instead of crashing; without saved results the Overview and Results pages say how to create them. The first visit to *Risk estimator* trains the three models on the 303 rows (about 2 s), after which they are cached.

## Pages
| Page | What it does | Data source |
|---|---|---|
| Overview | Headline ROC AUC with 95% CI, Brier score, qualitative findings | `results/metrics/bootstrap_*.csv` |
| Data explorer | Distributions, feature-vs-target rates, Spearman heat map, missingness (descriptive only) | `data/raw/processed.cleveland.data` |
| Model results | Fold-level and pooled tables, ROC/PR curves by repeat, calibration plot and slopes, exploratory threshold slider, paired differences | `results/metrics/*.csv` |
| Risk estimator (demo) | 13 inputs -> probability under each model, disagreement warning, optional SHAP explanation for the MLP | models refitted in-process |
| Methods and limitations | Pipeline, BN structure and edge table, evaluation design, attribution, limitations/ethics, report download | `results/figures/`, `reports/main.pdf` |

## Design decisions
- **Logic is separate from UI.** `src/webapp/` (content, `inference.py`, `results_io.py`) is plain Python with unit tests; `app/views/*.py` only draws widgets. Each page is a function, which is also what makes pages testable one at a time.
- **No new numbers.** Overview and Results display values read from `results/metrics/`; nothing is re-estimated in the app. The threshold slider recomputes hard-label metrics from the saved out-of-fold predictions and is labelled exploratory.
- **Final demo models.** The estimator refits the study's pipelines on all 303 rows (no held-out data exists for them, so no performance is claimed). Hyper-parameters are the *joint setting selected most often by the inner cross-validation* in the nested evaluation (BN pseudo-count 2.0, Naive Bayes alpha 2.0, MLP weight decay 0.01 and dropout 0.5); there is no new tuning.
- **Same preprocessing as the study.** Unknown `ca`/`thal` is replaced by the training mode, exactly as in the evaluation, and the page says so. Inputs are validated against the documented value domains and plausible ranges; values outside the training range trigger a warning that the models are extrapolating.
- **Honest framing on the estimator page.** Population note (referred-for-angiography cohort, 45.9% positive, specialist-test inputs), calibration note (BN and NB over-confident), and a warning when the models disagree by 20 percentage points or more.
- **Explanations.** Kernel SHAP on the 13 raw predictors for one profile (30 background rows, 300 samples, seeded), with a caveat that it describes the model, not causes.
- **Privacy.** Entered values live only in the browser session's `st.session_state`; nothing is written to disk or logged (tested by comparing a file snapshot of the project before and after use). Usage-statistics telemetry is disabled in `.streamlit/config.toml`. There are no file uploads.
- **Accessibility.** Colour-blind-safe (Okabe-Ito) palette; models are always identified by labels as well as colour.

## Research notes (what informed the design)
Checked on 2026-10-22 against the Streamlit documentation and the installed package.
- **Streamlit 1.50.0** requires Python >= 3.9 (not 3.9.7), so it runs in the project's Python 3.9.6 environment; `pip check` reports no conflicts (it pins `packaging<26`, which pip downgraded to 25.0).
- **Navigation:** `st.navigation` with `st.Page` accepts plain callables; the entry file acts as the router and must call `.run()` on the selected page. The default page lives at the root URL, so it must not be given its own `url_path` (otherwise `/overview` shows "Page not found").
- **Caching:** `st.cache_resource` returns the *same shared object* to every session without copying and requires thread-safe return values; `st.cache_data` returns a copy. Hence the fitted models are cached with `cache_resource` and every prediction or explanation goes through a lock, while dataframes use `cache_data`.
- **Testing:** `streamlit.testing.v1.AppTest` simulates a running app without a browser, one page at a time, and does not rerun on its own. The documentation page reached was for a newer release (labelled v1.66); in the installed 1.50 `AppTest` has no `form_submit_button` collection, and the form's submit button appears under `button`. Tests therefore select it by label.
- **Not found / not done:** the docs pages consulted did not list a minimum version for `AppTest`, and they do not cover every unsupported feature. Browser-level behaviour (layout, charts) is not covered by `AppTest`; it was checked once by driving a real Chrome through its DevTools protocol (screenshots of every page, a form submission and a SHAP request), which is a manual check, not an automated test.

## Tests
`tests/test_webapp.py` (34 tests): input validation and error messages, defaults, hyper-parameter selection rule and fallback, determinism, direction sanity for extreme profiles, SHAP shape and reproducibility, results helpers matching the saved summaries, every page rendering with the disclaimer and without exceptions, the entry point, estimator submit/unknown/explain flows, and no files written during use.

## Deployment (not tested)
The dataset is deliberately not in Git (CC BY 4.0, attribution required; see `data/raw/README.md`), and the Results/Overview pages need `results/metrics/` and `results/figures/`. A hosted deployment (for example Streamlit Community Cloud or a container) must therefore either (a) mount the archive and results at start-up, or (b) commit `processed.cleveland.data` with the attribution shown in the app, which the licence allows but the team should decide. No Dockerfile is provided because it could not be tested here. Before exposing the app publicly: keep the disclaimer on every page, do not add logging of inputs, and add the team's names and a code licence.

## Known limitations
- Single-user local use was tested; concurrent users were not load-tested (a lock serialises predictions).
- The SHAP explanation is an approximation and takes 1-2 s.
- Python 3.9 is past end of life; Streamlit and its dependencies emit deprecation warnings under it.
