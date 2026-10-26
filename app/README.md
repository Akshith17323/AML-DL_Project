# Streamlit web interface

An educational front end for the Phase 1 study: explore the data, look at the saved evaluation results, try the six models on a hypothetical patient, and read the methods and limitations. **It is an academic demonstration, not a medical device, and every page says so.**

## Run

```bash
# from the project root, with the environment from requirements-lock.txt and the dataset in data/raw/ (see data/raw/README.md)
python -m src.evaluation.run_evaluation        # once (about 10 min); the Overview and Results pages read results/metrics/
python -m src.evaluation.interpret             # once; adds the feature-importance and Bayesian-network figures
streamlit run app/streamlit_app.py             # opens http://localhost:8501
```

Without the dataset, the pages that need it show instructions instead of crashing. Without saved results, the Overview and Results pages say how to create them. The first visit to *Risk estimator* trains the six models on the 303 rows (a few seconds); after that they are cached.

## Pages
| Page | What it does | Data source |
|---|---|---|
| Overview | ROC AUC with 95% interval and accuracy for each model, short findings | `results/metrics/bootstrap_ci.csv` |
| Data explorer | Distributions, feature-vs-target rates, correlation heat map, missing values (descriptive only) | `data/raw/processed.cleveland.data` |
| Model results | Summary tables, confusion matrices, ROC/PR curves, threshold slider, paired differences | `results/metrics/*.csv` |
| Risk estimator (demo) | 13 inputs -> probability of disease under each of the six models, with a warning when they disagree | models refitted in the app |
| Methods and limitations | Preprocessing, the six models, the Bayesian-network structure, evaluation design, feature importance, limitations and ethics | `results/figures/` |

## Design decisions
- **Logic is separate from the UI.** `src/webapp/` (texts, `inference.py`, `results_io.py`) is plain Python with unit tests; `app/views/*.py` only draws widgets.
- **No new numbers.** Overview and Results show values read from `results/metrics/`; nothing is re-estimated in the app. The threshold slider recomputes metrics from the saved out-of-fold predictions and is labelled exploratory.
- **Demo models.** The estimator refits the study's pipelines on all 303 rows, using for each model the hyper-parameter setting chosen most often by the nested cross-validation. No performance is claimed for these refitted models.
- **Same preprocessing as the study.** Unknown `ca`/`thal` is replaced by the training mode, as in the evaluation, and the page says so. Inputs are checked against the documented value ranges; values outside the training range trigger an "extrapolating" warning.
- **Privacy.** Entered values live only in the browser session; nothing is written to disk or logged (a test compares a snapshot of the project folder before and after use). Usage telemetry is off in `.streamlit/config.toml`.
- **Accessibility.** Colour-blind-safe (Okabe-Ito) palette; models are always labelled, never identified by colour alone.

## Tests
`tests/test_webapp.py`: input validation, default values, hyper-parameter selection (including restoring integer and `None` values), determinism, sensible direction for extreme profiles, results helpers matching the saved summaries, every page rendering with the disclaimer and without errors, the estimator flows, and no files written during use.

## Known limitations
- Tested for single-user local use only.
- Deployment is not set up: the dataset is not in Git, so a host would have to provide it.
- Python 3.9 is past end of life; Streamlit and its dependencies print deprecation warnings under it.
