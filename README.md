# Heart Disease Risk Modelling: Phase 1 baseline and Phase 2 implementation

## Phase 2 additions

Phase 2 is an extension of Phase 1, not a replacement project. Every Phase 1 path remains in place, while Phase 2 files sit beside their Phase 1 equivalents and add all four independent processed UCI cohorts (920 patients), a regularised PyTorch MLP, a cross-fitted Bayesian Network--MLP stack, OptunaSearchCV/TPE hyperparameter selection without a handwritten candidate loop, cross-cohort validation, ablations, and an IEEEtran report.

```bash
python -m src.data.verify_phase2
python -m src.data.eda_phase2
python -m src.evaluation.run_phase2 --pilot       # safe one-fold/one-trial execution check
python -m src.evaluation.phase2_ablations --pilot
python -m src.evaluation.make_phase2_report_assets
```

Open `notebooks/02_phase2_all_cohorts.ipynb` for the visible, executed four-cohort source audit and previews. `verify_phase2` also extracts all four processed source files into `data/raw/`; model code continues to validate and read the canonical archive members directly.

The pilot is not a final performance experiment. Full runs omit `--pilot` and write under `results/phase2/main/`; freeze the configured search budget before interpreting them. See `config.phase2.yaml` and `reports/phase2/`.

### How Phase 2 extends the Phase 1 structure

| Phase 1 file/path (retained) | Phase 2 extension beside it |
|---|---|
| `config.yaml` | `config.phase2.yaml` |
| `data/raw/processed.cleveland.data` | Hungarian, Switzerland, and VA processed files in the same folder |
| `notebooks/01_eda.ipynb` | `notebooks/02_phase2_all_cohorts.ipynb` |
| `src/data/{loader,verify,eda}.py/.ipynb` | `verify_phase2.py/.ipynb`, `eda_phase2.py/.ipynb`; the shared loader supports both phases |
| `src/models/` classical models | `mlp.py/.ipynb`, `hybrid.py/.ipynb`, `phase2.py/.ipynb`, `learning_curves_phase2.py/.ipynb` |
| `src/evaluation/run_evaluation.py/.ipynb` | `run_phase2.py/.ipynb`, `phase2_search.py/.ipynb`, `phase2_ablations.py/.ipynb` |
| `tests/test_*.py/.ipynb` | matching `test_phase2_*.py/.ipynb` files |
| `results/metrics/`, `results/figures/` | isolated `results/phase2/{pilot,main}/` outputs |
| `reports/project_report/`, `reports/presentation/` | `reports/phase2/` IEEE sources and generated tables |

No Phase 1 file is renamed or removed. As in Phase 1, every non-empty Python file under `app/`, `src/`, and `tests/` has an adjacent notebook with the same basename.

## Retained Phase 1 baseline

The detailed Phase 1 documentation remains below because Phase 2 extends it rather than overwriting it. Phase 1 compares six classifiers on the UCI Heart Disease Cleveland data: a domain-informed **Bayesian network**, **Naive Bayes**, **Logistic Regression**, an **SVM**, a **Random Forest** and **Gradient Boosting**.

> **Neither phase is a clinically validated diagnostic system.** Phase 1 uses 303 patients from one centre; Phase 2 uses 920 historical patients from four cohorts and includes cross-cohort evaluation. Neither must be used for medical decisions.

### Phase 1 at a glance
- **Question:** under one identical, leakage-safe nested cross-validation, how well do six classical machine-learning models predict angiographic heart disease, and does the choice of model matter?
- **Answer:** all six do well (pooled ROC AUC 0.890-0.910, accuracy 81.8%-84.9%). Five cannot be told apart; every paired difference whose 95% interval excludes zero involves Gradient Boosting, which is slightly behind.
- **What drives the predictions:** `ca` (vessels on fluoroscopy), `thal` (thallium scan) and `cp` (chest pain type), for every model examined.
- **Limits:** one centre, 303 patients, internal validation only; not clinical evidence.
- **Deliverables:** code and tests (`src/`, `tests/`), executed EDA notebook, a Jupyter notebook copy of every source file, the project report and slide deck (`reports/`), and a Streamlit web app (`app/`).

## Contents
1. [Team](#1-team)
2. [Quick start](#2-quick-start)
3. [Project layout](#3-project-layout)
4. [Data](#4-data)
5. [Methodology](#5-methodology)
6. [Results](#6-results)
7. [Feature importance](#7-feature-importance)
8. [Reproducibility](#8-reproducibility)
9. [Limitations](#9-limitations)
10. [Git workflow](#10-git-workflow)
11. [Licence, citation and deliverables](#11-licence-citation-and-deliverables)

---

## 1. Team

| Member | Phase 1 work | Explains in the viva |
|---|---|---|
| Ayush | Data loading, verification, EDA and notebook, preprocessing / feature engineering (`src/data/`, `src/features/`, `notebooks/01_eda.ipynb`) | Dataset quality and EDA, feature engineering |
| Vansh | Bayesian network (structure, conditional probability tables, inference), data splits, metrics, Methods page (`src/models/bayesian_network.py`, `src/evaluation/splits.py`, `metrics.py`, `app/views/methods.py`) | Theoretical rigour (probabilistic graphical models) |
| Hardik | Logistic Regression and SVM, model pipelines, the script that trains and compares all models (`src/models/pipelines.py`, `src/evaluation/run_evaluation.py`) | Model application (linear and kernel models) |
| Akshith | Random Forest and Gradient Boosting, hyper-parameter tuning with nested CV, feature importance (`src/evaluation/cv.py`, `src/evaluation/interpret.py`) | Model application (ensembles), tuning |
| Kabir | Naive Bayes, the web app adapted to the ML models (`src/models/naive_bayes.py`, `app/`, `src/webapp/`) | Model application, demo |

## 2. Quick start

Phase 1 was tested on Python 3.9.6; the Phase 2 extension and saved runs were tested on Python 3.11.16 (macOS).

```bash
# 1. environment (exact tested versions)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt        # or: pip install -r requirements.txt  (minimum versions)

# 2. Phase 1 dataset: put the ignored source archive at data/raw/heart+disease.zip, then
python -m src.data.verify                   # lists the archive, extracts processed.cleveland.data, validates it

# 3. run everything, in this order
python -m src.data.eda                      # exploratory analysis + figures            (seconds)
python -m src.evaluation.run_evaluation     # nested CV for all six models, metrics, figures (about 10 min)
python -m src.evaluation.interpret          # feature importance + Bayesian-network figure  (about 1 min)
python -m pytest -q                         # all tests; result-dependent tests skip if step 3 has not been run
streamlit run app/streamlit_app.py          # web app, http://localhost:8501 (see app/README.md)

# optional
python -m src.evaluation.run_evaluation --analyse   # recompute tables/figures from the saved predictions
python tools/py_to_notebooks.py             # regenerate the .ipynb copies after editing any .py file
python tools/run_notebooks.py               # execute every notebook and report failures
```

If `.venv` was copied from another machine or folder, its launcher scripts (`streamlit`, `pytest`) still point at the old path. Recreate it, or call the tools through the environment's Python: `.venv/bin/python -m streamlit run app/streamlit_app.py`.

**Notebooks.** Every non-empty `.py` file in `app/`, `src` and `tests/` has a generated Jupyter notebook next to it with the same name. The `.py` files are the source of truth (the app, tests and commands import them). The Phase 1 and Phase 2 evaluation notebooks default to no training, so *Run All* cannot accidentally repeat the full searches.

## 3. Project layout

```
config.yaml                       Phase 1 configuration
config.phase2.yaml                Phase 2 extension configuration
requirements.txt / *-lock.txt     shared minimum and exact dependencies
data/raw/                         source archive/docs and four processed cohort files
notebooks/01_eda.ipynb            retained Phase 1 Cleveland EDA
notebooks/02_phase2_all_cohorts.ipynb  executed Phase 2 four-cohort audit
src/data/                         shared loader + Phase 1 and Phase 2 verification/EDA
src/features/                     shared leakage-safe preprocessing
src/models/                       Phase 1 models + Phase 2 MLP and BN--MLP hybrid
src/evaluation/                   Phase 1 evaluation + Phase 2 search/runs/ablations
app/, src/webapp/                 retained Phase 1 Streamlit app and its logic
tests/                            Phase 1 regression tests + Phase 2 tests
results/metrics/, results/figures/  retained Phase 1 outputs
results/phase2/                   isolated Phase 2 pilot and main outputs
reports/project_report/, presentation/  retained Phase 1 submissions
reports/phase2/                   IEEE Phase 2 source, references, and generated tables
tools/                            shared notebook generation/execution tools
```

## 4. Data

**Phase 2 source.** UCI Machine Learning Repository, *Heart Disease* (https://archive.ics.uci.edu/dataset/45/heart+disease), archive `heart+disease.zip` (23 files, documented in `heart-disease.names`). Phase 2 uses exactly the four independent processed files: Cleveland (303 rows), Hungarian (294), Switzerland (123), and Long Beach VA (200), for 920 patients. Raw/reprocessed alternatives are not added because they represent the same cohorts. Hashes, licence, citation, and visible file paths are in `data/raw/README.md` and `results/phase2/main/metrics/data_verification.json`.

**Retained Phase 1 source.** The Phase 1 commands still use only `processed.cleveland.data`, preserving the original 303-patient baseline and its saved outputs.

**Schema** (no header, `?` = missing): `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num`.

**Phase 2 observed statistics** (`results/phase2/main/metrics/data_verification.json`): 920 rows, 411 negative and 509 positive. Missing predictor cells by cohort are Cleveland 6, Hungarian 782, Switzerland 273, and VA 698. The original target and cohort provenance are preserved; preprocessing handles missing values inside training folds.

**Target.** `num` is preserved. Binary target `target = 0` if `num = 0`, `1` if `num` in {1, 2, 3, 4}; other or missing values raise an error. Phase 1 has 164 negative / 139 positive; Phase 2 has 411 negative / 509 positive.

**Phase 1 EDA summary** (`notebooks/01_eda.ipynb`, descriptive only): no invalid or implausible values; binary target close to balanced; only one predictor pair has |Spearman| >= 0.5 (`oldpeak`-`slope`, 0.595); sparse levels (`restecg = 1`: 4 rows, `thal = 6`: 18, `ca = 3`: 20, `slope = 3`: 21, `cp = 1`: 23); `cp = 4` (asymptomatic) has the highest positive rate (0.729), which probably reflects referral into angiography. The Phase 2 cohort audit is in `notebooks/02_phase2_all_cohorts.ipynb`. `ca`, `thal`, `oldpeak`, `slope`, `exang`, `thalach` come from exercise/imaging work-ups, so performance would not transfer to initial triage. Correlation is not causation: confounding, selection and reverse relationships all produce the same patterns.

## 5. Methodology

### 5.1 Preprocessing (`src/features/preprocessing.py`)

| Type | Features |
|---|---|
| continuous | age, trestbps, chol, thalach, oldpeak |
| binary | sex, fbs, exang |
| ordinal | ca (0-3), slope (1 up, 2 flat, 3 down) |
| nominal | cp, restecg, thal (numeric codes carry no order) |

Both pipelines start with a guard that raises if `num` or `target` is passed as a predictor.

- **Numeric pipeline (Logistic Regression, SVM, Random Forest, Gradient Boosting):** median / most-frequent imputation, standardisation of continuous and ordinal features, one-hot nominal features (unseen category -> all zeros). 20 columns.
- **Discrete pipeline (Bayesian network, Naive Bayes):** continuous features cut into train-fold tertiles, everything integer-coded from the training categories; missing or unseen values become the training mode. 13 columns.

All data-dependent quantities are learned in `fit` on the training fold only; the evaluation refits the whole pipeline in every inner and outer fold.

### 5.2 Models

**Bayesian network** (`src/models/bayesian_network.py`, pgmpy 0.1.26): 14 nodes, 16 edges fixed before any evaluation (risk factors -> disease -> findings). Only the conditional probability tables are learned, with a Dirichlet prior (pseudo-count tuned). Inference is exact variable elimination. A missing value or an unseen state raises an error. Figure: `results/figures/bn_structure.png`.

| Edge | Kind | Rationale |
|---|---|---|
| age -> target | domain | Age is an established cardiovascular risk factor. |
| sex -> target | domain | Sex is an established risk factor for coronary disease. |
| chol -> target | domain | Serum cholesterol is an established risk factor. |
| age -> trestbps | weak | Resting blood pressure tends to rise with age (not made a direct parent of `target`: too many CPD cells for n = 303). |
| age -> fbs | weak | Fasting glucose tends to drift upward with age (weak; links `fbs` without enlarging the target CPD). |
| target -> cp | domain | Chest pain type is a symptom manifestation of disease. |
| target -> exang | domain | Exercise-induced angina is a manifestation of ischaemic disease. |
| cp -> exang | modelling | Both describe angina symptoms; relaxes the Naive Bayes independence assumption. |
| target -> restecg | domain | Resting ECG abnormalities can accompany disease. |
| target -> thal | domain | Perfusion defects on the scan reflect disease. |
| target -> ca | domain | Number of vessels with visible stenosis reflects disease extent. |
| target -> thalach | domain | Reduced peak heart rate is a sign of reduced exercise capacity. |
| age -> thalach | domain | Peak heart rate falls with age. |
| target -> oldpeak | domain | Exercise ST depression is a diagnostic sign of ischaemia. |
| target -> slope | domain | Peak-exercise ST slope is a diagnostic sign of ischaemia. |
| oldpeak -> slope | modelling | Both measure the exercise ST response (strongest predictor pair in EDA, Spearman 0.6). |

Kinds: *domain* = clinical reasoning taken as an assumption, *weak* = plausible but weakly motivated, *modelling* = added to relax Naive Bayes independence. The `oldpeak -> slope` edge was suggested by the exploratory analysis. The structure is **not** a causal model.

**Naive Bayes** (`src/models/naive_bayes.py`): `CategoricalNB` on the same discrete inputs; Laplace smoothing tuned.

**Logistic Regression** and **SVM** (`src/models/pipelines.py`): L2-regularised logistic regression (C tuned); RBF-kernel SVM (C and gamma tuned) with Platt scaling for probabilities.

**Random Forest** and **Gradient Boosting** (`src/models/pipelines.py`): scikit-learn ensembles; number of trees, depth and (for boosting) learning rate tuned.

### 5.3 Evaluation design (`src/evaluation/`)

- **Outer CV:** 5-fold stratified x 5 repeats = 25 identical outer folds for every model (seed 42).
- **Nested tuning:** inside each outer training fold, a stratified 5-fold inner CV over the grid below, selected by inner out-of-fold log loss; the best setting is refitted on the whole outer training fold. Outer labels are used only for scoring.
- **Threshold:** 0.5, fixed in advance.
- **Metrics:** accuracy, ROC AUC, PR AUC, sensitivity, specificity, precision, F1, confusion matrix, Brier score.
- **Uncertainty:** mean ± SD over the 25 folds; 2000-sample patient-level paired bootstrap (95%) for pooled metrics and differences between models.
- **Integrity checks** (every run): each patient predicted exactly once per repeat, valid probabilities, identical folds across models.

| Model | Grid | Chosen most often |
|---|---|---|
| Bayesian network | `pseudo_count` 0.5, 1, 2, 5, 10 | pseudo_count = 10 (25/25) |
| Naive Bayes | `alpha` 0.5, 1, 2, 5, 10 | alpha = 10 (25/25) |
| Logistic Regression | `C` 0.01, 0.1, 1, 10 | C = 1 (14/25) |
| SVM | `C` 0.1, 1, 10; `gamma` scale, 0.01 | C = 1, gamma = 0.01 (23/25) |
| Random Forest | `n_estimators` 200, 500; `max_depth` 3, 5, None | n_estimators = 200, max_depth = 5 (8/25) |
| Gradient Boosting | `learning_rate` 0.05, 0.1; `n_estimators` 100, 200; `max_depth` 2, 3 | learning_rate = 0.05, n_estimators = 100, max_depth = 2 (20/25) |

The Bayesian network and Naive Bayes chose the largest smoothing value in every fold. Their grids had already been widened from {0.5, 1, 2} for this run because an earlier run also chose its top value; they were not widened again after seeing these results (see Limitations).

## 6. Results

All numbers come from `results/metrics/*.csv` (threshold 0.5).

**Fold-level mean ± SD (25 outer folds, about 60 patients each)**

| Model | Accuracy | ROC AUC | Sensitivity | Specificity | Precision | F1 | Brier |
|---|---|---|---|---|---|---|---|
| Bayesian network | 0.829 ± 0.050 | 0.912 ± 0.036 | 0.793 ± 0.077 | 0.861 ± 0.068 | 0.833 ± 0.070 | 0.810 ± 0.057 | 0.121 ± 0.027 |
| Naive Bayes | 0.828 ± 0.049 | 0.907 ± 0.034 | 0.794 ± 0.088 | 0.857 ± 0.067 | 0.830 ± 0.072 | 0.808 ± 0.058 | 0.129 ± 0.028 |
| Logistic Regression | 0.847 ± 0.045 | 0.911 ± 0.036 | 0.780 ± 0.072 | 0.905 ± 0.057 | 0.878 ± 0.066 | 0.823 ± 0.054 | 0.120 ± 0.023 |
| SVM | 0.849 ± 0.046 | 0.910 ± 0.037 | 0.790 ± 0.078 | 0.899 ± 0.056 | 0.872 ± 0.066 | 0.826 ± 0.056 | 0.119 ± 0.026 |
| Random Forest | 0.834 ± 0.048 | 0.906 ± 0.037 | 0.778 ± 0.080 | 0.882 ± 0.053 | 0.850 ± 0.062 | 0.810 ± 0.059 | 0.126 ± 0.020 |
| Gradient Boosting | 0.818 ± 0.044 | 0.895 ± 0.047 | 0.772 ± 0.074 | 0.857 ± 0.056 | 0.824 ± 0.059 | 0.795 ± 0.052 | 0.131 ± 0.029 |

**Pooled out-of-fold, with patient-level bootstrap 95% CI**

| Model | Accuracy | ROC AUC | Brier |
|---|---|---|---|
| Bayesian network | 0.830 [0.785, 0.869] | 0.910 [0.876, 0.940] | 0.121 [0.096, 0.148] |
| Naive Bayes | 0.828 [0.785, 0.867] | 0.905 [0.870, 0.935] | 0.129 [0.102, 0.159] |
| Logistic Regression | 0.848 [0.809, 0.885] | 0.904 [0.869, 0.936] | 0.120 [0.097, 0.143] |
| SVM | 0.849 [0.809, 0.886] | 0.907 [0.873, 0.939] | 0.119 [0.096, 0.142] |
| Random Forest | 0.834 [0.793, 0.872] | 0.903 [0.868, 0.934] | 0.126 [0.107, 0.147] |
| Gradient Boosting | 0.818 [0.778, 0.857] | 0.890 [0.853, 0.923] | 0.131 [0.109, 0.155] |

**Mean confusion counts per repeat** (303 patients), TN / FP / FN / TP: Bayesian network 141.2 / 22.8 / 28.8 / 110.2; Naive Bayes 140.6 / 23.4 / 28.6 / 110.4; Logistic Regression 148.4 / 15.6 / 30.6 / 108.4; SVM 147.4 / 16.6 / 29.2 / 109.8; Random Forest 144.6 / 19.4 / 30.8 / 108.2; Gradient Boosting 140.6 / 23.4 / 31.6 / 107.4.

**Paired comparisons.** 12 of 15 ROC AUC differences and 12 of 15 accuracy differences have a 95% interval that includes zero. The ones that do not:

| Comparison | Metric | Difference [95% CI] |
|---|---|---|
| Bayesian Network - Gradient Boosting | ROC AUC | +0.020 [+0.003, +0.036] |
| Logistic Regression - Gradient Boosting | accuracy | +0.029 [+0.005, +0.052] |
| SVM - Gradient Boosting | accuracy | +0.030 [+0.007, +0.053] |
| SVM - Gradient Boosting | ROC AUC | +0.017 [+0.001, +0.032] |
| Random Forest - Gradient Boosting | accuracy | +0.016 [+0.001, +0.031] |
| Random Forest - Gradient Boosting | ROC AUC | +0.013 [+0.003, +0.023] |

All of them involve Gradient Boosting, so the other five models cannot be told apart. With 30 intervals examined, this is modest evidence, not proof.

## 7. Feature importance

`python -m src.evaluation.interpret` refits Logistic Regression, Random Forest and Gradient Boosting on the five repeat-0 outer folds (with the hyper-parameters selected in each fold; the refits reproduce the saved predictions exactly) and measures **permutation importance**: the drop in ROC AUC on the held-out patients when one predictor is shuffled (30 shuffles). Outputs: `results/metrics/feature_importance.csv`, `results/figures/feature_importance.png`.

| Rank | Logistic Regression | Random Forest | Gradient Boosting |
|---|---|---|---|
| 1 | ca (0.056) | thal (0.054) | ca (0.059) |
| 2 | thal (0.030) | ca (0.042) | thal (0.046) |
| 3 | cp (0.027) | cp (0.033) | cp (0.040) |
| 4 | thalach (0.012) | sex (0.008) | sex (0.017) |
| 5 | sex (0.011) | age (0.007) | age (0.012) |
| 6 | oldpeak (0.007) | oldpeak (0.006) | oldpeak (0.011) |

All three rank `ca`, `thal` and `cp` highest. Importance describes the fitted models, not causes; correlated predictors share credit.

## 8. Reproducibility

One project seed (`project.seed = 42`) drives the outer and inner folds, model randomness, the bootstrap and the permutations. Exact versions are in `requirements-lock.txt` and `results/metrics/evaluation_manifest.json` (Python 3.9.6, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, pgmpy 0.1.26, streamlit 1.50.0).

**Verified on 2026-10-26 (macOS, CPU):** `python -m pytest -q` passes; the full evaluation and `interpret` ran without errors; refitted models reproduce the saved out-of-fold predictions to within 1e-15; all five web pages open without errors. Not verified: other operating systems, Python versions or hardware.

## 9. Limitations

- Only 303 patients, one centre, one period; no external validation.
- Referral and selection bias: prevalence (45.9%) is not a population rate; asymptomatic chest pain has the highest disease rate.
- Predictors include exercise testing, fluoroscopy and perfusion imaging, so the models do not apply to first-contact screening.
- The binary target pools `num` 1-4; severity is discarded. Sparse categories make probability tables unstable. 206 men vs 97 women; no fairness analysis.
- Six rows with missing `ca`/`thal` are filled with the training mode.
- Bayesian network and Naive Bayes smoothing chose the top grid value in every fold.
- The Bayesian network structure is an assumption.
- pgmpy is pinned below 1.0 (1.0 needs Python >= 3.10 and renames `BayesianNetwork`). Python 3.9 is past end of life and some libraries print deprecation warnings.
- The notebooks are generated copies; after editing a `.py` file, run `python tools/py_to_notebooks.py`.

## 10. Git workflow

1. `main` is protected: no direct pushes; it must always pass `python -m pytest -q`.
2. Branch per task from an up-to-date `main` (`name/topic`), keep branches short-lived, commit small, in the imperative.
3. Open a pull request early; one approving review from another member and a green test run.
4. Never change evaluation settings, grids or model choices after looking at outer-fold results without declaring it as a new experiment; never tune on outer labels; regenerate (do not hand-edit) tables and figures.

## 11. Licence, citation and deliverables

- **Data:** UCI Heart Disease, CC BY 4.0; not redistributed here. Cite: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X. Principal investigator for the Cleveland data: Robert Detrano, M.D., Ph.D.
- **Code:** no licence file has been chosen yet.
- **Web app:** `streamlit run app/streamlit_app.py` serves five pages (Overview, Data explorer, Model results, Risk estimator (demo), Methods and limitations). Details in `app/README.md`.
- **Report:** `reports/project_report/Heart_Disease_Risk_Modelling_Report.docx`, about 15 pages, structured as the course sections (introduction, related work, data and EDA, preprocessing, theory, Bayesian network and Naive Bayes, linear/kernel/ensemble models, experimental design, results, interpretation, software, limitations, conclusion, references, team contributions). Fill in roll numbers, course and supervisor, then export to PDF from Word.
- **Presentation:** `reports/presentation/Heart_Disease_Risk_Modelling.pptx`, 18 slides with the presenter of each slide marked and speaker notes on every slide.
- The report and slides were generated from `results/metrics/`; if the evaluation is rerun, regenerate or re-check their numbers.
- **Phase 2** (not in this branch): the earlier MLP / SHAP work is in Git history (commit `a4ea3a8`).
