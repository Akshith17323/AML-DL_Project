# Team push plan: Phase 2 — corrected from professor feedback

**Status: proposed work, not implemented results.** Continue the existing Git repository in `heart-disease-risk-modelling/`. Keep `TEAM_PUSH_PLAN.md` as the Phase 1 history; do not initialise Git again or overwrite its saved results.

## 1. Confirmed requirements

The professor's Phase 1 feedback, as reported by the team, is:

1. Submit the Phase 2 report in **IEEE standard format**.
2. Do **not write manual loops for hyperparameter selection**.
3. Use **all available heart-disease datasets**, not only Cleveland.
4. Integrate the existing ML work with the Phase 2 deep-learning component and use an **advanced hyperparameter-selection method**.

The official Phase 2 brief additionally requires architecture logic, regularisation, ablation studies, technical validation, literature, presentation, viva, and code quality. The plan below implements integration as a Bayesian Network–MLP stack and advanced selection as OptunaSearchCV with TPE. These are technically appropriate proposed implementations; no particular hybrid architecture or optimisation library was named in the feedback supplied here.

## 2. What the Phase 2 project will do

### Data: use all four independent UCI cohorts

Use the four consistently formatted processed patient cohorts in `data/raw/heart+disease.zip`:

| Cohort | File | Rows | Positive | Negative |
|---|---|---:|---:|---:|
| Cleveland | `processed.cleveland.data` | 303 | 139 | 164 |
| Hungarian | `processed.hungarian.data` | 294 | 106 | 188 |
| Switzerland | `processed.switzerland.data` | 123 | 115 | 8 |
| Long Beach VA | `processed.va.data` | 200 | 149 | 51 |
| **Total** | | **920** | **509** | **411** |

Add a `cohort` provenance column for splitting, auditing, and cohort-specific reporting, but do not use it as a predictor in the primary models. Preserve the binary target mapping `num = 0 → 0`, `num = 1–4 → 1`.

Do not concatenate raw, processed, and `reprocessed.hungarian.data` versions as if they were different patients. They are alternate representations and would duplicate cohorts. Record exactly which four files and hashes were used.

The extra cohorts have extensive missingness, especially `ca`, `thal`, and `slope`. Do not use complete-case deletion. Audit missingness by cohort and fit imputation, scaling, encoding, discretisation, and optional missing indicators on training data only. For the BN, define and justify whether missing values are an explicit discrete state or are imputed within each training fold.

### Models

Retain the six Phase 1 references:

1. Bayesian Network
2. Naive Bayes
3. Logistic Regression
4. SVM
5. Random Forest
6. Gradient Boosting

Add the required Phase 2 deep-learning model:

7. A small regularised feed-forward MLP using scaled numeric and one-hot categorical inputs, dropout, weight decay, Adam, binary cross-entropy, and early stopping.

Recommended starting architecture: transformed input → 16 → 8 → 1. Infer the input width in each fit rather than hard-coding 20 because category availability can differ across cohorts/folds.

Add an integrated ML–DL model to satisfy the integration feedback:

8. A cross-fitted BN + MLP probability stack with a regularised logistic combiner. It must learn from training-side out-of-fold base predictions, never in-sample or outer-test predictions. The integration requirement is reported evaluator feedback; this particular safe stacking design is the team's proposed implementation.

Do not add CNNs or LSTMs to tabular patient rows merely to appear more advanced.

## 3. Hyperparameter selection without handwritten loops

The current `src/evaluation/cv.py` manually expands the grid and runs `for params in combos`. Replace that Phase 2 path with a library search estimator.

Recommended implementation:

- **OptunaSearchCV with a seeded TPE study** for MLP and, if built, the hybrid.
- `scoring="neg_log_loss"`, a declared inner `StratifiedKFold`, fixed trial budget, and `refit=True`.
- Save the library's `best_params_`, `best_score_`, `cv_results_`, trial history, seed, search space, and runtime.
- Keep the old Phase 1 grid path only for reproducibility; do not use it as Phase 2's advanced selector.
- If OptunaSearchCV is incompatible with the agreed environment, obtain professor approval and use `BayesSearchCV` or `RandomizedSearchCV`. Do not silently return to a home-grown parameter loop.

The library necessarily evaluates candidates internally. “No loops” should be implemented as **no team-written loop over parameter combinations**. Iterating over outer evaluation folds or models is not hyperparameter selection; use `cross_validate`/splitter APIs where practical and document this distinction.

Initial MLP search space:

- hidden layout: `16`, `16-8`, or `32-16`
- dropout: `0.1–0.5`
- learning rate: `1e-4–3e-3`, log scale
- weight decay: `1e-5–1e-2`, log scale
- batch size: `16` or `32`

Freeze the search space, folds, metric, and trial budget before the final run. Start with a small pilot to verify compatibility and runtime. Pruning is optional, not a professor-confirmed requirement, and should only be added if the estimator/library supports it cleanly.

## 4. Evaluation design

### Primary comparison

Use nested, patient-independent stratified cross-validation on the pooled 920-row data:

```text
Outer stratified CV: untouched performance estimates
  └── library search estimator with inner stratified CV: hyperparameter selection
        └── MLP training-only validation split: early stopping
```

All preprocessing stays inside each estimator pipeline. Outer-test patients must not affect preprocessing, hyperparameters, early stopping, thresholds, or optional stacking.

### Cross-cohort robustness

Also run a four-fold `LeaveOneGroupOut` analysis using `cohort` as the group: train on three hospitals and evaluate on the fourth. Hyperparameter selection remains inside the three training cohorts. Report each held-out cohort separately because Switzerland has only eight negative cases.

Call this **cross-cohort validation**, not proof of clinical validity. The four cohorts are historical datasets from the same UCI collection, not a prospective clinical study.

### Required outputs

- Overall and per-cohort ROC AUC, PR AUC, sensitivity, specificity, accuracy, Brier score, and log loss
- Calibration/reliability plots and cohort-specific class prevalence
- MLP training/validation curves
- Patient-level confidence intervals and paired model comparisons
- Runtime and hyperparameter-search evidence
- Ablations: no dropout, no weight decay, no early stopping, and one versus two hidden layers
- Integration ablation: BN alone versus MLP alone versus simple average versus learned stack

Store Phase 2 runs under `results/phase2/<run_id>/` with a config snapshot, four file hashes, cohort counts, code revision, split identity, seeds, package versions, selected typed parameters, predictions, and metrics. Preserve Phase 1 outputs in `results/metrics/` and `results/figures/`.

## 5. IEEE report requirement

Adapt the existing LaTeX material in `heart-disease-team-files/reports/` into `heart-disease-risk-modelling/reports/phase2/`; do not copy its old three-model numbers or Cleveland-only claims.

The report must use `IEEEtran` and contain:

1. Title, authors, affiliations, and abstract
2. Introduction and explicit research questions
3. Related work / DL literature review with citations
4. All-four-cohort dataset audit and missing-data analysis
5. Preprocessing and leakage controls
6. ML baselines and justified MLP architecture
7. Library-based hyperparameter-selection method
8. Nested and cross-cohort validation protocols
9. Main results, calibration, uncertainty, runtime, and ablations
10. Integrated BN–MLP method, comparison, and ablation results
11. Limitations, ethics, not-for-clinical-use statement, and conclusion
12. IEEE-style references

Generate tables and figures from one identified result run. Add an audit that fails when report values disagree with saved artifacts. Build the final PDF and check formatting, citations, figure readability, author details, and any page limit communicated by the professor.

## 6. Detailed team ownership

These are proposed future assignments, not claims about work already completed.

### 0. Vansh / repo owner — Phase 2 foundation, Bayesian evaluation, and hybrid

Branch: `vansh/phase2-foundation`, then `vansh/phase2-bn`

- Agree and test one Python version across all laptops.
- Add compatible PyTorch and Optuna Integration dependencies and regenerate the exact lock.
- Add `config.phase2.yaml` with the four files, cohort labels, search spaces, budgets, split strategy, and isolated result paths.
- Preserve Phase 1's configuration and six-model behavior.
- Update BN missing-data handling for the multi-cohort data and justify it.
- Implement the cross-fitted BN–MLP estimator and regularised logistic combiner after Hardik's MLP interface is merged.
- Test that every meta-model training probability comes from a base model that did not fit that patient.
- Implement/review `LeaveOneGroupOut` cohort splits and probability/calibration checks.
- Maintain Methods and limitations documentation.
- Test that dependencies, split identities, and BN probabilities are reproducible.

Viva: BN DAG/CPTs/inference, cross-fitting and stacking, missing-state decision, cohort shift, calibration, and why held-out hospitals are harder than random patient folds.

### 1. Ayush — all-cohort loading, EDA, and leakage-safe preprocessing

Branch: `ayush/phase2-all-data`

- Extend the loader to safely extract and load the four processed files only.
- Add `cohort`, retain stable patient IDs/provenance, validate 920 total rows and the target counts above.
- Verify schemas, value domains, hashes, duplicates within/across files, and missingness by cohort.
- Produce combined and cohort-specific EDA, prevalence, feature distributions, and missingness figures.
- Update numeric/one-hot and discrete preprocessing for the larger missingness pattern.
- Ensure preprocessing learns only from training folds, including the MLP early-stopping boundary.
- Add tests showing that held-out extreme values cannot change imputation/scaling/bin edges.
- Write the IEEE dataset, EDA, preprocessing, and data-limitations sections.

Viva: why processed files are used, why alternate archive files are not extra patients, cohort imbalance, missing-data handling, and leakage prevention.

### 2. Akshith — advanced search API and ablation evidence

Branch: `akshith/phase2-search`, then `akshith/phase2-experiments`

- Remove the Phase 2 dependency on `expand_grid` and the handwritten `for params in combos` search.
- Configure OptunaSearchCV/TPE through one shared search builder; do not write a custom candidate loop.
- Define validated search spaces with Hardik/Vansh and use inner CV with negative log loss.
- Save and reload typed best parameters, `cv_results_`, trial histories, budgets, seeds, and runtime.
- Add tests for search-space validity, deterministic split use, no outer-label access, and artifact replay.
- Build the MLP ablation runner using fixed non-ablated settings.
- Audit pooled and cohort-held-out predictions, file/split hashes, uncertainty, and report numbers.
- Write the IEEE hyperparameter-selection, ablation, results, and limitations sections.

Viva: why handwritten search loops were removed, TPE versus grid/random search, inner versus outer CV, log-loss selection, and why best inner score is not final performance.

### 3. Hardik — MLP and primary evaluation driver

Branch: `hardik/phase2-mlp`, then `hardik/phase2-evaluation`

- Adapt the existing PyTorch MLP without overwriting the newer six-model project.
- Split raw training rows before fitting preprocessing for early stopping.
- Support the declared layouts, dropout, weight decay, learning rate, batch size, and stopping switch.
- Record loss histories, epochs, parameter count, seeds, and train/stopping row counts.
- Expose cloneable sklearn-compatible `fit`/`predict_proba` behavior so the search API can own CV.
- Add MLP fitting, probability, cloning, reproducibility, early-stopping, and learning-curve tests.
- Make the active registry/config support Phase 1 and Phase 2 without changing old saved-result expectations.
- Update the evaluation driver for 920 rows, dynamic model counts, pooled and cohort-held-out runs, and isolated output paths.
- Run the agreed pilot and final experiments with Akshith; commit only verified compact artifacts.

Viva: MLP forward/backpropagation, BCE, Adam, regularisation, early stopping, architecture logic, and library-managed hyperparameter selection.

### 4. Kabir — app, final demo fitting, and IEEE asset audit

Branch: `kabir/phase2-webapp`, then `kabir/phase2-report-assets`

- Make the app load a selected Phase 2 run rather than assume Cleveland-only Phase 1 files.
- Show all-cohort and cohort-specific results, MLP architecture, search evidence, ablations, and calibration.
- Handle missing/incomplete result artifacts clearly.
- Add an offline, library-based final hyperparameter selection on all 920 rows and save typed final settings.
- Fit/cache demo models once; never launch a new search for each app prediction.
- Label full-data demo fits separately from nested-CV performance estimates.
- Preserve input validation and the not-for-clinical-use disclaimer.
- Adapt report-asset generation and build an audit linking IEEE tables/figures to one run.
- Prepare and test the live demo.

Viva: saved artifacts, final model versus evaluation models, caching, input validation, cohort limitations, and why the app is not a medical device.

### All five — report, presentation, and viva

- Ayush coordinates the paper; Kabir owns generated assets/audit; every member writes and reviews their technical section.
- Use actual authors, affiliations, contributions, and citations.
- Ensure README, app, IEEE paper, slides, and speaker notes use the same verified run.
- Each member uses their own Git identity and focused PRs and can explain the whole pipeline.
- Never invent an experiment, metric, contribution, or claim that the MLP/hybrid is superior.

## 7. Merge order

```text
1. Vansh: environment, Phase 2 config, and interfaces
2. Ayush: four-cohort loader, audit, preprocessing, and tests
3. Akshith: library search estimator replacing manual parameter loops
4. Hardik: sklearn-compatible MLP and learning curves
5. Vansh/Hardik: BN updates and primary evaluation wiring
6. Akshith/Hardik: pilot, frozen protocol, main runs, and ablations
7. Vansh with Hardik: cross-fitted BN–MLP hybrid and integration tests
8. Akshith/Hardik: integration ablation and final verified results
9. Kabir: app, final demo fit, and report-asset audit
10. Everyone: IEEE paper, presentation, and individual viva
```

Each PR must pass focused tests before merge; do not intentionally leave `main` broken for a later teammate.

## 8. Completion checklist

- [ ] Exactly four independent processed cohorts loaded and audited; no duplicate archive variants
- [ ] 920 rows, cohort provenance, target counts, missingness, and hashes verified
- [ ] No handwritten loop over hyperparameter candidates in the Phase 2 path
- [ ] Search implemented through a tested library estimator with saved evidence
- [ ] Regularised MLP implemented with leakage-safe early stopping
- [ ] Pooled nested evaluation and held-out-cohort evaluation completed
- [ ] MLP ablations, calibration, uncertainty, and learning curves reported
- [ ] Cross-fitted BN–MLP integration and constituent/simple-average ablations completed
- [ ] IEEEtran report builds and all values trace to one saved run
- [ ] App, README, slides, and paper agree and retain clinical disclaimers
- [ ] Every member's commits and stated contribution match their actual work
