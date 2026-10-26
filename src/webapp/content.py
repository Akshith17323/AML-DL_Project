"""Fixed texts shown by the web interface."""

DISCLAIMER = (
    "**Academic research demonstration. Not a medical device and not clinically validated.** "
    "The models were built on 303 historical patients from one centre and evaluated only by internal "
    "cross-validation. Do not use anything shown here for diagnosis, treatment or any health decision."
)

ATTRIBUTION = (
    "Data: UCI Machine Learning Repository, *Heart Disease* (Cleveland subset), licensed CC BY 4.0. "
    "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). doi:10.24432/C52P4X. "
    "Principal investigator for the Cleveland data: Robert Detrano, M.D., Ph.D."
)

POPULATION_NOTE = (
    "Probabilities describe how a model fitted to *patients referred for coronary angiography in a historical "
    "Cleveland cohort* (45.9% positive) scores the values entered. They are not a risk for a member of the general "
    "population, and they use results of exercise testing, fluoroscopy and perfusion imaging that are not available "
    "at first contact."
)

CALIBRATION_NOTE = (
    "In the study the Bayesian network and Naive Bayes were over-confident (calibration slopes below 1), so their "
    "extreme probabilities should not be taken at face value; the MLP was better calibrated. All three share the "
    "same data limitations."
)

MLP_EXPLANATION_NOTE = (
    "SHAP values describe how *this fitted MLP* moves its output for the entered values relative to 30 reference "
    "patients. They are an approximation (sampling-based Kernel SHAP), are shared among correlated predictors, and "
    "say nothing about causes or about what would happen if a value were changed."
)

DATA_MISSING_HELP = (
    "The dataset is not distributed with this repository. Place the original archive at "
    "`data/raw/heart+disease.zip` and run `python -m src.data.verify`, then restart the app "
    "(see `data/raw/README.md`)."
)
