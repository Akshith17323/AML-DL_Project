"""Generate the comprehensive Phase 2 PowerPoint presentation for the team."""
from __future__ import annotations

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUTPUT_PATH = Path("reports/presentation/Heart_Disease_Risk_Modelling_Phase2.pptx")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color Palette: Clean Executive Navy & Crimson
COLOR_NAVY = RGBColor(16, 42, 77)       # #102A4D
COLOR_CRIMSON = RGBColor(178, 34, 34)   # #B22222
COLOR_CHARCOAL = RGBColor(34, 34, 34)   # #222222
COLOR_MUTED = RGBColor(90, 90, 90)      # #5A5A5A
COLOR_BG_CARD = RGBColor(245, 247, 250) # Light blue-gray

blank_slide_layout = prs.slide_layouts[6]

def add_header(slide, title_text: str, category_text: str, speaker_name: str | None = None):
    # Category tag / Speaker pill
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = category_text.upper()
    if speaker_name:
        p_tag.text += f"  |  SPEAKER: {speaker_name.upper()}"
    p_tag.font.size = Pt(11)
    p_tag.font.bold = True
    p_tag.font.color.rgb = COLOR_CRIMSON
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.8))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(24)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_NAVY

def add_bullet_card(slide, left: float, top: float, width: float, height: float, title: str, points: list[str]):
    shape = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height)) # Rectangle
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLOR_BG_CARD
    shape.line.color.rgb = RGBColor(218, 224, 233)
    shape.line.width = Pt(1)

    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_right = Inches(0.25)
    tf.margin_top = Inches(0.2)
    tf.margin_bottom = Inches(0.2)

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(15)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_NAVY
    p0.space_after = Pt(10)

    for pt in points:
        p = tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_CHARCOAL
        p.space_after = Pt(6)

# -------------------------------------------------------------
# SLIDE 1: Title Slide
# -------------------------------------------------------------
slide1 = prs.slides.add_slide(blank_slide_layout)
t_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.3), Inches(2.2))
tf = t_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Multi-Cohort Heart Disease Risk Modelling"
p.font.size = Pt(32)
p.font.bold = True
p.font.color.rgb = COLOR_NAVY

p2 = tf.add_paragraph()
p2.text = "Bayesian–Neural Integration, Automated Hyperparameter Search & Domain Shift Evaluation"
p2.font.size = Pt(18)
p2.font.bold = True
p2.font.color.rgb = COLOR_CRIMSON
p2.space_before = Pt(8)

p3 = tf.add_paragraph()
p3.text = "Applied Machine Learning & Deep Learning (AML-DL) — Final Phase 2 Project Defense"
p3.font.size = Pt(13)
p3.font.color.rgb = COLOR_MUTED
p3.space_before = Pt(6)

# Team Table Card
add_bullet_card(slide1, 1.0, 3.8, 11.333, 3.0, "Project Team & Module Ownership", [
    "Hardik Hathwal (2401010176) — Tabular MLP Deep Learning, Evaluation Driver, SHAP Interaction Analysis",
    "Ayush (2401010121) — Four-Cohort Data Consolidation, Missingness Audit & Leakage-Safe Preprocessing",
    "Akshith (2401010230) — Library-Managed Optuna/TPE Hyperparameter Selection & Ablation Studies",
    "Vansh (2401010494) — Project Foundation, Bayesian Network Architecture & Cross-Fitted BN–MLP Hybrid",
    "Kabir (2401010205) — Multi-Cohort Streamlit Webapp, Full-Data Demo Fitting & IEEE Asset Audit",
    "Faculty Supervisor: Mr. Shobhit Johri | School of Technology, Rishihood University"
])

# -------------------------------------------------------------
# SLIDE 2: Response to Professor Feedback (All)
# -------------------------------------------------------------
slide2 = prs.slides.add_slide(blank_slide_layout)
add_header(slide2, "Responding to Phase 1 Feedback & Evaluation Criteria", "Project Scope & Governance", "Hardik Hathwal")
add_bullet_card(slide2, 0.8, 1.8, 5.7, 5.0, "Professor Phase 1 Directives", [
    "Requirement 1: IEEE Standard Format — Migrated project report to formal two-column IEEE conference paper with verified figures.",
    "Requirement 2: Eliminate Manual Loops — Removed all handwritten parameter search loops ('for params in combos') from evaluation code.",
    "Requirement 3: All Available Cohorts — Expanded beyond 303 Cleveland rows to all four independent UCI cohorts (920 total patients).",
    "Requirement 4: ML-DL Integration — Designed a cross-fitted Bayesian Network + MLP probability stack with logistic combiner."
])
add_bullet_card(slide2, 6.8, 1.8, 5.7, 5.0, "Phase 2 Technical Rigour & Controls", [
    "Leakage-Safe Nested Cross-Validation: Outer 5-fold CV for performance estimation; inner 3-fold CV for model selection.",
    "No-Loop Optuna/TPE Search: Automated hyperparameter search with fixed trial budgets, deterministic seeds, and log-loss objective.",
    "Hospital Domain Shift: Four-fold Leave-One-Cohort-Out analysis evaluating real-world generalization across hospital sites.",
    "Strict Non-Clinical Status: Academic benchmarking prototype; explicit disclaimers against diagnostic deployment."
])

# -------------------------------------------------------------
# SLIDE 3: Four-Cohort Data Engineering (Ayush)
# -------------------------------------------------------------
slide3 = prs.slides.add_slide(blank_slide_layout)
add_header(slide3, "Multi-Centre Dataset Provenance & Target Mapping", "Data Engineering & Audit", "Ayush")
add_bullet_card(slide3, 0.8, 1.8, 5.7, 5.0, "Four Independent Patient Cohorts", [
    "Cleveland Clinic Foundation (USA): 303 patients (139 positive / 164 negative; 45.9% prevalence).",
    "Hungarian Institute of Cardiology (Budapest): 294 patients (106 positive / 188 negative; 36.1% prevalence).",
    "University Hospital Zurich (Switzerland): 123 patients (115 positive / 8 negative; 93.5% prevalence).",
    "Veterans Affairs Medical Center (Long Beach, CA): 200 patients (149 positive / 51 negative; 74.5% prevalence).",
    "Consolidated Dataset: Exactly 920 patients (509 positive / 411 negative; 55.3% global prevalence).",
    "Audit Integrity: Checked SHA-256 hashes; rejected alternate raw/reprocessed files to prevent duplicate patient leakage."
])
add_bullet_card(slide3, 6.8, 1.8, 5.7, 5.0, "Clinical Target & Feature Domain", [
    "Target Definition: num=0 indicates angiographic negative (<50% diameter stenosis); num=1..4 indicates positive (>=50% stenosis).",
    "13 Standard Predictors: Age, Sex, Chest Pain (cp), Resting Blood Pressure (trestbps), Serum Cholesterol (chol), Fasting Blood Sugar (fbs).",
    "Resting ECG (restecg), Max Heart Rate (thalach), Exercise Induced Angina (exang), ST Depression (oldpeak), Slope, Vessels (ca), Thal.",
    "Cohort Column Provenance: cohort retained strictly for audit and Leave-One-Group-Out validation, NEVER as a predictive feature."
])

# -------------------------------------------------------------
# SLIDE 4: Missingness & Leakage-Safe Preprocessing (Ayush)
# -------------------------------------------------------------
slide4 = prs.slides.add_slide(blank_slide_layout)
add_header(slide4, "Severe Real-World Missingness & Leakage Controls", "Preprocessing & Quality", "Ayush")
add_bullet_card(slide4, 0.8, 1.8, 5.7, 5.0, "Missing Data Distribution by Site", [
    "Cleveland: Near-complete cases (only 6 missing values across 303 rows; 4 in ca, 2 in thal).",
    "Hungarian: 782 missing cells; ca missing in 98.6%, thal missing in 56.5%, slope missing in 64.6%.",
    "Switzerland: 273 missing cells; ca missing in 95.9%, thal missing in 89.4%, slope missing in 86.9%.",
    "Long Beach VA: 698 missing cells; ca missing in 99.0%, thal missing in 82.5%, slope missing in 43.5%.",
    "Why Not Complete-Case Deletion? Dropping incomplete rows would discard >67% of data, biasing toward patients given invasive angiograms."
])
add_bullet_card(slide4, 6.8, 1.8, 5.7, 5.0, "Leakage-Safe Preprocessing Pipelines", [
    "Training-Only Fitting: Imputation medians, standard scaling means/stds, and one-hot encoders fitted strictly on training folds.",
    "Median Continuous Imputation: Simple, robust replacement preventing data leakage across outer evaluation splits.",
    "Explicit Missing Indicators: Added for sparse predictors to let neural and linear models learn structural absence.",
    "Zero-Leakage Early Stopping: MLP raw training folds are split BEFORE preprocessing, ensuring early-stopping rows remain pristine."
])

# -------------------------------------------------------------
# SLIDE 5: Bayesian Network Architecture (Vansh)
# -------------------------------------------------------------
slide5 = prs.slides.add_slide(blank_slide_layout)
add_header(slide5, "Domain-Informed Probabilistic Graphical Model", "Bayesian Modelling", "Vansh")
add_bullet_card(slide5, 0.8, 1.8, 5.7, 5.0, "Probabilistic Reasoning & DAG Structure", [
    "Pearl's Causal Principles: Encodes domain cardiological pathways rather than learning naive correlations from noisy data.",
    "Graph Topology: Disease node influences symptomatic and physiological nodes: Angina, ECG, Slope, ST Depression, Vessels, Heart Rate.",
    "Tertile Discretisation: Continuous features (age, trestbps, chol, thalach, oldpeak) partitioned into Low, Medium, High clinical bins.",
    "Parameter Estimation: Bayesian Dirichlet equivalent uniform priors prevent zero-probability traps in sparse conditional probability tables."
])
add_bullet_card(slide5, 6.8, 1.8, 5.7, 5.0, "Bayesian Strengths & Discretisation Trade-offs", [
    "Transparency: Exact conditional probabilities inspectable for every medical variable.",
    "Missing Evidence Resilience: Exact variable elimination allows inference when diagnostic tests are unavailable.",
    "Discretisation Bottleneck: Forcing continuous physiological variables into 3 bins discards gradient subtleties.",
    "Conditional Independence: Rigid DAG assumes features are independent given the disease state, missing joint non-linear synergies."
])

# -------------------------------------------------------------
# SLIDE 6: Cross-Fitted BN–MLP Hybrid Stacking (Vansh)
# -------------------------------------------------------------
slide6 = prs.slides.add_slide(blank_slide_layout)
add_header(slide6, "Cross-Fitted Bayesian–Neural Integration", "Hybrid Architecture", "Vansh")
add_bullet_card(slide6, 0.8, 1.8, 5.7, 5.0, "Stacking Design & Leakage Prevention", [
    "Integration Objective: Combine Pearl's transparent probabilistic reasoning with deep learning's non-linear feature representations.",
    "The In-Sample Trap: Training a meta-learner on in-sample predictions causes extreme over-confidence and calibration failure.",
    "Cross-Fitting Solution: K-fold split internal to the training set; base BN and MLP generate out-of-fold probabilistic predictions.",
    "Meta-Learner: L2-regularised Logistic Regression combiner trained on out-of-fold probability vectors [p_BN, p_MLP]."
])
add_bullet_card(slide6, 6.8, 1.8, 5.7, 5.0, "Implementation Verification", [
    "Strict Isolation: Outer test fold patients never influence base models, out-of-fold generation, or logistic combiner parameters.",
    "Unit Testing: Verified via test_hybrid_meta_features_are_out_of_fold_probabilities that meta-features derive purely from out-of-fold fits.",
    "Empirical Finding: Integration matches standalone MLP (0.889 vs 0.888 ROC AUC); does not establish unilateral superiority."
])

# -------------------------------------------------------------
# SLIDE 7: Automated Hyperparameter Selection (Akshith)
# -------------------------------------------------------------
slide7 = prs.slides.add_slide(blank_slide_layout)
add_header(slide7, "Library-Managed Search: Eliminating Manual Loops", "Automated Optimization", "Akshith")
add_bullet_card(slide7, 0.8, 1.8, 5.7, 5.0, "The Handwritten Loop Problem", [
    "Phase 1 Limitation: src/evaluation/cv.py manually expanded grids and executed 'for params in combos' loops.",
    "Risks: Prone to selection bias, difficult convergence tracking, and unprincipled grid resolution on continuous variables.",
    "Modern Replacement: Replaced entirely with OptunaSearchCV powered by Tree-structured Parzen Estimators (TPE).",
    "Zero Handwritten Loops: Library owns candidate trial sampling, evaluation, pruning, and internal refitting."
])
add_bullet_card(slide7, 6.8, 1.8, 5.7, 5.0, "Search Formulation & Objective", [
    "Objective Metric: Negative Log Loss (scoring='neg_log_loss') penalises miscalibrated confidence over raw thresholded accuracy.",
    "Inner Stratified K-Fold: 3-fold inner cross-validation nested strictly inside outer training partitions.",
    "Fixed Trial Budget: Exactly 12 trials per fold in main runs; reproducible via deterministic random seeds.",
    "Artifact Persistence: Saved best_params, cv_results, trial histories, runtimes, and parameter search spaces for audit."
])

# -------------------------------------------------------------
# SLIDE 8: Controlled Ablation Studies (Akshith)
# -------------------------------------------------------------
slide8 = prs.slides.add_slide(blank_slide_layout)
add_header(slide8, "Controlled Ablations: Quantifying Architectural Choices", "Empirical Ablations", "Akshith")
add_bullet_card(slide8, 0.8, 1.8, 5.7, 5.0, "MLP Regulariser & Depth Ablations", [
    "Full Regularised MLP: ROC AUC = 0.889 | Log Loss = 0.417",
    "Ablate Dropout (p = 0.0): ROC AUC = 0.891 | Log Loss = 0.422 (Slight loss increase under calibration)",
    "Ablate Weight Decay (lambda = 0.0): ROC AUC = 0.888 | Log Loss = 0.422 (Marginal over-fitting)",
    "Ablate Early Stopping: ROC AUC = 0.892 | Log Loss = 0.419 (Higher epoch variance)",
    "Single Hidden Layer (16 -> 1): ROC AUC = 0.892 | Log Loss = 0.412 (Matches/slightly exceeds 2-layer depth)"
])
add_bullet_card(slide8, 6.8, 1.8, 5.7, 5.0, "Integration Suite Ablations", [
    "Standalone Bayesian Network: ROC AUC = 0.876 | Log Loss = 0.474",
    "Standalone Tabular MLP: ROC AUC = 0.889 | Log Loss = 0.417",
    "Simple Probability Average: ROC AUC = 0.889 | Log Loss = 0.417",
    "Cross-Fitted Logistic Stack: ROC AUC = 0.890 | Log Loss = 0.417",
    "Conclusion: Adding a meta-learning layer does not significantly outperform simple unweighted averaging on clinical tabular data."
])

# -------------------------------------------------------------
# SLIDE 9: Tabular MLP Architecture (Hardik)
# -------------------------------------------------------------
slide9 = prs.slides.add_slide(blank_slide_layout)
add_header(slide9, "Regularised Feedforward Neural Network (PyTorch)", "Deep Learning Architecture", "Hardik Hathwal")
add_bullet_card(slide9, 0.8, 1.8, 5.7, 5.0, "Network Design & Regularisation Defense", [
    "Input Dimension: Dynamically inferred (~20 one-hot and scaled features across folds).",
    "Layer Layout: Input -> Linear(16) -> ReLU -> Dropout(p) -> Linear(8) -> ReLU -> Linear(1) -> Sigmoid.",
    "Optimization: Adam optimiser with log-uniform learning rate (1e-4 to 3e-3), weight decay (1e-5 to 1e-2).",
    "Loss Function: Binary Cross-Entropy (BCE) with continuous probability outputs.",
    "Mini-Batch Training: Shuffled batches of 16 or 32 samples across maximum 120 epochs."
])
add_bullet_card(slide9, 6.8, 1.8, 5.7, 5.0, "Leakage-Safe Early Stopping & Scikit-Learn API", [
    "Internal Validation Split: 15% of raw training rows set aside before fitting preprocessing transformers.",
    "Patience Logic: Early stopping triggers when validation loss fails to improve for 12 consecutive epochs.",
    "Loss History Tracking: Epoch-by-epoch train/stopping curves logged and stored in results/phase2/.",
    "Scikit-Learn Contract: Fully cloneable via BaseEstimator and ClassifierMixin; seamlessly integrates into OptunaSearchCV."
])

# -------------------------------------------------------------
# SLIDE 10: Nested Cross-Validation Driver (Hardik)
# -------------------------------------------------------------
slide10 = prs.slides.add_slide(blank_slide_layout)
add_header(slide10, "Nested Cross-Validation & Comprehensive Evaluation", "Primary Evaluation", "Hardik Hathwal")
add_bullet_card(slide10, 0.8, 1.8, 5.7, 5.0, "Two-Tier Nested Evaluation Protocol", [
    "Outer Stratified 5-Fold CV: Partitions 920 patients into 5 untouched test sets (184 patients each).",
    "Inner Stratified 3-Fold CV: Library search selects optimal hyperparameters strictly inside training folds.",
    "Zero Test Leakage: Outer test patients never influence imputation, scaling, hyperparameter tuning, or early stopping.",
    "Model Coverage: Simultaneously benchmarks all 8 models (BN, Naive Bayes, LR, SVM, RF, Gradient Boosting, MLP, Hybrid)."
])
add_bullet_card(slide10, 6.8, 1.8, 5.7, 5.0, "Pooled Evaluation Benchmark (920 Patients)", [
    "SVM (RBF Kernel): ROC AUC = 0.895 [0.874, 0.916] | Brier = 0.126 | Log Loss = 0.405",
    "Logistic Regression: ROC AUC = 0.889 [0.866, 0.909] | Brier = 0.131 | Log Loss = 0.417",
    "BN-MLP Hybrid Stack: ROC AUC = 0.889 [0.864, 0.909] | Brier = 0.129 | Log Loss = 0.414",
    "Tabular MLP: ROC AUC = 0.888 [0.865, 0.909] | Brier = 0.131 | Log Loss = 0.418",
    "Gradient Boosting: ROC AUC = 0.889 [0.868, 0.910] | Brier = 0.131 | Log Loss = 0.416",
    "Bayesian Network: ROC AUC = 0.875 [0.851, 0.896] | Brier = 0.142 | Log Loss = 0.449"
])

# -------------------------------------------------------------
# SLIDE 11: Cross-Cohort Domain Shift (Hardik & Vansh)
# -------------------------------------------------------------
slide11 = prs.slides.add_slide(blank_slide_layout)
add_header(slide11, "Hospital Domain Shift: Leave-One-Cohort-Out", "Generalization Limits", "Hardik & Vansh")
add_bullet_card(slide11, 0.8, 1.8, 5.7, 5.0, "Leave-One-Group-Out Protocol", [
    "Simulation: Train models on 3 hospitals; test on the 4th held-out institution.",
    "Hospital-Specific Folds: Held out Cleveland (303), Hungarian (294), Switzerland (123), and Long Beach VA (200).",
    "Extreme Clinical Imbalance: Switzerland cohort has 115 positive cases vs only 8 negative cases (93.5% prevalence).",
    "Non-Random Shift: Changes in demographic age, baseline comorbidities, referral patterns, and missing diagnostic tests."
])
add_bullet_card(slide11, 6.8, 1.8, 5.7, 5.0, "Performance Drop Under Hospital Shift", [
    "Naive Bayes: Drops from 0.879 -> 0.832 ROC AUC [0.805, 0.858]",
    "Logistic Regression: Drops from 0.889 -> 0.827 ROC AUC [0.799, 0.855]",
    "BN-MLP Hybrid: Drops from 0.889 -> 0.825 ROC AUC [0.796, 0.853]",
    "Tabular MLP: Drops from 0.888 -> 0.809 ROC AUC [0.778, 0.838] (Greatest sensitivity to missingness shift)",
    "Takeaway: Internal pooled cross-validation drastically overestimates diagnostic transferability to new clinical sites."
])

# -------------------------------------------------------------
# SLIDE 12: SHAP Interaction Analysis (Hardik)
# -------------------------------------------------------------
slide12 = prs.slides.add_slide(blank_slide_layout)
add_header(slide12, "Explainability: Non-linear MLP Surfaces vs Rigid BN", "Model Interpretability", "Hardik Hathwal")
add_bullet_card(slide12, 0.8, 1.8, 5.7, 5.0, "Continuous Non-linear Surfaces (MLP)", [
    "Chronotropic Incompetence (Age x Thalach): Depressed maximum heart rate (<120 bpm) sharply escalates risk in elderly (>60 yrs).",
    "Smooth Manifold: PyTorch MLP smoothly tracks the physiological threshold across continuous coordinates.",
    "Silent Ischemia (Trestbps x CP): Elevated systolic BP produces compounding risk when chest pain is asymptomatic (cp=4).",
    "Biological Alignment: Matches clinical literature where asymptomatic diabetic patients face elevated acute cardiac events."
])
add_bullet_card(slide12, 6.8, 1.8, 5.7, 5.0, "Bayesian Network Discretisation Blindspots", [
    "Step-Function Artifacts: Tertile discretisation creates artificial discontinuities at bin edges (e.g., 119 bpm vs 121 bpm).",
    "Conditional Independence Bottleneck: Fixed DAG treats BP and Chest Pain as independent given disease state, missing synergy.",
    "Diagnostic Value: Proves that while MLP and BN achieve comparable ROC AUC, MLP learns superior physiological interaction manifolds."
])

# -------------------------------------------------------------
# SLIDE 13: Clinical Web Application (Kabir)
# -------------------------------------------------------------
slide13 = prs.slides.add_slide(blank_slide_layout)
add_header(slide13, "Interactive Multi-Cohort Clinical Web Application", "Application Engineering", "Kabir")
add_bullet_card(slide13, 0.8, 1.8, 5.7, 5.0, "Streamlit Architecture & Multi-Cohort Views", [
    "Dynamic Run Loader: Reads audited Phase 2 results directly from results/phase2/main/ rather than hardcoded metrics.",
    "Data Explorer: Displays patient distributions, missingness matrices, and cohort prevalence across all 4 hospital sites.",
    "Model Evaluation Dashboard: Interactive ROC/PR curves, reliability/calibration curves, and paired confidence intervals.",
    "Architectural Inspector: Displays PyTorch MLP layer dimensions, Optuna search spaces, and ablation tables."
])
add_bullet_card(slide13, 6.8, 1.8, 5.7, 5.0, "Risk Estimator & Demo Caching", [
    "Full-Data Demo Models: Offline final selection on all 920 rows saved to final_demo_models.joblib.",
    "Zero-Search Inference: App never launches on-the-fly hyperparameter searches; instant latency (<20 ms).",
    "Clear Separation: Strict visual distinction between evaluation CV estimates vs full-data demo predictions.",
    "Robust Input Validation: Enforces physiological boundaries (e.g., age 20-100, BP 80-200 mmHg) with error indicators."
])

# -------------------------------------------------------------
# SLIDE 14: IEEE Asset Audit & Disclaimers (Kabir)
# -------------------------------------------------------------
slide14 = prs.slides.add_slide(blank_slide_layout)
add_header(slide14, "Automated IEEE Asset Generation & Clinical Safety", "Auditing & Ethics", "Kabir")
add_bullet_card(slide14, 0.8, 1.8, 5.7, 5.0, "Automated LaTeX Asset Pipeline", [
    "Asset Generation Engine: src/evaluation/make_phase2_report_assets.py reads JSON/CSV run artifacts.",
    "Traceable LaTeX Tables: Generates pooled_results.tex, cross_cohort_results.tex, and ablations.tex automatically.",
    "Zero Manual Copy-Pasting: Prevents transcription errors between experimental code and final publication paper.",
    "Regression Tests: Automated tests (test_phase2_outputs.py, test_phase2_saved_results.py) verify file hashes and metrics."
])
add_bullet_card(slide14, 6.8, 1.8, 5.7, 5.0, "Clinical Safety & Regulatory Disclaimers", [
    "Not a Medical Device: Prominently displayed disclaimers across README, IEEE paper, app, and presentation slides.",
    "Historical Data Limitations: 1980s retrospective patient cohort does not reflect contemporary medical practice or statin usage.",
    "Invasive Referral Bias: Cohort consists of patients already referred for invasive fluoroscopy / angiography.",
    "Ethical Governance: Explicit statements that model predictions must never replace clinical assessment by certified physicians."
])

# -------------------------------------------------------------
# SLIDE 15: Primary Results Summary (All)
# -------------------------------------------------------------
slide15 = prs.slides.add_slide(blank_slide_layout)
add_header(slide15, "Complete Experimental Findings: Pooled vs Cross-Cohort", "Empirical Synthesis", "Hardik Hathwal")
add_bullet_card(slide15, 0.8, 1.8, 5.7, 5.0, "Pooled 5-Fold Nested CV (920 Patients)", [
    "SVM (RBF): ROC AUC 0.895 [0.874, 0.916] | PR AUC 0.893 | Brier 0.126 | Log Loss 0.405",
    "BN-MLP Hybrid: ROC AUC 0.889 [0.864, 0.909] | PR AUC 0.893 | Brier 0.129 | Log Loss 0.414",
    "Gradient Boosting: ROC AUC 0.889 [0.868, 0.910] | PR AUC 0.890 | Brier 0.131 | Log Loss 0.416",
    "Logistic Regression: ROC AUC 0.889 [0.866, 0.909] | PR AUC 0.896 | Brier 0.131 | Log Loss 0.417",
    "Tabular MLP: ROC AUC 0.888 [0.865, 0.909] | PR AUC 0.889 | Brier 0.131 | Log Loss 0.418",
    "Random Forest: ROC AUC 0.886 [0.863, 0.908] | PR AUC 0.884 | Brier 0.133 | Log Loss 0.422",
    "Naive Bayes: ROC AUC 0.879 [0.854, 0.900] | PR AUC 0.881 | Brier 0.139 | Log Loss 0.444",
    "Bayesian Network: ROC AUC 0.875 [0.851, 0.896] | PR AUC 0.884 | Brier 0.142 | Log Loss 0.449"
])
add_bullet_card(slide15, 6.8, 1.8, 5.7, 5.0, "Core Scientific Takeaways", [
    "1. Indistinguishable Top Models: 95% bootstrap confidence intervals overlap across SVM, LR, Gradient Boosting, MLP, and Hybrid.",
    "2. Tabular Deep Learning Parity: A tuned MLP matches but does not beat classical linear/kernel baselines on ~900 tabular rows.",
    "3. Stacking Gains are Marginal: Learning a meta-combiner adds model complexity without yielding meaningful discrimination gains.",
    "4. Cross-Cohort Vulnerability: Performance drops from ~0.89 to ~0.81-0.83 under domain shift, revealing true deployment limits."
])

# -------------------------------------------------------------
# SLIDE 16: Viva Defense & Team Ownership (All)
# -------------------------------------------------------------
slide16 = prs.slides.add_slide(blank_slide_layout)
add_header(slide16, "Individual Ownership & Viva Voce Defense Areas", "Project Defense", "All Team Members")
add_bullet_card(slide16, 0.8, 1.8, 5.7, 5.0, "Individual Viva Defense Topics", [
    "Hardik: MLP forward/backprop, BCE loss, Adam, weight decay, early-stopping boundary, SHAP non-linear interactions.",
    "Ayush: Four-cohort loading, duplicate archive variants, class imbalance, training-only imputation, leakage prevention.",
    "Akshith: Handwritten search loop elimination, TPE vs grid/random, inner vs outer CV, log-loss optimization, ablations.",
    "Vansh: Pearl's DAG, conditional probability tables, tertile discretisation, cross-fitting mechanics, stacking calibration.",
    "Kabir: Saved artifacts, full-data demo fitting vs CV estimates, caching, input validation, clinical safety disclaimers."
])
add_bullet_card(slide16, 6.8, 1.8, 5.7, 5.0, "Summary & Project Links", [
    "IEEE Conference Paper: reports/phase2/main.pdf (compiled with Tectonic/IEEEtran)",
    "Interactive Streamlit App: streamlit run app/streamlit_app.py",
    "GitHub Repository: https://github.com/Akshith17323/AML-DL_Project",
    "Branches: hardik/phase2-mlp, hardik/phase2-evaluation, hardik/phase2-report",
    "Thank you to our supervisor Mr. Shobhit Johri and faculty evaluators. Questions & Discussion."
])

prs.save(str(OUTPUT_PATH))
print(f"Presentation successfully saved to {OUTPUT_PATH}")
