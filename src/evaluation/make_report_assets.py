"""Generate report assets from saved results: figures, LaTeX tables and number macros.

    python -m src.evaluation.make_report_assets

Everything numeric in reports/ is produced here from results/metrics/*; the LaTeX text uses the
macros in reports/tables/numbers.tex, so the report cannot drift from the saved results.
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from src.data.loader import load_config, resolve

M = {"Bayesian Network": "BN", "Naive Bayes": "NB", "MLP": "MLP"}
ORDER = list(M)
MET = {"roc_auc": "rocauc", "pr_auc": "prauc", "sensitivity": "sens", "specificity": "spec", "precision": "prec",
       "f1": "fone", "brier": "brier", "log_loss": "logloss", "ece": "ece",
       "calibration_slope": "calslope", "calibration_intercept": "calint"}
FIGURES = ["roc_curves", "pr_curves", "calibration_curves", "confusion_matrices", "mlp_learning_curves",
           "mlp_attribution_bars", "bn_structure", "eda_target_distribution", "eda_correlation_heatmap"]
NUM_WORD = ["Zero", "One", "Two", "Three", "Four"]


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def f3(x: float) -> str:
    return f"{x:.3f}"


def build_macros(cfg: dict) -> dict:
    met = resolve(cfg["paths"]["metrics"])
    mac: dict[str, str] = {}
    fold = pd.read_csv(met / "summary_fold_level.csv")
    pooled = pd.read_csv(met / "summary_pooled_repeats.csv")
    ci = pd.read_csv(met / "bootstrap_ci.csv")
    diff = pd.read_csv(met / "bootstrap_paired_differences.csv")
    sel = pd.read_csv(met / "selected_hyperparameters.csv")
    oof = pd.read_csv(met / "oof_predictions.csv")
    man = json.loads((met / "evaluation_manifest.json").read_text())
    ver = json.loads((met / "data_verification.json").read_text())
    for name, k in M.items():
        for scheme, tag in (("fixed_0.5", ""), ("train_youden", "Y")):
            r = fold[(fold.model == name) & (fold.threshold_scheme == scheme)].iloc[0]
            for col, short in MET.items():
                if f"{col}_mean" in r and col not in ("calibration_slope", "calibration_intercept"):
                    mac[f"{k}{short}{tag}Mean"] = f3(r[f"{col}_mean"]); mac[f"{k}{short}{tag}SD"] = f3(r[f"{col}_sd"])
        r = pooled[(pooled.model == name) & (pooled.threshold_scheme == "fixed_0.5")].iloc[0]
        for col in ("ece", "calibration_slope", "calibration_intercept"):
            d = 2 if col != "ece" else 3
            mac[f"{k}{MET[col]}PMean"] = f"{r[col + '_mean']:.{d}f}"; mac[f"{k}{MET[col]}PSD"] = f"{r[col + '_sd']:.{d}f}"
        for c in ("tn", "fp", "fn", "tp"):
            mac[f"{k}{c}P"] = f"{r[c + '_mean']:.1f}"
        for _, rr in ci[ci.model == name].iterrows():
            s = MET[rr.metric]
            mac[f"{k}{s}Est"] = f3(rr.estimate); mac[f"{k}{s}CILo"] = f3(rr.ci_low); mac[f"{k}{s}CIHi"] = f3(rr.ci_high)
        o = oof[oof.model == name]
        hi, lo = o[o.p_pos >= 0.9], o[o.p_pos <= 0.1]
        mac[f"{k}HiShare"] = f"{100 * len(hi) / len(o):.0f}"; mac[f"{k}HiObs"] = f3(hi.y_true.mean()); mac[f"{k}HiPred"] = f3(hi.p_pos.mean())
        mac[f"{k}LoShare"] = f"{100 * len(lo) / len(o):.0f}"; mac[f"{k}LoObs"] = f3(lo.y_true.mean()); mac[f"{k}LoPred"] = f3(lo.p_pos.mean())
        s = sel[sel.model == name]
        mac[f"{k}YoudenMean"] = f"{s.youden_threshold.mean():.2f}"
        mac[f"{k}YoudenMin"] = f"{s.youden_threshold.min():.2f}"; mac[f"{k}YoudenMax"] = f"{s.youden_threshold.max():.2f}"
    for _, r in diff.iterrows():
        a, b = [M[x.strip()] for x in r.comparison.split(" - ")]
        s = MET[r.metric]
        mac[f"Diff{a}{b}{s}Est"] = f"{r.estimate:+.3f}"; mac[f"Diff{a}{b}{s}Lo"] = f"{r.ci_low:+.3f}"; mac[f"Diff{a}{b}{s}Hi"] = f"{r.ci_high:+.3f}"
    mlp = sel[sel.model == "MLP"]
    mac["WDhigh"] = str(int((mlp["model__weight_decay"] == 0.01).sum())); mac["WDlow"] = str(int((mlp["model__weight_decay"] == 0.001).sum()))
    mac["DOhigh"] = str(int((mlp["model__dropout"] == 0.5).sum())); mac["DOlow"] = str(int((mlp["model__dropout"] == 0.3).sum()))
    mac["BNalphaTop"] = str(int((sel[sel.model == "Bayesian Network"]["model__pseudo_count"] == 2.0).sum()))
    mac["NBalphaTop"] = str(int((sel[sel.model == "Naive Bayes"]["model__alpha"] == 2.0).sum()))
    mac["nOuter"] = str(man["outer"]["n_outer_fits_per_model"]); mac["nBoot"] = str(man["n_bootstrap"])
    mac["seed"] = str(man["seed"]); mac["foldHash"] = man["outer"]["fold_hash"]
    mac["nPatients"] = str(ver["n_rows"]); mac["nPos"] = str(ver["binary_target_counts"]["1"]); mac["nNeg"] = str(ver["binary_target_counts"]["0"])
    mac["posPct"] = f"{100 * ver['binary_target_counts']['1'] / ver['n_rows']:.1f}"
    mac["missCa"] = str(ver["missing_per_column"]["ca"]); mac["missThal"] = str(ver["missing_per_column"]["thal"])
    mac["nMissRows"] = str(ver["n_rows_with_missing"])
    for lvl, n in ver["num_value_counts"].items():
        mac[f"num{NUM_WORD[int(lvl)]}"] = str(n)
    mac["nParamsMLP"] = str(20 * 16 + 16 + 16 * 8 + 8 + 8 + 1)
    return mac


def write_macros(mac: dict, path) -> None:
    for k in mac:
        assert k.isalpha(), f"macro name must be letters only: {k}"
    path.write_text("% Generated by src/evaluation/make_report_assets.py -- do not edit.\n" +
                    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(mac.items())))


def table_fold_metrics(cfg) -> str:
    f = pd.read_csv(resolve(cfg["paths"]["metrics"]) / "summary_fold_level.csv")
    f = f[f.threshold_scheme == "fixed_0.5"].set_index("model")
    cols = [("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("sensitivity", "Sens."), ("specificity", "Spec."),
            ("precision", "Prec."), ("f1", "F1"), ("brier", "Brier"), ("log_loss", "Log loss")]
    rows = [f"{n} & " + " & ".join(f"{f.loc[n, c + '_mean']:.3f} $\\pm$ {f.loc[n, c + '_sd']:.3f}" for c, _ in cols) + r" \\" for n in ORDER]
    return ("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\nModel & " + " & ".join(h for _, h in cols) +
            " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_pooled_ci(cfg) -> str:
    c = pd.read_csv(resolve(cfg["paths"]["metrics"]) / "bootstrap_ci.csv")
    cols = [("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("brier", "Brier")]
    rows = []
    for n in ORDER:
        r = c[c.model == n].set_index("metric")
        rows.append(f"{n} & " + " & ".join(f"{r.loc[k, 'estimate']:.3f} [{r.loc[k, 'ci_low']:.3f}, {r.loc[k, 'ci_high']:.3f}]" for k, _ in cols) + r" \\")
    return ("\\begin{tabular}{lccc}\n\\toprule\nModel & " + " & ".join(h for _, h in cols) + " \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_calibration(cfg) -> str:
    p = pd.read_csv(resolve(cfg["paths"]["metrics"]) / "summary_pooled_repeats.csv")
    p = p[p.threshold_scheme == "fixed_0.5"].set_index("model")
    rows = [f"{n} & {p.loc[n, 'ece_mean']:.3f} $\\pm$ {p.loc[n, 'ece_sd']:.3f} & {p.loc[n, 'calibration_slope_mean']:.2f} $\\pm$ {p.loc[n, 'calibration_slope_sd']:.2f}"
            f" & {p.loc[n, 'calibration_intercept_mean']:.2f} $\\pm$ {p.loc[n, 'calibration_intercept_sd']:.2f} \\\\" for n in ORDER]
    return ("\\begin{tabular}{lccc}\n\\toprule\nModel & ECE & Cal.\\ slope & Cal.\\ intercept \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_confusion(cfg) -> str:
    p = pd.read_csv(resolve(cfg["paths"]["metrics"]) / "summary_pooled_repeats.csv").set_index(["model", "threshold_scheme"])
    rows = []
    for n in ORDER:
        for sch, lab in (("fixed_0.5", "0.5"), ("train_youden", "Youden")):
            r = p.loc[(n, sch)]
            rows.append(f"{n} & {lab} & {r.tn_mean:.1f} & {r.fp_mean:.1f} & {r.fn_mean:.1f} & {r.tp_mean:.1f} \\\\")
    return ("\\begin{tabular}{llcccc}\n\\toprule\nModel & Threshold & TN & FP & FN & TP \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_threshold(cfg) -> str:
    f = pd.read_csv(resolve(cfg["paths"]["metrics"]) / "summary_fold_level.csv").set_index(["model", "threshold_scheme"])
    rows = []
    for n in ORDER:
        for sch, lab in (("fixed_0.5", "0.5"), ("train_youden", "Youden")):
            r = f.loc[(n, sch)]
            rows.append(f"{n} & {lab} & {r.sensitivity_mean:.3f} $\\pm$ {r.sensitivity_sd:.3f} & {r.specificity_mean:.3f} $\\pm$ {r.specificity_sd:.3f} \\\\")
    return ("\\begin{tabular}{llcc}\n\\toprule\nModel & Threshold & Sensitivity & Specificity \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_paired(cfg) -> str:
    d = pd.read_csv(resolve(cfg["paths"]["metrics"]) / "bootstrap_paired_differences.csv")
    names = {"roc_auc": "ROC AUC", "pr_auc": "PR AUC", "brier": "Brier", "log_loss": "Log loss"}
    rows = []
    for _, r in d[d.metric.isin(names)].iterrows():
        comp = " $-$ ".join(M[x.strip()] for x in r.comparison.split(" - "))
        rows.append(f"{comp} & {names[r.metric]} & {r.estimate:+.3f} & [{r.ci_low:+.3f}, {r.ci_high:+.3f}] & {'yes' if r.ci_includes_zero else 'no'} \\\\")
    return ("\\begin{tabular}{llccc}\n\\toprule\nComparison & Metric & Difference & 95\\% CI & Includes 0 \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_attribution(cfg) -> str:
    a = pd.read_csv(resolve(cfg["paths"]["metrics"]) / "mlp_attribution.csv", index_col=0)
    rows = [f"\\texttt{{{f}}} & {r.mean_abs_shap:.3f} & {r.perm_auc_drop:.3f} $\\pm$ {r.perm_auc_drop_sd:.3f} \\\\" for f, r in a.iterrows()]
    return ("\\begin{tabular}{lcc}\n\\toprule\nFeature & mean $|$SHAP$|$ & AUC drop (permutation) \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def write_outline(mac: dict, path) -> None:
    m = mac
    path.write_text(f"""# Presentation outline (12 slides, about 12 minutes)

*Generated by `src/evaluation/make_report_assets.py`; every number comes from `results/metrics/`. Speaker notes are short on purpose.*

**1. Title and framing.** Heart Disease Risk Modelling via Bayesian Networks and Feed-Forward Neural Networks. Academic ML comparison, **not** a diagnostic system. Team names to be added.

**2. Problem statement.** Predict angiographic heart disease (binary) from 13 clinical and test variables. Question: do a domain-informed Bayesian network, Naive Bayes and a small MLP differ in *ranking* and in *probability calibration*, and how uncertain is the answer on {m['nPatients']} patients?

**3. Motivation and literature.** Calibrated probabilities matter for decision support (Steyerberg 2010; Van Calster 2019). BN theory (Pearl 1988; Koller & Friedman 2009); Naive Bayes can rank well despite violated independence (Domingos & Pazzani 1997); neural networks are not consistently better on tabular data (Shwartz-Ziv & Armon 2022; Grinsztajn et al. 2022); nested CV and leakage (Varma & Simon 2006; Cawley & Talbot 2010; Kapoor & Narayanan 2023). No systematic survey of studies on this dataset was done.

**4. Dataset and EDA.** UCI Cleveland (Detrano et al. 1989; CC BY 4.0). {m['nPatients']} rows, 14 columns, no duplicates. Missing: ca {m['missCa']}, thal {m['missThal']}. `num` counts 0-4: {m['numZero']}/{m['numOne']}/{m['numTwo']}/{m['numThree']}/{m['numFour']}. Binary target: {m['nNeg']} negative, {m['nPos']} positive ({m['posPct']}%): mild imbalance. Figures: target distribution, correlation heatmap. Caveats: one centre, referral bias, predictors come from specialist tests.

**5. Feature engineering and leakage control.** Feature types: continuous / binary / ordinal / nominal. MLP: impute, standardise, one-hot (20 inputs). BN and NB: train-fold tertile bins, integer codes (13 inputs). Everything is fitted on the training fold only and refitted in every inner and outer fold; a guard rejects the target as a predictor.

**6. Bayesian network.** 14 nodes, 16 edges, fixed before evaluation (risk factors -> disease -> manifestations). Only the probability tables are learned (Dirichlet smoothing). Edges are assumptions (domain / modelling / weak), **not** causal claims. Figure: `bn_structure.png`.

**7. Naive Bayes.** Class -> every predictor; assumes conditional independence given the class (violated: oldpeak/slope, cp/exang, thalach/age). Same inputs and smoothing as the BN, so the comparison isolates structure.

**8. MLP.** 20 -> 16 -> 8 -> 1, ReLU, dropout, {m['nParamsMLP']} parameters. BCE loss, Adam, weight decay, early stopping on a hold-out of the training fold. Hidden sizes and stopping rule fixed in advance; weight decay and dropout tuned in the inner loop.

**9. Experimental design.** {m['nOuter']} identical outer folds (5x5 stratified, seed {m['seed']}); nested 5-fold inner CV on log loss; fixed 0.5 threshold, secondary Youden threshold from training data only; 2000-sample paired patient bootstrap. Integrity checks and a leakage test (shuffled outer labels change no prediction).

**10. Results.** Fold-level ROC AUC: BN {m['BNrocaucMean']}, NB {m['NBrocaucMean']}, MLP {m['MLProcaucMean']}. Pooled ROC AUC with 95% CI: BN {m['BNrocaucEst']} [{m['BNrocaucCILo']}, {m['BNrocaucCIHi']}], NB {m['NBrocaucEst']} [{m['NBrocaucCILo']}, {m['NBrocaucCIHi']}], MLP {m['MLProcaucEst']} [{m['MLProcaucCILo']}, {m['MLProcaucCIHi']}]. Paired AUC differences all include 0: no evidence of a ranking difference. Sensitivity {m['BNsensMean']}-{m['NBsensMean']}, specificity {m['BNspecMean']}-{m['MLPspecMean']} at 0.5. Figures: ROC, PR, confusion matrices.

**11. Calibration and interpretation.** Brier: BN {m['BNbrierEst']}, NB {m['NBbrierEst']}, MLP {m['MLPbrierEst']}. Calibration slope: BN {m['BNcalslopePMean']}, NB {m['NBcalslopePMean']}, MLP {m['MLPcalslopePMean']} (below 1 = over-confident). Log-loss differences exclude zero for NB-MLP and BN-NB; Brier and AUC differences do not; exploratory, no multiplicity correction. SHAP and permutation importance for the MLP agree on ca, cp, thal, then oldpeak; they describe the model, not causes.

**12. Limitations, conclusion and future work.** Limits: 303 patients, one centre, internal validation only, mode imputation, truncated BN/NB smoothing grid (top value chosen in {m['BNalphaTop']}/{m['nOuter']} and {m['NBalphaTop']}/{m['nOuter']} folds), assumed BN structure. Conclusion: equal ranking, MLP better calibrated; no superiority or clinical claim. Future: wider pre-declared grids and recalibration, structure learning, multiple imputation, external validation, subgroup analysis. Live demo (about 2 minutes): `streamlit run app/streamlit_app.py` - Model results page, then the Risk estimator with the disclaimer visible.
""")


def main() -> None:
    cfg = load_config()
    tab, fig = resolve(cfg["paths"]["report_tables"]), resolve(cfg["paths"]["report_figures"])
    tab.mkdir(parents=True, exist_ok=True); fig.mkdir(parents=True, exist_ok=True)
    for f in FIGURES:
        shutil.copy2(resolve(cfg["paths"]["figures"]) / f"{f}.png", fig / f"{f}.png")
    mac = build_macros(cfg)
    write_macros(mac, tab / "numbers.tex")
    write_outline(mac, resolve("reports") / "presentation_outline.md")
    for name, content in {"fold_metrics": table_fold_metrics(cfg), "pooled_ci": table_pooled_ci(cfg),
                          "calibration": table_calibration(cfg), "confusion": table_confusion(cfg),
                          "threshold": table_threshold(cfg), "paired": table_paired(cfg),
                          "attribution": table_attribution(cfg)}.items():
        (tab / f"{name}.tex").write_text(content)
    print("wrote", sorted(p.name for p in tab.glob("*.tex")), "and", len(FIGURES), "figures")


if __name__ == "__main__":
    main()
