"""Generate every numeric table of the Phase 2 IEEE report from one saved run.

Nothing is typed by hand: each table reads results/phase2/main/metrics (plus the
paired-difference CSVs written by paired_comparisons.py) and writes a LaTeX
fragment. key_numbers.json collects the scalars quoted in the text so that
audit_report_numbers.py can check them against the PDF source.

Usage (from the repository root):
    python reports/phase2/ieee/scripts/make_ieee_tables.py \
        results/phase2/main/metrics reports/phase2/ieee/tables
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ORDER = ["Bayesian Network", "Naive Bayes", "Logistic Regression", "SVM",
         "Random Forest", "Gradient Boosting", "MLP", "BN-MLP Stack"]
SHORT = {"Bayesian Network": "BN", "Naive Bayes": "NB", "Logistic Regression": "LR",
         "SVM": "SVM", "Random Forest": "RF", "Gradient Boosting": "GB",
         "MLP": "MLP", "BN-MLP Stack": "BN--MLP stack"}
COHORTS = ["Cleveland", "Hungarian", "Switzerland", "VA"]


def f3(x: float) -> str:
    return f"{x:.3f}"


def bold_best(values: dict, higher=True) -> dict:
    best = max(values.values()) if higher else min(values.values())
    return {k: (r"\textbf{" + f3(v) + "}" if round(v, 3) == round(best, 3) else f3(v))
            for k, v in values.items()}


def write(out: Path, name: str, body: str) -> None:
    (out / name).write_text(body.strip() + "\n")


def main_table(summary, ci, protocol):
    s = summary[summary.cohort == "ALL"].set_index("model").loc[ORDER]
    c = ci.pivot(index="model", columns="metric", values=["ci_low", "ci_high"])
    auc = bold_best(s.roc_auc.to_dict()); pr = bold_best(s.pr_auc.to_dict())
    br = bold_best(s.brier.to_dict(), False); ll = bold_best(s.log_loss.to_dict(), False)
    ece = bold_best(s.ece.to_dict(), False)
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"Model & ROC AUC [95\% CI] & PR AUC & Brier & Log loss & ECE & Acc. & Sens. & Spec. \\",
             r"\midrule"]
    for m in ORDER:
        lo, hi = c.loc[m, ("ci_low", "roc_auc")], c.loc[m, ("ci_high", "roc_auc")]
        if m == "MLP":
            lines.append(r"\midrule")
        lines.append(f"{SHORT[m]} & {auc[m]} [{lo:.3f}, {hi:.3f}] & {pr[m]} & {br[m]} & {ll[m]} & "
                     f"{ece[m]} & {f3(s.loc[m,'accuracy'])} & {f3(s.loc[m,'sensitivity'])} & "
                     f"{f3(s.loc[m,'specificity'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def cohort_table(pooled, cross):
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r" & \multicolumn{4}{c}{Pooled nested CV (by cohort)} & \multicolumn{4}{c}{Held-out cohort (LOCO)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Model & Cle & Hun & Swi & VA & Cle & Hun & Swi & VA \\", r"\midrule"]
    p = pooled.set_index(["model", "cohort"]).roc_auc
    x = cross.set_index(["model", "cohort"]).roc_auc
    for m in ORDER:
        if m == "MLP":
            lines.append(r"\midrule")
        lines.append(SHORT[m] + " & " + " & ".join(f3(p[(m, c)]) for c in COHORTS) + " & "
                     + " & ".join(f3(x[(m, c)]) for c in COHORTS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def shift_table(pooled, cross):
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Model & Pooled AUC & LOCO AUC & Mean held-out AUC & $\Delta$ (LOCO$-$pooled) \\",
             r"\midrule"]
    rows = {}
    for m in ORDER:
        pa = pooled[(pooled.model == m) & (pooled.cohort == "ALL")].roc_auc.item()
        xa = cross[(cross.model == m) & (cross.cohort == "ALL")].roc_auc.item()
        mean_h = cross[(cross.model == m) & (cross.cohort.isin(COHORTS))].roc_auc.mean()
        rows[m] = (pa, xa, mean_h, xa - pa)
    best_mean = max(v[2] for v in rows.values())
    for m in ORDER:
        pa, xa, mh, d = rows[m]
        mh_s = r"\textbf{" + f3(mh) + "}" if round(mh, 3) == round(best_mean, 3) else f3(mh)
        if m == "MLP":
            lines.append(r"\midrule")
        lines.append(f"{SHORT[m]} & {f3(pa)} & {f3(xa)} & {mh_s} & ${d:+.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), rows


def paired_table(pp, pc):
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"Comparison ($A-B$) & Pooled $\Delta$AUC [95\% CI] & LOCO $\Delta$AUC [95\% CI] \\",
             r"\midrule"]
    for (a, b), _ in pp[pp.metric == "roc_auc"].groupby(["model_a", "model_b"], sort=False):
        cells = []
        for df in (pp, pc):
            r = df[(df.model_a == a) & (df.model_b == b) & (df.metric == "roc_auc")].iloc[0]
            txt = f"${r.difference:+.3f}$ [${r.ci_low:+.3f}$, ${r.ci_high:+.3f}$]"
            cells.append(txt + (r"$^{*}$" if r.excludes_zero else ""))
        lines.append(f"{SHORT[a]} $-$ {SHORT[b]} & {cells[0]} & {cells[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ablation_table(ab):
    names = {"full": "Full MLP (16--8, $p{=}0.3$, wd $10^{-3}$, ES)",
             "no_dropout": "$-$ dropout (p = 0)", "no_weight_decay": "$-$ weight decay (0)",
             "no_early_stopping": "$-$ early stopping (150 ep.)",
             "one_hidden_layer": "One hidden layer (16)",
             "BN": "BN alone", "MLP": "MLP alone", "simple_average": "Simple average",
             "learned_stack": "Learned stack (cross-fitted)"}
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Variant & ROC AUC (mean $\pm$ SD) & Log loss & Fit (s) \\", r"\midrule",
             r"\multicolumn{4}{l}{\textit{(a) MLP regularisation and depth}} \\"]
    for suite in ("mlp", "integration"):
        if suite == "integration":
            lines += [r"\midrule", r"\multicolumn{4}{l}{\textit{(b) BN--MLP integration}} \\"]
        for _, r in ab[ab.suite == suite].iterrows():
            lines.append(f"{names[r.variant]} & {r.roc_auc_mean:.3f} $\\pm$ {r.roc_auc_sd:.3f} & "
                         f"{r.log_loss_mean:.3f} & {r.fit_seconds:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def search_table(sel):
    rows = []
    for m in ORDER:
        d = sel[sel.model == m]
        params = [json.loads(p) for p in d.best_params]
        keys = sorted(params[0])
        parts = []
        for k in keys:
            vals = [p[k] for p in params]
            kk = k.replace("model__", "").replace("_", r"\_")
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals) \
                    and not all(float(v).is_integer() for v in vals):
                parts.append(f"{kk} {min(vals):.3g}--{max(vals):.3g}")
            else:
                cnt = Counter("None" if v is None else str(v) for v in vals)
                parts.append(kk + " " + ", ".join(f"{v}({n})" for v, n in cnt.most_common()))
        rows.append((m, "; ".join(parts), d.runtime_seconds.sum()))
    lines = [r"\begin{tabular}{p{0.16\linewidth}p{0.62\linewidth}r}", r"\toprule",
             r"Model & Selected values over the 5 pooled outer folds & Time (s) \\", r"\midrule"]
    for m, txt, t in rows:
        lines.append(f"{SHORT[m]} & {txt} & {t:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main(metrics_dir: str, out_dir: str) -> None:
    md, out = Path(metrics_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(md / "summary_pooled.csv"); sx = pd.read_csv(md / "summary_cross-cohort.csv")
    cp = pd.read_csv(md / "bootstrap_ci_pooled.csv"); cx = pd.read_csv(md / "bootstrap_ci_cross-cohort.csv")
    ab = pd.read_csv(md / "ablations.csv"); sel = pd.read_csv(md / "selected_hyperparameters_pooled.csv")
    pp = pd.read_csv(out / "paired_differences_pooled.csv")
    pc = pd.read_csv(out / "paired_differences_cross-cohort.csv")

    write(out, "t_pooled.tex", main_table(sp, cp, "pooled"))
    write(out, "t_cross.tex", main_table(sx, cx, "cross-cohort"))
    write(out, "t_cohort_auc.tex", cohort_table(sp, sx))
    shift, rows = shift_table(sp, sx)
    write(out, "t_shift.tex", shift)
    write(out, "t_paired.tex", paired_table(pp, pc))
    write(out, "t_ablation.tex", ablation_table(ab))
    write(out, "t_search.tex", search_table(sel))

    keys = {
        "pooled_auc": sp[sp.cohort == "ALL"].set_index("model").roc_auc.round(3).to_dict(),
        "cross_auc": sx[sx.cohort == "ALL"].set_index("model").roc_auc.round(3).to_dict(),
        "mean_heldout_auc": {m: round(v[2], 3) for m, v in rows.items()},
        "loco_minus_pooled_auc": {m: round(v[3], 3) for m, v in rows.items()},
        "n_trials_total_pooled": int(len(pd.read_csv(md / "optuna_trials_pooled.csv"))),
        "n_trials_total_cross": int(len(pd.read_csv(md / "optuna_trials_cross-cohort.csv"))),
    }
    (out / "key_numbers.json").write_text(json.dumps(keys, indent=2))
    print(json.dumps(keys, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
