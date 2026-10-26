"""Consistency audit between results, report, tables, figures, README and presentation outline.

    python -m src.evaluation.audit_report

Exits non-zero if any check fails. It does not recompute results; it checks that everything written
for readers still agrees with results/metrics/* and the configuration.
"""
from __future__ import annotations

import hashlib
import re
import sys

import pandas as pd

from src.data.loader import load_config, resolve
from src.evaluation import make_report_assets as mra


def _read(p) -> str:
    return resolve(p).read_text()


def _md5(p) -> str:
    return hashlib.md5(resolve(p).read_bytes()).hexdigest()


def check_bibliography() -> list[tuple[str, bool, str]]:
    tex, bib = _read("reports/main.tex"), _read("reports/references.bib")
    cited = {k.strip() for m in re.findall(r"\\cite\{([^}]*)\}", tex) for k in m.split(",")}
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.M))
    return [("every cited key exists in references.bib", cited <= keys, str(sorted(cited - keys))),
            ("every bib entry is cited", keys <= cited, str(sorted(keys - cited)))]


def check_files() -> list[tuple[str, bool, str]]:
    tex = _read("reports/main.tex")
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
    tabs = re.findall(r"\\input\{([^}]*)\}", tex)
    miss = [f for f in figs if not resolve(f"reports/figures/{f}").exists()] + [t for t in tabs if not resolve(f"reports/{t}").exists()]
    stale = [f for f in mra.FIGURES if _md5(f"reports/figures/{f}.png") != _md5(f"results/figures/{f}.png")]
    return [("figures and tables referenced by main.tex exist", not miss, str(miss)),
            ("report figures are identical to results/figures", not stale, str(stale)),
            ("main.pdf exists", resolve("reports/main.pdf").exists(), "compile with tectonic (reports/README.md)")]


def check_assets_not_stale(cfg) -> list[tuple[str, bool, str]]:
    mac = mra.build_macros(cfg)
    out = [("numbers.tex matches current results", _read("reports/tables/numbers.tex").split("\n", 1)[1] ==
            "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(mac.items())), "rerun make_report_assets")]
    for name, fn in {"fold_metrics": mra.table_fold_metrics, "pooled_ci": mra.table_pooled_ci, "calibration": mra.table_calibration,
                     "confusion": mra.table_confusion, "threshold": mra.table_threshold, "paired": mra.table_paired,
                     "attribution": mra.table_attribution}.items():
        out.append((f"tables/{name}.tex matches current results", _read(f"reports/tables/{name}.tex") == fn(cfg), ""))
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        mra.write_outline(mac, pathlib.Path(d) / "o.md")
        out.append(("presentation_outline.md matches current results", (pathlib.Path(d) / "o.md").read_text() == _read("reports/presentation_outline.md"), ""))
    return out


def check_claims(cfg) -> list[tuple[str, bool, str]]:
    """Qualitative statements made in the report/README, checked against the saved results."""
    met = resolve(cfg["paths"]["metrics"])
    d = pd.read_csv(met / "bootstrap_paired_differences.csv")
    f = pd.read_csv(met / "summary_fold_level.csv")
    p = pd.read_csv(met / "summary_pooled_repeats.csv")
    sel = pd.read_csv(met / "selected_hyperparameters.csv")
    f5, y = f[f.threshold_scheme == "fixed_0.5"].set_index("model"), f[f.threshold_scheme == "train_youden"].set_index("model")
    p5 = p[p.threshold_scheme == "fixed_0.5"].set_index("model")
    ex = lambda cmp, m: not d[(d.comparison == cmp) & (d.metric == m)].ci_includes_zero.iloc[0]
    res = [
        ("all ROC/PR AUC paired CIs include zero", bool(d[d.metric.isin(["roc_auc", "pr_auc"])].ci_includes_zero.all()), ""),
        ("no Brier paired CI excludes zero", bool(d[d.metric == "brier"].ci_includes_zero.all()), ""),
        ("log-loss CI excludes zero for NB-MLP and BN-NB but not BN-MLP",
         ex("Naive Bayes - MLP", "log_loss") and ex("Bayesian Network - Naive Bayes", "log_loss") and not ex("Bayesian Network - MLP", "log_loss"), ""),
        ("MLP has lowest Brier, log loss and ECE", all(f5.loc["MLP", c + "_mean"] == f5[c + "_mean"].min() for c in ("brier", "log_loss", "ece")), ""),
        ("Naive Bayes has the worst pooled ECE and calibration slope farthest from 1",
         p5.ece_mean.idxmax() == "Naive Bayes" and (p5.calibration_slope_mean - 1).abs().idxmax() == "Naive Bayes", ""),
        ("MLP calibration slope closest to 1", (p5.calibration_slope_mean - 1).abs().idxmin() == "MLP", ""),
        ("BN and NB slopes < 1 (over-confident)", bool((p5.loc[["Bayesian Network", "Naive Bayes"], "calibration_slope_mean"] < 1).all()), ""),
        ("Youden threshold lowered sensitivity for every model", all(y.loc[m, "sensitivity_mean"] < f5.loc[m, "sensitivity_mean"] for m in mra.ORDER), ""),
        ("BN/NB chose the top smoothing value in every fold",
         bool((sel[sel.model == "Bayesian Network"]["model__pseudo_count"] == 2.0).all() and (sel[sel.model == "Naive Bayes"]["model__alpha"] == 2.0).all()), ""),
        ("grids in config match the text (alpha/pseudo-count {0.5,1,2}; MLP wd {1e-3,1e-2}, dropout {0.3,0.5})",
         cfg["tuning"]["naive_bayes"]["model__alpha"] == [0.5, 1.0, 2.0] and cfg["tuning"]["bayesian_network"]["model__pseudo_count"] == [0.5, 1.0, 2.0]
         and cfg["tuning"]["mlp"] == {"model__weight_decay": [0.001, 0.01], "model__dropout": [0.3, 0.5]}, ""),
        ("MLP architecture in config matches the text (16, 8; 481 parameters)",
         list(cfg["models"]["mlp"]["hidden_sizes"]) == [16, 8] and 20 * 16 + 16 + 16 * 8 + 8 + 8 + 1 == 481, ""),
        ("evaluation design in config matches the text (5x5 outer, 5 inner, threshold 0.5, 2000 bootstrap)",
         cfg["cv"]["outer_folds"] == 5 and cfg["cv"]["outer_repeats"] == 5 and cfg["cv"]["inner_folds"] == 5
         and cfg["evaluation"]["primary_threshold"] == 0.5 and cfg["evaluation"]["n_bootstrap"] == 2000, ""),
    ]
    return res


def check_readme(cfg) -> list[tuple[str, bool, str]]:
    readme = _read("README.md")
    met = resolve(cfg["paths"]["metrics"])
    f = pd.read_csv(met / "summary_fold_level.csv")
    f = f[f.threshold_scheme == "fixed_0.5"].set_index("model")
    ci = pd.read_csv(met / "bootstrap_ci.csv")
    p = pd.read_csv(met / "summary_pooled_repeats.csv")
    p = p[p.threshold_scheme == "fixed_0.5"].set_index("model")
    att = pd.read_csv(met / "mlp_attribution.csv", index_col=0)
    expected = []
    for n in mra.ORDER:
        expected += [" | ".join(f"{f.loc[n, c + '_mean']:.3f} ± {f.loc[n, c + '_sd']:.3f}" for c in
                                ("roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "brier", "log_loss"))]
        r = ci[ci.model == n].set_index("metric")
        expected += [" | ".join(f"{r.loc[k, 'estimate']:.3f} [{r.loc[k, 'ci_low']:.3f}, {r.loc[k, 'ci_high']:.3f}]" for k in ("roc_auc", "pr_auc", "brier"))]
        expected += [f"{p.loc[n, 'ece_mean']:.3f} ± {p.loc[n, 'ece_sd']:.3f} | {p.loc[n, 'calibration_slope_mean']:.2f} ± {p.loc[n, 'calibration_slope_sd']:.2f} | "
                     f"{p.loc[n, 'calibration_intercept_mean']:.2f} ± {p.loc[n, 'calibration_intercept_sd']:.2f}"]
    expected += [f"| {k} | {r.mean_abs_shap:.3f} | {r.perm_auc_drop:.3f} ± {r.perm_auc_drop_sd:.3f} |" for k, r in att.iterrows()]
    missing = [e for e in expected if e not in readme]
    import subprocess
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings"], cwd=resolve("."),
                         capture_output=True, text=True).stdout
    n_tests = int(re.search(r"(\d+) tests? collected", out).group(1))
    stated = {int(x) for x in re.findall(r"# (\d+) tests;", readme) + re.findall(r"(\d+) tests \(unit", readme)}
    return [("README result tables match results/metrics", not missing, str(missing[:3])),
            (f"README test count matches collected tests ({n_tests})", stated == {n_tests}, f"README says {sorted(stated)}")]


def main() -> int:
    cfg = load_config()
    results = (check_bibliography() + check_files() + check_assets_not_stale(cfg) + check_claims(cfg) + check_readme(cfg))
    for name, ok, info in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {info}" if not ok and info else ""))
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
