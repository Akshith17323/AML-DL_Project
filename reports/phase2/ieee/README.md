# Phase 2 IEEE report

`main.tex` is an `IEEEtran` conference paper. Every numeric table is generated from the saved
run in `results/phase2/main/metrics` and checked by an audit script; nothing is typed by hand.

```bash
# from the repository root
python reports/phase2/ieee/scripts/paired_comparisons.py results/phase2/main/metrics reports/phase2/ieee/tables
python reports/phase2/ieee/scripts/make_ieee_tables.py   results/phase2/main/metrics reports/phase2/ieee/tables
python reports/phase2/ieee/scripts/audit_report_numbers.py reports/phase2/ieee
cd reports/phase2/ieee && tectonic main.tex     # or: latexmk -pdf main.tex, or upload the folder to Overleaf
```

Figures are read directly from `results/phase2/main/figures/`. If you upload only this folder to
Overleaf, copy those PNGs into a `figures/` subfolder (the `\graphicspath` already looks there).

`paired_comparisons.py` adds the paired patient-level bootstrap differences (2,000 resamples) that the
saved run did not contain; it only reads saved out-of-fold predictions and does not retrain anything.

`Phase2_IEEE_Report.pdf` was compiled in a sandbox without the official `IEEEtran.cls`, using a
local stand-in that reproduces the IEEE conference layout. Rebuild with the official class
(the commands above) before submitting.
