# Report build instructions

`main.tex` is an IEEE conference-format paper (`IEEEtran`) with `references.bib`. All numbers in the text, tables and the presentation outline are generated from `results/metrics/*`; do not edit `tables/*.tex` or `figures/*` by hand.

## Regenerate assets and build

```bash
# from the project root, after the experiments have been run (see the top-level README)
python -m src.evaluation.make_report_assets      # writes reports/tables/*.tex, numbers.tex, figures/*.png, presentation_outline.md

# compile (any one of these)
cd reports
tectonic -X compile main.tex                     # tested: Tectonic 0.17.0; downloads TeX support files on first run (needs internet)
latexmk -pdf main.tex                            # TeX Live / MacTeX with IEEEtran, booktabs, cite installed (not tested here)
pdflatex main && bibtex main && pdflatex main && pdflatex main    # manual equivalent (not tested here)

# audit report, README, tables, figures and outline against the results
cd .. && python -m src.evaluation.audit_report
```

Tectonic was installed with `brew install tectonic` (remove with `brew uninstall tectonic`). Compilation with Tectonic produces a few cosmetic warnings (Times font-shape substitution; one slightly overfull line in the references).

## Files
- `main.tex`, `references.bib`: the paper (14 required sections; author block is a placeholder to be filled in by the team).
- `tables/numbers.tex`: generated number macros; `tables/*.tex`: generated tables; `tables/model_comparison_fold_level.csv`: compact CSV table.
- `figures/`: copies of `results/figures/*.png` used by the paper.
- `presentation_outline.md`: 12-slide outline (generated).
- `main.pdf`: compiled output (regenerate; do not edit).

## References
Every entry in `references.bib` was checked on 2026-10-22 against Crossref (DOIs), arXiv, JMLR or Open Library (ISBNs). Entries without a DOI are cited by arXiv number or journal volume/pages. Re-verify before submission.
