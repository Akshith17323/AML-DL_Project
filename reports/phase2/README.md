# Phase 2 IEEE report

`main.tex` uses the IEEE conference class. Tables are generated from the complete saved Phase 2 runs.

```bash
python -m src.evaluation.run_phase2 --protocol pooled
python -m src.evaluation.run_phase2 --protocol cross-cohort
python -m src.evaluation.run_phase2 --protocol pooled --analyse
python -m src.evaluation.run_phase2 --protocol cross-cohort --analyse
python -m src.evaluation.phase2_ablations
python -m src.models.learning_curves_phase2
python -m src.evaluation.make_phase2_report_assets
cd reports/phase2
tectonic main.tex                 # or latexmk -pdf main.tex
```

The report remains an academic model comparison, not clinical validation.
