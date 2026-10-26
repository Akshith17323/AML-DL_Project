# Reports

| File | What it is |
|---|---|
| `project_report/Heart_Disease_Risk_Modelling_Report.docx` | Phase 1 project report (about 15 pages, Word) |
| `presentation/Heart_Disease_Risk_Modelling.pptx` | Phase 1 presentation (18 slides, presenter marked on each slide, speaker notes) |

Both were generated from `results/metrics/` and `results/figures/` after running

```bash
python -m src.evaluation.run_evaluation
python -m src.evaluation.interpret
```

They are not regenerated automatically: if the evaluation is rerun, re-check every number in them.

Before submitting:
- fill in roll numbers, course, supervisor and institution on the report's title page and slide 1;
- in Word, right-click the table of contents and choose **Update Field** after any edit that moves pages;
- export the report to PDF from Word (File > Save As > PDF).
