from src.data.loader import PROJECT_ROOT, PHASE2_CONFIG_PATH, load_config
from src.evaluation.phase2_ablations import integration_variants, mlp_variants
from src.evaluation.run_phase2 import cross_cohort_splits, pooled_splits


def test_phase2_splits_cover_every_patient_once():
    from src.data.loader import get_xy, load_all_cohorts
    cfg = load_config(PHASE2_CONFIG_PATH); df = load_all_cohorts(config=cfg); X, y = get_xy(df)
    for splits in (pooled_splits(y, cfg), cross_cohort_splits(X, y, df.cohort)):
        validation = [i for _, va in splits for i in va]
        assert sorted(validation) == list(range(920))
        assert all(not set(tr) & set(va) for tr, va in splits)


def test_ablation_suites_change_only_declared_components():
    cfg = load_config(PHASE2_CONFIG_PATH)
    mlp = mlp_variants(cfg)
    assert mlp["no_dropout"].dropout == 0
    assert mlp["no_weight_decay"].weight_decay == 0
    assert mlp["no_early_stopping"].early_stopping is False
    assert mlp["one_hidden_layer"].hidden_layout == "16"
    assert set(integration_variants(cfg)) == {"BN", "MLP", "simple_average", "learned_stack"}


def test_ieee_report_sources_exist_and_do_not_claim_final_results():
    report = PROJECT_ROOT / "reports/phase2/main.tex"
    text = report.read_text()
    assert "\\documentclass[conference]{IEEEtran}" in text
    assert "920" in text and "OptunaSearchCV" in text and "cross-fits" in text
    assert "Leave-one-cohort-out results" in text and "does not demonstrate" in text
    assert (report.parent / "references.bib").exists()
    for name in ("pooled_results.tex", "cross_cohort_results.tex", "ablations.tex"):
        assert (report.parent / "tables" / name).exists()


def test_generated_phase2_notebooks_default_to_no_training():
    import json
    for path in (PROJECT_ROOT / "src/evaluation/run_phase2.ipynb",
                 PROJECT_ROOT / "src/evaluation/phase2_ablations.ipynb"):
        notebook = json.loads(path.read_text())
        code = "\n".join(cell.get("source", "") if isinstance(cell.get("source", ""), str)
                         else "".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "MODE = None" in code or "RUN_PILOT = False" in code


def test_phase2_preserves_phase1_layout_and_notebook_companions():
    required = [
        "config.yaml", "config.phase2.yaml", "notebooks/01_eda.ipynb",
        "notebooks/02_phase2_all_cohorts.ipynb", "data/raw/processed.cleveland.data",
        "data/raw/processed.hungarian.data", "data/raw/processed.switzerland.data",
        "data/raw/processed.va.data", "src/data/loader.py",
        "src/data/verify.py", "src/data/verify_phase2.py", "src/data/eda.py",
        "src/data/eda_phase2.py", "src/features/preprocessing.py",
        "src/models/pipelines.py", "src/models/mlp.py", "src/models/hybrid.py",
        "src/evaluation/run_evaluation.py", "src/evaluation/run_phase2.py",
        "reports/project_report/Heart_Disease_Risk_Modelling_Report.docx",
        "reports/presentation/Heart_Disease_Risk_Modelling.pptx", "reports/phase2/main.tex",
    ]
    assert not [path for path in required if not (PROJECT_ROOT / path).exists()]

    python_files = [
        path for folder in ("app", "src", "tests")
        for path in (PROJECT_ROOT / folder).rglob("*.py")
        if path.read_text().strip()
    ]
    assert not [path for path in python_files if not path.with_suffix(".ipynb").exists()]
