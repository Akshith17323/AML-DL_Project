"""Tests for report asset generation and the consistency audit."""
import pytest

from src.data.loader import PROJECT_ROOT, load_config
from src.evaluation import audit_report as audit
from src.evaluation import make_report_assets as mra

needs_results = pytest.mark.skipif(not (PROJECT_ROOT / "results/metrics/bootstrap_ci.csv").exists(),
                                   reason="run the evaluation first")


def test_tex_escaping():
    assert mra.tex("a_b 5% & c") == r"a\_b 5\% \& c"


def test_write_macros_rejects_non_letter_names(tmp_path):
    mra.write_macros({"okName": "1"}, tmp_path / "n.tex")
    assert (tmp_path / "n.tex").read_text().count("\\newcommand") == 1
    with pytest.raises(AssertionError):
        mra.write_macros({"bad_name": "1"}, tmp_path / "m.tex")
    with pytest.raises(AssertionError):
        mra.write_macros({"bad2": "1"}, tmp_path / "m.tex")


@needs_results
def test_macros_cover_what_the_report_uses():
    import re
    cfg = load_config()
    mac = mra.build_macros(cfg)
    tex = (PROJECT_ROOT / "reports/main.tex").read_text()
    used = set(re.findall(r"\\([A-Za-z]+)", tex))
    defined_in_tex = set(re.findall(r"\\(?:newcommand|def)\{?\\([A-Za-z]+)", tex))
    candidates = {u for u in used if re.match(r"^(BN|NB|MLP|Diff|nP|nO|nB|num|pos|miss|seed|fold|WD|DO)", u)}
    assert candidates - set(mac) - defined_in_tex == set(), candidates - set(mac)


@needs_results
def test_tables_are_balanced_tabulars():
    cfg = load_config()
    for fn in (mra.table_fold_metrics, mra.table_pooled_ci, mra.table_calibration, mra.table_confusion,
               mra.table_threshold, mra.table_paired, mra.table_attribution):
        t = fn(cfg)
        assert t.count("\\begin{tabular}") == t.count("\\end{tabular}") == 1
        assert "nan" not in t.lower()


@needs_results
def test_consistency_audit_checks_that_do_not_need_the_compiled_pdf():
    cfg = load_config()
    results = (audit.check_bibliography() + audit.check_assets_not_stale(cfg) + audit.check_claims(cfg))
    failed = [(n, i) for n, ok, i in results if not ok]
    assert not failed, failed
