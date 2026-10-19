import pandas as pd

from src.data import eda
from src.data.loader import load_cleveland


def test_audit_real_data_clean():
    a = eda.audit(load_cleveland())
    assert a["duplicates"] == 0
    assert a["missing"] == {"ca": 4, "thal": 2}
    assert a["invalid_categorical"] == {} and a["implausible_continuous"] == {}


def test_audit_flags_invalid_values():
    df = load_cleveland()
    df.loc[0, "cp"] = 9
    df.loc[1, "age"] = 250
    a = eda.audit(df)
    assert a["invalid_categorical"] == {"cp": [9.0]}
    assert a["implausible_continuous"]["age"] == 1


def test_target_summary_counts():
    t = eda.target_summary(load_cleveland())
    assert t["original"] == {0: 164, 1: 55, 2: 36, 3: 35, 4: 13}
    assert t["binary"] == {0: 164, 1: 139}


def test_redundant_pairs_threshold():
    corr = pd.DataFrame([[1, .9, 0], [.9, 1, 0], [0, 0, 1]], columns=list("abc"), index=list("abc"))
    assert [(p["a"], p["b"]) for p in eda.redundant_pairs(corr)] == [("a", "b")]


def test_make_figures_writes_files(tmp_path):
    names = eda.make_figures(load_cleveland(), tmp_path)
    assert names and all((tmp_path / n).stat().st_size > 0 for n in names)
