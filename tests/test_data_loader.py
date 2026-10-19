import numpy as np
import pandas as pd
import pytest

from src.data import loader
from src.data.loader import (EXPECTED_COLUMNS, FEATURE_COLUMNS, load_cleveland, load_raw,
                             map_binary_target, verify_dataset, get_xy)

DOCUMENTED = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
              "exang", "oldpeak", "slope", "ca", "thal", "num"]


def _write(tmp_path, rows):
    p = tmp_path / "d.data"
    p.write_text("\n".join(",".join(map(str, r)) for r in rows) + "\n")
    return p


ROW = [63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0, 0]


def test_column_order_constants():
    assert EXPECTED_COLUMNS == DOCUMENTED
    assert loader.load_config()["data"]["columns"] == DOCUMENTED


def test_real_file_column_order_and_shape():
    df = load_raw()
    assert list(df.columns) == DOCUMENTED
    assert df.shape == (303, 14)


def test_missing_marker_becomes_nan(tmp_path):
    row = ROW.copy(); row[11] = "?"; row[12] = "?"
    df = load_raw(_write(tmp_path, [ROW, row]))
    assert df["ca"].isna().tolist() == [False, True]
    assert df["thal"].isna().tolist() == [False, True]
    assert pd.api.types.is_float_dtype(df["ca"])
    assert df.isna().sum().sum() == 2


def test_real_file_missing_counts():
    df = load_raw()
    assert df.isna().sum()[["ca", "thal"]].to_dict() == {"ca": 4, "thal": 2}
    assert df.drop(columns=["ca", "thal"]).isna().sum().sum() == 0


def test_wrong_column_count_rejected(tmp_path):
    with pytest.raises(Exception):
        load_raw(_write(tmp_path, [ROW[:-1]]))


def test_target_mapping():
    s = pd.Series([0, 1, 2, 3, 4, 0])
    assert map_binary_target(s).tolist() == [0, 1, 1, 1, 1, 0]


@pytest.mark.parametrize("bad", [5, -1, 2.5])
def test_invalid_target_values_raise(bad):
    with pytest.raises(ValueError, match="Invalid target"):
        map_binary_target(pd.Series([0, 1, bad]))


def test_missing_target_raises():
    with pytest.raises(ValueError, match="missing"):
        map_binary_target(pd.Series([0, np.nan]))


def test_invalid_target_in_file_raises(tmp_path):
    bad = ROW.copy(); bad[13] = 7
    with pytest.raises(ValueError):
        load_cleveland(_write(tmp_path, [ROW, bad]))


def test_original_target_preserved_and_binary_added():
    df = load_cleveland()
    assert set(df["num"].unique()) == {0, 1, 2, 3, 4}
    assert ((df["num"] > 0).astype(int) == df["target"]).all()
    X, y = get_xy(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert "num" not in X and "target" not in X
    assert len(X) == len(y) == 303


def test_verify_dataset_keys():
    rep = verify_dataset(load_cleveland())
    assert rep["n_rows"] == 303
    assert sum(rep["binary_target_counts"].values()) == 303
    assert sum(rep["num_value_counts"].values()) == 303
