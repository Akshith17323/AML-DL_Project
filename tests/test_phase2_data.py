import pandas as pd

from src.data.loader import (FEATURE_COLUMNS, cohort_file_hashes, get_xy,
                             load_all_cohorts, verify_all_cohorts)


def test_all_four_processed_cohorts_are_loaded_once():
    df = load_all_cohorts()
    assert df.groupby("cohort", sort=False).size().to_dict() == {
        "Cleveland": 303, "Hungarian": 294, "Switzerland": 123, "VA": 200}
    assert len(df) == 920 and df.cohort_patient_id.is_unique
    assert df.target.value_counts().sort_index().to_dict() == {0: 411, 1: 509}


def test_phase2_features_exclude_provenance_and_targets():
    df = load_all_cohorts(); X, y = get_xy(df)
    assert list(X) == FEATURE_COLUMNS and len(X) == len(y) == 920
    assert not {"cohort", "cohort_patient_id", "num", "target"} & set(X)


def test_phase2_missingness_and_hash_manifest():
    report = verify_all_cohorts(load_all_cohorts())
    assert report["cohorts"]["Cleveland"]["missing_predictor_cells"] == 6
    assert report["cohorts"]["Hungarian"]["missing_predictor_cells"] == 782
    assert report["cohorts"]["Switzerland"]["missing_predictor_cells"] == 273
    assert report["cohorts"]["VA"]["missing_predictor_cells"] == 698
    hashes = cohort_file_hashes()
    assert set(hashes) == {"Cleveland", "Hungarian", "Switzerland", "VA"}
    assert all(len(value) == 64 for value in hashes.values())


def test_no_target_missing_and_numeric_schema():
    df = load_all_cohorts()
    assert not df[["num", "target"]].isna().any(axis=None)
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in FEATURE_COLUMNS)
