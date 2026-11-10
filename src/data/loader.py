"""Loading and validation of the UCI Cleveland heart disease data."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PHASE2_CONFIG_PATH = PROJECT_ROOT / "config.phase2.yaml"

FEATURE_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                   "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
ORIGINAL_TARGET = "num"
BINARY_TARGET = "target"
VALID_NUM_VALUES = (0, 1, 2, 3, 4)
EXPECTED_COLUMNS = FEATURE_COLUMNS + [ORIGINAL_TARGET]
RAW_MEMBER = "processed.cleveland.data"
COHORT_FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungarian": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA": "processed.va.data",
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def inspect_archive(archive_path: str | Path) -> list[str]:
    """List archive members without extracting anything."""
    with zipfile.ZipFile(resolve(archive_path)) as zf:
        return zf.namelist()


def extract_raw_file(archive_path: str | Path, dest_dir: str | Path,
                     member: str = RAW_MEMBER) -> Path:
    """Extract a single named member into dest_dir (no-op if already present)."""
    dest = resolve(dest_dir)
    out = dest / member
    if out.exists():
        return out
    with zipfile.ZipFile(resolve(archive_path)) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"{member} not found in {archive_path}")
        zf.extract(member, dest)
    return out


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def map_binary_target(num: pd.Series) -> pd.Series:
    """Map num 0 -> 0 and {1,2,3,4} -> 1. Raises on any other value (incl. NaN)."""
    if num.isna().any():
        raise ValueError("Target contains missing values")
    invalid = sorted(set(num.unique()) - set(VALID_NUM_VALUES))
    if invalid:
        raise ValueError(f"Invalid target values: {invalid}; expected {list(VALID_NUM_VALUES)}")
    return (num > 0).astype(int).rename(BINARY_TARGET)


def load_raw(path: str | Path | None = None, config: dict | None = None) -> pd.DataFrame:
    """Read the headerless file, apply column names, treat '?' as NaN.

    The original `num` column is kept unchanged; the binary target is not added here.
    """
    cfg = config or load_config()
    data_cfg = cfg["data"]
    columns = data_cfg["columns"]
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"Configured columns differ from documented order: {columns}")
    file_path = resolve(path or data_cfg["raw_file"])
    df = pd.read_csv(file_path, header=None, na_values=[data_cfg["missing_marker"]],
                     skipinitialspace=True)
    if df.shape[1] != len(columns) or df.isna().all(axis=1).any():
        raise ValueError(f"Expected {len(columns)} columns per row, found {df.shape[1]}")
    df.columns = columns
    return df


def load_cleveland(path: str | Path | None = None, config: dict | None = None) -> pd.DataFrame:
    """Load data with names, NaN for missing, original `num`, and binary `target`."""
    df = load_raw(path, config)
    df[BINARY_TARGET] = map_binary_target(df[ORIGINAL_TARGET])
    return df


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features (no target columns) and binary target."""
    return df[FEATURE_COLUMNS].copy(), df[BINARY_TARGET].copy()


def load_all_cohorts(archive_path: str | Path | None = None,
                     config: dict | None = None) -> pd.DataFrame:
    """Load the four independent processed UCI cohorts from the source archive.

    Raw/reprocessed archive members are alternate forms of these cohorts and are
    deliberately not concatenated as extra patients.
    """
    cfg = config or load_config(PHASE2_CONFIG_PATH)
    data_cfg = cfg["data"]
    cohorts = data_cfg.get("cohorts", COHORT_FILES)
    if dict(cohorts) != COHORT_FILES:
        raise ValueError(f"Phase 2 cohorts must be exactly {COHORT_FILES}")
    archive = resolve(archive_path or data_cfg["archive"])
    frames = []
    with zipfile.ZipFile(archive) as zf:
        absent = [member for member in cohorts.values() if member not in zf.namelist()]
        if absent:
            raise FileNotFoundError(f"Cohort file(s) missing from archive: {absent}")
        for cohort, member in cohorts.items():
            with zf.open(member) as source:
                df = pd.read_csv(source, header=None, na_values=[data_cfg["missing_marker"]],
                                 skipinitialspace=True)
            if df.shape[1] != len(EXPECTED_COLUMNS) or df.isna().all(axis=1).any():
                raise ValueError(f"{member}: expected {len(EXPECTED_COLUMNS)} columns, found {df.shape[1]}")
            df.columns = EXPECTED_COLUMNS
            df[BINARY_TARGET] = map_binary_target(df[ORIGINAL_TARGET])
            df.insert(0, "cohort_patient_id", [f"{cohort}:{i}" for i in range(len(df))])
            df.insert(0, "cohort", cohort)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if out["cohort_patient_id"].duplicated().any():
        raise RuntimeError("Cohort patient identifiers are not unique")
    return out


def cohort_file_hashes(archive_path: str | Path | None = None,
                       config: dict | None = None) -> dict[str, str]:
    """SHA-256 for each processed member used by Phase 2."""
    cfg = config or load_config(PHASE2_CONFIG_PATH)
    archive = resolve(archive_path or cfg["data"]["archive"])
    with zipfile.ZipFile(archive) as zf:
        return {cohort: hashlib.sha256(zf.read(member)).hexdigest()
                for cohort, member in cfg["data"]["cohorts"].items()}


def verify_all_cohorts(df: pd.DataFrame) -> dict:
    """Observed Phase 2 counts and missingness, overall and by cohort."""
    required = {"cohort", "cohort_patient_id", *EXPECTED_COLUMNS, BINARY_TARGET}
    if not required <= set(df.columns):
        raise ValueError(f"Combined data missing columns: {sorted(required - set(df.columns))}")
    by_cohort = {}
    for cohort, group in df.groupby("cohort", sort=False):
        by_cohort[cohort] = {
            "n_rows": int(len(group)),
            "negative": int((group[BINARY_TARGET] == 0).sum()),
            "positive": int((group[BINARY_TARGET] == 1).sum()),
            "missing_predictor_cells": int(group[FEATURE_COLUMNS].isna().sum().sum()),
            "missing_per_feature": {c: int(n) for c, n in group[FEATURE_COLUMNS].isna().sum().items() if n},
            "duplicate_rows": int(group[EXPECTED_COLUMNS].duplicated().sum()),
        }
    return {
        "n_rows": int(len(df)),
        "negative": int((df[BINARY_TARGET] == 0).sum()),
        "positive": int((df[BINARY_TARGET] == 1).sum()),
        "cohorts": by_cohort,
        "patient_ids_unique": bool(df["cohort_patient_id"].is_unique),
    }


def verify_dataset(df: pd.DataFrame) -> dict:
    """Observed dataset statistics (no assumptions)."""
    return {
        "n_rows": int(df.shape[0]),
        "n_columns_raw": int(len(EXPECTED_COLUMNS)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "n_rows_with_missing": int(df.isna().any(axis=1).sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "num_value_counts": {int(k): int(v) for k, v in
                             df[ORIGINAL_TARGET].value_counts().sort_index().items()},
        "binary_target_counts": ({int(k): int(v) for k, v in
                                  df[BINARY_TARGET].value_counts().sort_index().items()}
                                 if BINARY_TARGET in df else None),
    }
