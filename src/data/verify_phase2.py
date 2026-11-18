"""Verify all four Phase 2 cohorts and write their provenance manifest."""
from __future__ import annotations

import json

from src.data.loader import (PROJECT_ROOT, PHASE2_CONFIG_PATH, cohort_file_hashes,
                             extract_raw_file, load_all_cohorts, load_config, resolve,
                             sha256, verify_all_cohorts)


def main() -> dict:
    cfg = load_config(PHASE2_CONFIG_PATH)
    df = load_all_cohorts(config=cfg)
    source_files = {
        cohort: extract_raw_file(cfg["data"]["archive"], "data/raw", member)
        for cohort, member in cfg["data"]["cohorts"].items()
    }
    report = verify_all_cohorts(df)
    report["cohort_source_files"] = {
        cohort: str(path.relative_to(PROJECT_ROOT)) for cohort, path in source_files.items()
    }
    report["cohort_file_sha256"] = cohort_file_hashes(config=cfg)
    report["archive_sha256"] = sha256(resolve(cfg["data"]["archive"]))
    out = resolve(cfg["paths"]["metrics"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
