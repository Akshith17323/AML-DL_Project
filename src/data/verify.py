"""Verify the Cleveland dataset and write results/metrics/data_verification.json.

Run from the project root:  python -m src.data.verify
"""
import json

from src.data.loader import (inspect_archive, extract_raw_file, load_cleveland,
                             load_config, resolve, sha256, verify_dataset, RAW_MEMBER)


def main() -> dict:
    cfg = load_config()
    archive = cfg["data"]["archive"]
    members = inspect_archive(archive)
    print(f"Archive members ({len(members)}):", ", ".join(members))
    if RAW_MEMBER not in members:
        raise SystemExit(f"{RAW_MEMBER} missing from archive")
    raw = extract_raw_file(archive, resolve("data/raw"))
    df = load_cleveland(raw, cfg)
    report = verify_dataset(df)
    report["archive_members"] = members
    report["raw_file"] = str(raw.relative_to(resolve(".")))
    report["raw_file_sha256"] = sha256(raw)
    report["archive_sha256"] = sha256(resolve(archive))
    out = resolve(cfg["paths"]["metrics"]) / "data_verification.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "archive_members"}, indent=2))
    return report


if __name__ == "__main__":
    main()
