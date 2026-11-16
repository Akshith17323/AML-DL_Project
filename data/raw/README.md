# Raw data

This directory is **not committed to Git** (see `.gitignore`); only this file is. Place the original archive here:

    data/raw/heart+disease.zip

For the Phase 1 Cleveland-only path, run `python -m src.data.verify`.

For Phase 2, run:

    python -m src.data.verify_phase2

This extracts and verifies the four independent processed cohort files so they are visible in this directory:

- `processed.cleveland.data`
- `processed.hungarian.data`
- `processed.switzerland.data`
- `processed.va.data`

It records row counts, target counts, missingness, archive/member hashes, and source paths in `results/phase2/main/metrics/data_verification.json`. The analysis notebook is `notebooks/02_phase2_all_cohorts.ipynb`.

`heart-disease.names` contains the variable documentation.

## Source and terms
- UCI Machine Learning Repository, *Heart Disease* dataset: https://archive.ics.uci.edu/dataset/45/heart+disease (donated 6/30/1988).
- Creators: Andras Janosi (Hungarian Institute of Cardiology, Budapest), William Steinbrunn (University Hospital, Zurich), Matthias Pfisterer (University Hospital, Basel), Robert Detrano (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation). The Cleveland file comes from Detrano.
- License stated on the UCI page: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, which allows sharing and adaptation for any purpose with appropriate credit. Checked against the UCI page on 2026-10-22; re-check it before redistributing. Because attribution is required and the archive is third-party data, this repository links to the source instead of re-hosting it.
- Suggested citation (as given by UCI): Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
- The data owners also ask (in `heart-disease.names`) that publications name the principal investigator responsible for the data from each institution used.
- The archive's own `WARNING` file says `cleveland.data` is corrupted. Phase 2 therefore uses exactly the four `processed.*.data` cohort files listed above. Raw and reprocessed alternatives are not added because they are alternate representations, not extra patients.

## Integrity (this project's copy)
- `heart+disease.zip` SHA-256: `b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc`
- `processed.cleveland.data` SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`
- `processed.hungarian.data` SHA-256: `d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796`
- `processed.switzerland.data` SHA-256: `834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314`
- `processed.va.data` SHA-256: `e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f`
