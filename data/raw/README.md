# Raw data

This directory is **not committed to Git** (see `.gitignore`); only this file is. Place the original archive here:

    data/raw/heart+disease.zip

Then run `python -m src.data.verify`, which lists the archive, extracts only `processed.cleveland.data` (the primary file) into this directory, checks it, and writes `results/metrics/data_verification.json`. `heart-disease.names` (the variable documentation) can be extracted with `unzip data/raw/heart+disease.zip heart-disease.names -d data/raw`.

## Source and terms
- UCI Machine Learning Repository, *Heart Disease* dataset: https://archive.ics.uci.edu/dataset/45/heart+disease (donated 6/30/1988).
- Creators: Andras Janosi (Hungarian Institute of Cardiology, Budapest), William Steinbrunn (University Hospital, Zurich), Matthias Pfisterer (University Hospital, Basel), Robert Detrano (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation). The Cleveland file comes from Detrano.
- License stated on the UCI page: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, which allows sharing and adaptation for any purpose with appropriate credit. Checked against the UCI page on 2026-10-22; re-check it before redistributing. Because attribution is required and the archive is third-party data, this repository links to the source instead of re-hosting it.
- Suggested citation (as given by UCI): Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
- The data owners also ask (in `heart-disease.names`) that publications name the principal investigator responsible for the data from each institution used.
- The archive's own `WARNING` file says `cleveland.data` is corrupted and `processed.cleveland.data` is the usable file. This project uses only `processed.cleveland.data`.

## Integrity (this project's copy)
- `heart+disease.zip` SHA-256: `b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc`
- `processed.cleveland.data` SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`
