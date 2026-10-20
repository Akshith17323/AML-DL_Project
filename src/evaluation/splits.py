"""Shared stratified outer folds: every model must use exactly these splits."""
from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from src.data.loader import load_config, resolve


def outer_folds(y, config: dict | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic (train_idx, val_idx) list from config (seed, n folds, repeats)."""
    cfg = config or load_config()
    n, rep, seed = cfg["cv"]["outer_folds"], cfg["cv"].get("outer_repeats", 1), cfg["project"]["seed"]
    cv = (StratifiedKFold(n, shuffle=cfg["cv"]["shuffle"], random_state=seed) if rep == 1
          else RepeatedStratifiedKFold(n_splits=n, n_repeats=rep, random_state=seed))
    return [(tr, va) for tr, va in cv.split(np.zeros(len(y)), y)]


def save_folds(folds, path=None) -> None:
    p = resolve(path or "results/metrics/outer_folds.json")
    p.write_text(json.dumps([{"train": tr.tolist(), "val": va.tolist()} for tr, va in folds]))
