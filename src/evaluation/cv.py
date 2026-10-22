"""Nested cross-validated evaluation with identical outer folds for every model.

For each outer training fold: (1) inner stratified K-fold over a small hyper-parameter
grid, scored by inner out-of-fold log loss; (2) pick the best setting; (3) take a
decision threshold (Youden) from that setting's inner out-of-fold predictions;
(4) refit on the whole outer training fold; (5) predict the outer validation fold.
The outer validation labels are used only to score the predictions afterwards.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import youden_threshold


def fold_hash(folds) -> str:
    h = hashlib.sha256()
    for tr, va in folds:
        h.update(np.asarray(tr, dtype=np.int64).tobytes()); h.update(np.asarray(va, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def inner_oof(pipe, params: dict, X, y, inner_splits) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for tr, va in inner_splits:
        m = clone(pipe).set_params(**params).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("Inner out-of-fold predictions incomplete")
    return oof


def evaluate_model(name: str, pipe, grid: dict, X: pd.DataFrame, y: pd.Series, outer, cfg: dict,
                   n_splits_per_repeat: int, verbose: bool = True, fold_offset: int = 0):
    """Return (oof_df, selection_df). oof_df has one row per (repeat, outer fold, validation patient).

    ``fold_offset`` is the global id of ``outer[0]`` (lets a single fold be re-run exactly: the inner
    split seed depends on the global outer fold id)."""
    combos = expand_grid(grid)
    seed, k_inner = cfg["project"]["seed"], cfg["cv"]["inner_folds"]
    rows, sel = [], []
    for i, (tr, va) in enumerate(outer):
        f = fold_offset + i
        rep, fold = divmod(f, n_splits_per_repeat)
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        # inner splits depend only on (seed, outer fold id): identical for all models and settings
        inner = list(StratifiedKFold(k_inner, shuffle=True, random_state=seed + f).split(np.zeros(len(ytr)), ytr))
        scores, oofs = [], []
        for params in combos:
            oof = inner_oof(pipe, params, Xtr, ytr, inner)
            oofs.append(oof)
            scores.append(log_loss(ytr, np.clip(oof, 1e-15, 1 - 1e-15), labels=[0, 1]))
        b = int(np.argmin(scores))
        thr = youden_threshold(ytr, oofs[b])
        final = clone(pipe).set_params(**combos[b]).fit(Xtr, ytr)
        p = final.predict_proba(X.iloc[va])
        assert list(final.named_steps["model"].classes_) == [0, 1]
        rows.append(pd.DataFrame({"model": name, "repeat": rep, "fold": fold, "outer_id": f,
                                  "patient": va, "y_true": y.iloc[va].to_numpy(), "p_pos": p[:, 1],
                                  "p_neg": p[:, 0], "youden_threshold": thr}))
        sel.append({"model": name, "repeat": rep, "fold": fold, "outer_id": f,
                    **{k: v for k, v in combos[b].items()}, "inner_log_loss": scores[b],
                    "youden_threshold": thr, "n_train": len(tr), "n_val": len(va)})
        if verbose:
            print(f"[{name}] outer {i + 1}/{len(outer)} best={combos[b]} inner_logloss={scores[b]:.4f}", flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(sel)
