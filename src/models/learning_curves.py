"""MLP learning curves on the first repeat of the shared outer folds (default hyper-parameters).

Fits the MLP pipeline on each training fold, checks the probability output, and plots the inner training and
validation loss (figure used in the report). No performance metric is computed here.
Run from the project root:  python -m src.models.learning_curves
"""
import json
import time
import warnings
from importlib.metadata import version

import numpy as np
import torch
from sklearn.base import clone

from src.data.loader import get_xy, load_cleveland, load_config, resolve
from src.evaluation.curves import plot_learning_curves
from src.evaluation.splits import outer_folds
from src.models.pipelines import build_mlp_pipeline


def main() -> dict:
    cfg = load_config()
    X, y = get_xy(load_cleveland(config=cfg))
    folds = outer_folds(y, cfg)[:cfg["cv"]["outer_folds"]]  # first repeat
    recs, hists, bests = [], [], []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i, (tr, va) in enumerate(folds):
            t0 = time.perf_counter()
            pipe = clone(build_mlp_pipeline(cfg)).fit(X.iloc[tr], y.iloc[tr])
            fit_s = time.perf_counter() - t0
            m = pipe.named_steps["model"]
            p = pipe.predict_proba(X.iloc[va])
            assert p.shape == (len(va), 2) and np.allclose(p.sum(axis=1), 1) and np.isfinite(p).all()
            recs.append({"fold": i, "n_train": len(tr), "n_val": len(va), "n_features": m.n_features_in_,
                         "n_parameters": m.n_parameters_, "epochs_run": m.n_epochs_run_,
                         "best_epoch": m.best_epoch_, "stopped_early": bool(m.stopped_early_),
                         "best_inner_val_loss": round(min(m.history_["val_loss"]), 4),
                         "min_p_pos": round(float(p[:, 1].min()), 4), "max_p_pos": round(float(p[:, 1].max()), 4),
                         "fit_s": round(fit_s, 2)})
            hists.append(m.history_); bests.append(m.best_epoch_)
            print(recs[-1])
    fig = plot_learning_curves(hists, bests, resolve(cfg["paths"]["figures"]) / "mlp_learning_curves.png")
    report = {"torch": torch.__version__, "numpy": version("numpy"), "scikit-learn": version("scikit-learn"),
              "hyperparameters": cfg["models"]["mlp"], "seed": cfg["project"]["seed"],
              "warnings": sorted({str(w.message)[:200] for w in caught}),
              "folds": recs, "histories": hists}
    (resolve(cfg["paths"]["metrics"]) / "mlp_learning_curves.json").write_text(json.dumps(report))
    print("figure:", fig.relative_to(resolve(".")), "| warnings:", report["warnings"])
    return report


if __name__ == "__main__":
    main()
