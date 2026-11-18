"""Refit the first pooled-fold MLP selection and save its training/stopping loss curve."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import PHASE2_CONFIG_PATH, get_xy, load_all_cohorts, load_config, resolve
from src.evaluation.run_phase2 import pooled_splits
from src.models.phase2 import build_mlp


def main() -> dict:
    cfg = load_config(PHASE2_CONFIG_PATH)
    X, y = get_xy(load_all_cohorts(config=cfg)); train, _ = pooled_splits(y, cfg)[0]
    selected_path = resolve(cfg["paths"]["metrics"]) / "selected_hyperparameters_pooled.csv"
    selected = pd.read_csv(selected_path)
    row = selected[(selected.model == "MLP") & (selected.outer_id == 0)].iloc[0]
    params = json.loads(row.best_params)
    model = build_mlp(cfg).set_params(**params).fit(X.iloc[train], y.iloc[train])

    figures = resolve(cfg["paths"]["figures"]); figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(model.history_["train_loss"], label="weight-training loss")
    ax.plot(model.history_["val_loss"], label="stopping-validation loss")
    ax.axvline(model.best_epoch_, color="k", ls="--", lw=1, label=f"selected epoch {model.best_epoch_ + 1}")
    ax.set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Phase 2 MLP learning curve (pooled outer fold 0)")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "mlp_learning_curve.png", dpi=150); plt.close(fig)
    result = {"outer_id": 0, "best_params": params, "best_epoch": model.best_epoch_,
              "epochs_run": model.n_epochs_run_, "n_parameters": model.n_parameters_,
              "weight_train_rows": model.n_weight_train_, "stopping_rows": model.n_stopping_}
    (resolve(cfg["paths"]["metrics"]) / "mlp_learning_curve.json").write_text(json.dumps(result, indent=2))
    print(result)
    return result


if __name__ == "__main__":
    main()
