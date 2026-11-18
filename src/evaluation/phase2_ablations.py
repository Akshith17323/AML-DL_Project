"""Controlled MLP and integration ablations; configurations are not HPO candidates."""
from __future__ import annotations

import copy

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data.loader import PHASE2_CONFIG_PATH, get_xy, load_all_cohorts, load_config, resolve
from src.models.hybrid import BNMLPAverageClassifier
from src.models.phase2 import build_hybrid, build_mlp
from src.models.pipelines import build_bn_pipeline


def mlp_variants(cfg: dict) -> dict:
    base = build_mlp(cfg)
    return {
        "full": base,
        "no_dropout": copy.deepcopy(base).set_params(dropout=0.0),
        "no_weight_decay": copy.deepcopy(base).set_params(weight_decay=0.0),
        "no_early_stopping": copy.deepcopy(base).set_params(early_stopping=False),
        "one_hidden_layer": copy.deepcopy(base).set_params(hidden_layout="16"),
    }


def integration_variants(cfg: dict) -> dict:
    mlp = build_mlp(cfg)
    h = build_hybrid(cfg)
    average = BNMLPAverageClassifier(
        bn_pseudo_count=h.bn_pseudo_count, mlp_hidden_layout=h.mlp_hidden_layout,
        mlp_dropout=h.mlp_dropout, mlp_weight_decay=h.mlp_weight_decay,
        mlp_learning_rate=h.mlp_learning_rate, mlp_batch_size=h.mlp_batch_size,
        mlp_max_epochs=h.mlp_max_epochs, mlp_patience=h.mlp_patience,
        mlp_val_fraction=h.mlp_val_fraction, stacking_folds=h.stacking_folds,
        random_state=h.random_state)
    return {"BN": build_bn_pipeline(cfg), "MLP": mlp,
            "simple_average": average, "learned_stack": h}


def run(pilot=False) -> pd.DataFrame:
    cfg = load_config(PHASE2_CONFIG_PATH)
    if pilot:
        cfg = copy.deepcopy(cfg)
        cfg["models"]["mlp"].update({"max_epochs": 10, "patience": 3})
        cfg["cv"]["stacking_folds"] = 2
    X, y = get_xy(load_all_cohorts(config=cfg))
    cv = StratifiedKFold(2 if pilot else cfg["cv"]["outer_folds"], shuffle=True,
                         random_state=cfg["project"]["seed"])
    scoring = {"roc_auc": "roc_auc", "neg_log_loss": "neg_log_loss"}
    rows = []
    for suite, variants in (("mlp", mlp_variants(cfg)), ("integration", integration_variants(cfg))):
        for name, estimator in variants.items():
            scores = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
            rows.append({"suite": suite, "variant": name,
                         "roc_auc_mean": scores["test_roc_auc"].mean(),
                         "roc_auc_sd": scores["test_roc_auc"].std(ddof=1),
                         "log_loss_mean": -scores["test_neg_log_loss"].mean(),
                         "fit_seconds": scores["fit_time"].sum()})
            print(rows[-1], flush=True)
    out = resolve("results/phase2/pilot/metrics" if pilot else cfg["paths"]["metrics"])
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "ablations.csv", index=False)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--pilot", action="store_true")
    run(parser.parse_args().pilot)
