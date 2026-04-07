"""Optuna V9 multiclass hyperparameter tuning — Kaggle CPU kernel.

Document ID: ALICE-OPTUNA-V9
Version: 1.0.0

Bayesian optimization with TPESampler on multiclass (loss/draw/win)
with residual learning (init_score_alpha in search space).

ISO Compliance:
- ISO/IEC 42001:2023 — AI Management (traceability via SQLite study)
- ISO/IEC 25059:2023 — AI Quality (baseline comparison gate)
- ISO/IEC 5055:2021 — Code Quality (SRP, <300 lines)

References
----------
- Spec: docs/superpowers/specs/2026-04-07-optuna-v9-pipeline-design.md
- XGBoost: https://xgboost.readthedocs.io/en/stable/parameter.html
- CatBoost: https://catboost.ai/docs/en/concepts/parameter-tuning
- LightGBM: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ELO_BASELINE_LOGLOSS = 0.9766


def _get_search_param(
    trial: Any,
    name: str,
    space: dict,
    default_low: float,
    default_high: float,
    param_type: str = "float",
    log: bool = False,
) -> Any:
    """Suggest a parameter from the search space config or defaults."""
    cfg = space.get(name, {})
    low = cfg.get("low", default_low)
    high = cfg.get("high", default_high)
    use_log = cfg.get("log", log)
    if param_type == "int":
        return trial.suggest_int(name, int(low), int(high))
    return trial.suggest_float(name, low, high, log=use_log)


def create_xgboost_objective_v9(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
    config: dict,
) -> Any:
    """Create XGBoost multiclass Optuna objective with residual learning."""
    import xgboost as xgb

    space = config.get("optuna", {}).get("xgboost_search_space", {})
    shared = config.get("optuna", {}).get("shared", {})
    alpha_cfg = shared.get("init_score_alpha", {})

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    def objective(trial: Any) -> float:
        alpha = trial.suggest_float(
            "init_score_alpha",
            alpha_cfg.get("low", 0.3),
            alpha_cfg.get("high", 0.8),
        )
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": _get_search_param(trial, "max_depth", space, 3, 8, "int"),
            "eta": _get_search_param(trial, "eta", space, 0.001, 0.1, "float", log=True),
            "lambda": _get_search_param(trial, "reg_lambda", space, 0.1, 20.0, log=True),
            "alpha": _get_search_param(trial, "reg_alpha", space, 0.001, 1.0, log=True),
            "subsample": _get_search_param(trial, "subsample", space, 0.5, 1.0),
            "colsample_bytree": _get_search_param(trial, "colsample_bytree", space, 0.3, 1.0),
            "min_child_weight": _get_search_param(trial, "min_child_weight", space, 20, 200, "int"),
            "tree_method": "hist",
            "seed": 42,
            "nthread": int(os.environ.get("ALICE_NTHREAD", "4")),
            "verbosity": 0,
        }

        dtrain.set_base_margin((init_train * alpha).ravel())
        dvalid.set_base_margin((init_valid * alpha).ravel())

        try:
            from optuna_integration import XGBoostPruningCallback

            pruning_cb = XGBoostPruningCallback(trial, "valid-mlogloss")
            callbacks = [
                xgb.callback.EarlyStopping(rounds=200, save_best=True),
                pruning_cb,
            ]
        except ImportError:
            callbacks = [xgb.callback.EarlyStopping(rounds=200, save_best=True)]

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=50000,
            evals=[(dvalid, "valid")],
            callbacks=callbacks,
            verbose_eval=False,
        )
        return booster.best_score

    return objective


def create_catboost_objective_v9(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
    config: dict,
) -> Any:
    """Create CatBoost multiclass Optuna objective with residual learning."""
    from catboost import CatBoostClassifier, Pool

    space = config.get("optuna", {}).get("catboost_search_space", {})
    shared = config.get("optuna", {}).get("shared", {})
    alpha_cfg = shared.get("init_score_alpha", {})

    def objective(trial: Any) -> float:
        alpha = trial.suggest_float(
            "init_score_alpha",
            alpha_cfg.get("low", 0.3),
            alpha_cfg.get("high", 0.8),
        )
        params = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "iterations": 50000,
            "depth": _get_search_param(trial, "depth", space, 4, 10, "int"),
            "learning_rate": _get_search_param(trial, "learning_rate", space, 0.001, 0.1, log=True),
            "l2_leaf_reg": _get_search_param(trial, "l2_leaf_reg", space, 0.1, 20.0, log=True),
            "rsm": _get_search_param(trial, "rsm", space, 0.2, 0.8),
            "random_strength": _get_search_param(trial, "random_strength", space, 0.5, 5.0),
            "min_data_in_leaf": _get_search_param(trial, "min_data_in_leaf", space, 20, 200, "int"),
            "task_type": "CPU",
            "random_seed": 42,
            "verbose": 0,
            "early_stopping_rounds": 200,
        }

        train_pool = Pool(X_train, y_train, baseline=(init_train * alpha))
        valid_pool = Pool(X_valid, y_valid, baseline=(init_valid * alpha))

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool)

        # Discover metric key (unverified for multiclass eval_metric)
        val_scores = model.get_best_score().get("validation", {})
        if "MultiClass" in val_scores:
            return val_scores["MultiClass"]
        # Fallback: try first available key
        if val_scores:
            key = next(iter(val_scores))
            logger.warning("CatBoost metric key fallback: %s (expected MultiClass)", key)
            return val_scores[key]
        msg = f"No validation scores found. Keys: {model.get_best_score()}"
        raise ValueError(msg)

    return objective


def create_lightgbm_objective_v9(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    init_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    init_valid: np.ndarray,
    config: dict,
) -> Any:
    """Create LightGBM multiclass Optuna objective with residual learning."""
    import lightgbm as lgb

    space = config.get("optuna", {}).get("lightgbm_search_space", {})
    shared = config.get("optuna", {}).get("shared", {})
    alpha_cfg = shared.get("init_score_alpha", {})

    def objective(trial: Any) -> float:
        alpha = trial.suggest_float(
            "init_score_alpha",
            alpha_cfg.get("low", 0.3),
            alpha_cfg.get("high", 0.8),
        )
        max_depth = _get_search_param(trial, "max_depth", space, 3, 8, "int")
        raw_leaves = _get_search_param(trial, "num_leaves", space, 7, 63, "int")
        num_leaves = min(raw_leaves, 2**max_depth - 1)

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "learning_rate": _get_search_param(trial, "learning_rate", space, 0.001, 0.1, log=True),
            "reg_lambda": _get_search_param(trial, "reg_lambda", space, 0.1, 20.0, log=True),
            "feature_fraction": _get_search_param(trial, "feature_fraction", space, 0.3, 1.0),
            "bagging_fraction": _get_search_param(trial, "bagging_fraction", space, 0.5, 1.0),
            "bagging_freq": 1,
            "min_child_samples": _get_search_param(
                trial, "min_child_samples", space, 20, 200, "int"
            ),
            "seed": 42,
            "verbose": -1,
            "n_jobs": int(os.environ.get("ALICE_NTHREAD", "4")),
        }

        dtrain = lgb.Dataset(X_train, label=y_train, init_score=(init_train * alpha).ravel())
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=dtrain,
            init_score=(init_valid * alpha).ravel(),
        )

        try:
            from optuna_integration import LightGBMPruningCallback

            pruning_cb = LightGBMPruningCallback(trial, metric="multi_logloss")
            callbacks = [lgb.early_stopping(200), lgb.log_evaluation(0), pruning_cb]
        except ImportError:
            callbacks = [lgb.early_stopping(200), lgb.log_evaluation(0)]

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=50000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=callbacks,
        )
        return booster.best_score["valid"]["multi_logloss"]

    return objective


def main() -> None:
    """Run Optuna study for the model specified by ALICE_MODEL env var."""
    import optuna
    import yaml

    from scripts.baselines import compute_init_scores_from_features
    from scripts.cloud.train_kaggle import _load_features, _setup_kaggle_imports
    from scripts.features.draw_priors import build_draw_rate_lookup
    from scripts.kaggle_trainers import prepare_features

    _setup_kaggle_imports()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_name = os.environ.get("ALICE_MODEL", "xgboost")
    out_dir = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path("config/hyperparameters.yaml")
    if not config_path.exists():
        for p in [
            Path("/kaggle/input/alice-code/config/hyperparameters.yaml"),
            Path("/kaggle/input/datasets/pguillemin/alice-code/config/hyperparameters.yaml"),
        ]:
            if p.exists():
                config_path = p
                break
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded from %s", config_path)

    train, valid, test, features_dir = _load_features()
    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(
        train, valid, test
    )
    logger.info("Features: train=%s, valid=%s", X_train.shape, X_valid.shape)

    draw_lookup = build_draw_rate_lookup(train)
    init_train = compute_init_scores_from_features(X_train, draw_lookup)
    init_valid = compute_init_scores_from_features(X_valid, draw_lookup)
    logger.info("Init scores: train=%s, valid=%s", init_train.shape, init_valid.shape)

    objective_factory = {
        "xgboost": create_xgboost_objective_v9,
        "catboost": create_catboost_objective_v9,
        "lightgbm": create_lightgbm_objective_v9,
    }
    if model_name not in objective_factory:
        msg = f"Unknown model: {model_name}. Expected: {list(objective_factory)}"
        raise ValueError(msg)

    objective = objective_factory[model_name](
        X_train, y_train, init_train, X_valid, y_valid, init_valid, config
    )

    db_path = out_dir / f"optuna_{model_name}.db"
    study = optuna.create_study(
        study_name=f"alice_{model_name}_v9",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 100)
    timeout = optuna_config.get("timeout", 39600)

    logger.info(
        "Starting Optuna: model=%s, n_trials=%d, timeout=%ds",
        model_name,
        n_trials,
        timeout,
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    elapsed = time.time() - t0

    logger.info("Optuna complete: %d trials in %.0fs", len(study.trials), elapsed)
    logger.info("Best logloss: %.5f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    gate_pass = study.best_value < ELO_BASELINE_LOGLOSS
    logger.info(
        "GATE: best_logloss %.5f < Elo %.5f = %s",
        study.best_value,
        ELO_BASELINE_LOGLOSS,
        "PASS" if gate_pass else "FAIL",
    )

    best = {
        "model": model_name,
        "best_logloss": study.best_value,
        "best_params": study.best_params,
        "n_trials_completed": len([t for t in study.trials if t.state.name == "COMPLETE"]),
        "n_trials_pruned": len([t for t in study.trials if t.state.name == "PRUNED"]),
        "total_time_s": round(elapsed, 1),
        "gate_pass": gate_pass,
        "elo_baseline_logloss": ELO_BASELINE_LOGLOSS,
    }
    best_path = out_dir / f"best_params_{model_name}.json"
    best_path.write_text(json.dumps(best, indent=2))
    logger.info("Saved: %s", best_path)

    rows = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "value": t.value,
            "state": t.state.name,
            "duration_s": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete and t.datetime_start
                else None
            ),
        }
        row.update(t.params)
        rows.append(row)
    history = pd.DataFrame(rows)
    history_path = out_dir / f"trial_history_{model_name}.csv"
    history.to_csv(history_path, index=False)
    logger.info("Saved: %s", history_path)


if __name__ == "__main__":
    main()
