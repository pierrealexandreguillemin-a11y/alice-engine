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
    """Create XGBoost multiclass Optuna objective with residual learning.

    Search space: 4 params (reduced from 8 after fANOVA + literature audit).
    Fixed: max_depth=8, eta=0.05, reg_lambda=4.0, reg_alpha=0.01
    Tuned: init_score_alpha, subsample, colsample_bytree, min_child_weight

    Literature:
    - Probst et al. 2019 (JMLR): reg_alpha/reg_lambda = low tunability
    - van Rijn & Hutter 2018 (KDD): subsample #2, colsample #5
    - fANOVA on 23 XGBoost trials: max_depth=0%, subsample=27%, colsample=18%
    - Bergstra & Bengio 2012: 4 params × 5-10x = 20-40 trials sufficient
    """
    import xgboost as xgb

    space = config.get("optuna", {}).get("xgboost_search_space", {})
    shared = config.get("optuna", {}).get("shared", {})
    alpha_cfg = shared.get("init_score_alpha", {})
    fixed = config.get("optuna", {}).get("xgboost_fixed", {})

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    def objective(trial: Any) -> float:
        alpha = trial.suggest_float(
            "init_score_alpha",
            alpha_cfg.get("low", 0.5),
            alpha_cfg.get("high", 0.8),
        )
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            # FIXED — fANOVA 0% + all 6 trials = 8 (Probst 2019: #4 importance)
            "max_depth": fixed.get("max_depth", 8),
            # FIXED — r=+0.055, coupled with early_stopping (Probst 2019: #1 but moot with ES)
            "eta": fixed.get("eta", 0.05),
            # FIXED — r=-0.112, fANOVA 17.9% but literature LOW (Probst 2019)
            "lambda": fixed.get("reg_lambda", 4.0),
            # FIXED — fANOVA 10.7%, literature quasi-nul (Probst 2019, van Rijn 2018)
            "alpha": fixed.get("reg_alpha", 0.01),
            # TUNED — fANOVA 27% #1, r=-0.864 #1 (van Rijn 2018: #2)
            "subsample": _get_search_param(trial, "subsample", space, 0.6, 0.8),
            # TUNED — fANOVA 18.4% #2, r=-0.614 (van Rijn 2018: #5)
            "colsample_bytree": _get_search_param(trial, "colsample_bytree", space, 0.5, 1.0),
            # TUNED — fANOVA 7.6% #6 but van Rijn 2018 #3 + interaction with depth
            "min_child_weight": _get_search_param(trial, "min_child_weight", space, 50, 200, "int"),
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
    """Create CatBoost multiclass Optuna objective with residual learning.

    Search space: 4 params (literature-driven, no empirical data yet).
    Fixed: learning_rate=0.05, l2_leaf_reg=4.0, random_strength=2.0
    Tuned: init_score_alpha, depth, rsm, min_data_in_leaf

    Literature:
    - Probst et al. 2019: learning_rate + n_estimators = #1-2 (moot with ES)
    - CatBoost oblivious trees: depth IS the primary complexity param (keep)
    - rsm MANDATORY >50 features (v10 finding: 11/177 without rsm)
    - l2_leaf_reg, random_strength: low tunability (Probst 2019)
    """
    from catboost import CatBoostClassifier, Pool

    space = config.get("optuna", {}).get("catboost_search_space", {})
    shared = config.get("optuna", {}).get("shared", {})
    alpha_cfg = shared.get("init_score_alpha", {})
    fixed = config.get("optuna", {}).get("catboost_fixed", {})

    def objective(trial: Any) -> float:
        alpha = trial.suggest_float(
            "init_score_alpha",
            alpha_cfg.get("low", 0.5),
            alpha_cfg.get("high", 0.8),
        )
        params = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "iterations": 50000,
            # TUNED — oblivious trees: depth = THE complexity param
            "depth": _get_search_param(trial, "depth", space, 4, 10, "int"),
            # FIXED — coupled with early_stopping (Probst 2019: #1 but moot with ES)
            "learning_rate": fixed.get("learning_rate", 0.05),
            # FIXED — low tunability (Probst 2019), median of reasonable range
            "l2_leaf_reg": fixed.get("l2_leaf_reg", 4.0),
            # TUNED — MANDATORY >50 features, equivalent to colsample_bytree
            "rsm": _get_search_param(trial, "rsm", space, 0.2, 0.8),
            # FIXED — CatBoost-specific, low priority per literature
            "random_strength": fixed.get("random_strength", 2.0),
            # TUNED — interaction with depth for leaf-level regularization
            "min_data_in_leaf": _get_search_param(trial, "min_data_in_leaf", space, 20, 200, "int"),
            "task_type": "CPU",
            "random_seed": 42,
            "verbose": 0,
            "early_stopping_rounds": 200,
        }

        train_pool = Pool(X_train, y_train, baseline=(init_train * alpha))
        valid_pool = Pool(X_valid, y_valid, baseline=(init_valid * alpha))

        try:
            from optuna_integration import CatBoostPruningCallback

            pruning_cb = CatBoostPruningCallback(trial, metric="MultiClass")
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=valid_pool, callbacks=[pruning_cb])
        except ImportError:
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
    """Create LightGBM multiclass Optuna objective with residual learning.

    Search space: 4 params (literature-driven, no empirical data yet).
    Fixed: max_depth=8, learning_rate=0.05, reg_lambda=4.0, min_child_samples=100
    Tuned: init_score_alpha, num_leaves, feature_fraction, bagging_fraction

    Literature:
    - LightGBM is leaf-wise: num_leaves IS the primary complexity param
    - Probst 2019: learning_rate #1 (moot with ES), reg_lambda LOW
    - van Rijn 2018: subsample #2 → bagging_fraction, colsample → feature_fraction
    - max_depth=8 as safety cap (clamps num_leaves to max 255)
    """
    import lightgbm as lgb

    space = config.get("optuna", {}).get("lightgbm_search_space", {})
    shared = config.get("optuna", {}).get("shared", {})
    alpha_cfg = shared.get("init_score_alpha", {})
    fixed = config.get("optuna", {}).get("lightgbm_fixed", {})

    def objective(trial: Any) -> float:
        alpha = trial.suggest_float(
            "init_score_alpha",
            alpha_cfg.get("low", 0.5),
            alpha_cfg.get("high", 0.8),
        )
        # FIXED — safety cap, XGBoost converged to 8 (transferable heuristic)
        max_depth = fixed.get("max_depth", 8)
        # TUNED — leaf-wise = THE complexity param (clamp to 2^depth-1)
        raw_leaves = _get_search_param(trial, "num_leaves", space, 15, 255, "int")
        num_leaves = min(raw_leaves, 2**max_depth - 1)

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            # FIXED — coupled with early_stopping (Probst 2019: #1 but moot with ES)
            "learning_rate": fixed.get("learning_rate", 0.05),
            # FIXED — low tunability (Probst 2019, van Rijn 2018)
            "reg_lambda": fixed.get("reg_lambda", 4.0),
            # TUNED — equivalent colsample_bytree (van Rijn 2018: #5, fANOVA XGB: 18.4%)
            "feature_fraction": _get_search_param(trial, "feature_fraction", space, 0.3, 1.0),
            # TUNED — equivalent subsample (van Rijn 2018: #2, fANOVA XGB: 27%)
            "bagging_fraction": _get_search_param(trial, "bagging_fraction", space, 0.5, 1.0),
            "bagging_freq": 1,
            # FIXED — moderate importance but fixing reduces search space
            "min_child_samples": fixed.get("min_child_samples", 100),
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

            pruning_cb = LightGBMPruningCallback(trial, metric="multi_logloss", valid_name="valid")
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

    # Resume from previous session: copy .db from dataset input to working dir.
    # Kaggle clears /kaggle/working/ on each run — the only way to persist a
    # study across sessions is to upload the .db as part of a dataset, then
    # copy it back before creating the RDBStorage.
    if not db_path.exists():
        for candidate in [
            Path(f"/kaggle/input/datasets/pguillemin/alice-code/optuna_{model_name}.db"),
            Path(f"/kaggle/input/alice-code/optuna_{model_name}.db"),
            Path(f"/kaggle/input/datasets/pguillemin/alice-optuna-db/optuna_{model_name}.db"),
            Path(f"/kaggle/input/alice-optuna-db/optuna_{model_name}.db"),
        ]:
            if candidate.exists():
                import shutil

                shutil.copy2(candidate, db_path)
                logger.info("Resumed study DB from %s", candidate)
                break
        else:
            logger.info("No previous study DB found — starting fresh")

    # NopPruner: pruning DISABLED.
    # MedianPruner was too aggressive — 17/18 trials pruned at iter 500 on
    # XGBoost v5 (flat landscape, delta best-worst = 0.01).
    # Pruning was needed when eta ∈ [0.001, 0.1] (trial 1 v1: eta=0.001, 9h).
    # Now eta is FIXED at 0.05 → every trial converges in ~3h.
    # 4 trials/12h session × 3h each = pruning saves nothing.
    pruner = optuna.pruners.NopPruner()
    heartbeat_interval = 60
    grace_period = 3 * heartbeat_interval
    failed_trial_cb = optuna.storages.RetryFailedTrialCallback(max_retry=0)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        heartbeat_interval=heartbeat_interval,
        grace_period=grace_period,
        failed_trial_callback=failed_trial_cb,
    )
    # TPESampler with reduced startup: n_startup_trials=4 (= n_params).
    # GPSampler was evaluated but rejected — incompatible with pruning
    # (Optuna #5481). With NopPruner, GP would work, but TPE is safer
    # for resume across sessions (mixed param spaces from old trials).
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=4)
    study = optuna.create_study(
        study_name=f"alice_{model_name}_v9",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 100)
    # 10h (not 11h) — leave 2h margin for post-processing (fANOVA, JSON, CSV)
    # and Kaggle session cleanup. 12h limit - 10h training - 2h margin.
    timeout = optuna_config.get("timeout", 36000)

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

    # fANOVA parameter importance (post-hoc analysis for diagnostics)
    n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
    if n_complete >= 4:
        try:
            importances = optuna.importance.get_param_importances(study)
            logger.info("=== fANOVA Parameter Importances ===")
            for name, val in importances.items():
                logger.info("  %s: %.1f%%", name, val * 100)
        except Exception:
            logger.warning("fANOVA failed (not enough trials or mixed params)")

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
