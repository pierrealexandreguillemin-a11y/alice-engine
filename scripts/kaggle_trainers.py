"""ML trainers — CatBoost/XGBoost/LightGBM (ISO 5055/42001/24029)."""

from __future__ import annotations

import gc  # noqa: E401
import logging
import time
from typing import Any

import numpy as np
import pandas as pd  # noqa: TCH002 — used at runtime, not just type-checking

logger = logging.getLogger(__name__)

from scripts.kaggle_constants import (  # noqa: E402
    ADVANCED_CAT_FEATURES,
    BOOL_FEATURES,
    CATBOOST_CAT_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_COLUMN,
    LEAKY_COLUMNS,
    MODEL_EXTENSIONS,  # noqa: F401 — re-exported for train_kaggle.py
)


def _encode_categoricals(splits: list[pd.DataFrame]) -> dict:
    """Label-encode all categorical columns. Adds UNKNOWN class for inference safety."""
    # fmt: off
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415
    # fmt: on
    all_cat_cols = sorted(
        set(CATEGORICAL_FEATURES) | set(CATBOOST_CAT_FEATURES) | set(ADVANCED_CAT_FEATURES)
    )
    encoders: dict = {}
    train = splits[0]  # Fit on train only (no leakage from valid/test categories)
    for col in all_cat_cols:
        if col not in train.columns:
            continue
        enc = LabelEncoder()
        enc.fit(train[col].astype(str))
        if "UNKNOWN" not in enc.classes_:
            enc.classes_ = np.append(enc.classes_, "UNKNOWN")
        for split in splits:
            # Map unseen labels to UNKNOWN before transform
            split[col] = (
                split[col].astype(str).where(split[col].astype(str).isin(enc.classes_), "UNKNOWN")
            )
            split[col] = enc.transform(split[col])
        encoders[col] = enc
    return encoders


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X and y, keeping numeric + casting bools to int."""
    from scripts.features.helpers import NON_PLAYED  # noqa: PLC0415

    # Exclude non-played games via type_resultat (NOT resultat_blanc)
    if "type_resultat" in df.columns:
        df = df[~df["type_resultat"].isin(NON_PLAYED)].copy()
    TARGET_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 2}  # 2.0=victoire jeunes FFE (J02 §4.1)
    y = df[LABEL_COLUMN].map(TARGET_MAP).astype(int)
    for col in BOOL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    X = df.select_dtypes(include=["int64", "int32", "float64", "float32"])
    drop = [  # target + outcome + leakage columns
        c
        for c in X.columns
        if c in (LABEL_COLUMN, "resultat_noir") or "resultat" in c.lower() or c in LEAKY_COLUMNS
    ]
    X = X.drop(columns=drop, errors="ignore")
    return X, y


def prepare_features(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Label-encode categoricals, split X/y, drop remaining strings."""
    encoders = _encode_categoricals([train, valid, test])
    X_train, y_train = _split_xy(train)
    X_valid, y_valid = _split_xy(valid)
    X_test, y_test = _split_xy(test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, encoders


_GPU_CACHE: bool | None = None


def _has_gpu() -> bool:
    """Detect NVIDIA GPU availability (cached)."""
    global _GPU_CACHE  # noqa: PLW0603
    if _GPU_CACHE is not None:
        return _GPU_CACHE
    try:
        import subprocess  # noqa: PLC0415

        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)  # noqa: S603, S607
        _GPU_CACHE = result.returncode == 0
    except Exception:
        _GPU_CACHE = False
    return _GPU_CACHE


def default_hyperparameters() -> dict:
    """Optimal hyperparameters for Kaggle CPU training (ADR-003: all models CPU)."""
    # fmt: off
    return {
        "global": {
            "random_seed": 42,
            "early_stopping_rounds": 200,
            "eval_metric": "multi_logloss",
            # Init score shrink: reduce Elo prior dominance so features can express corrections.
            # Alpha < 1 = confidence in Elo prior (Ash & Adams 2020, NeurIPS). Not a hack.
            # v15: models converge in 89-133 iters → prior too strong → alpha=0.7
            "init_score_alpha": 0.7,
        },
        "catboost": {
            "iterations": 50000, "depth": 4, "border_count": 128,
            "learning_rate": 0.005, "l2_leaf_reg": 10, "min_data_in_leaf": 200,
            "random_strength": 3, "bagging_temperature": 1, "model_size_reg": 0.5,
            "rsm": 0.3,  # Feature subsampling — MANDATORY >50 features (v10 bug: 11/177 sans rsm)
            "thread_count": 4, "task_type": "CPU",  # rsm incompatible GPU (CatBoost: pairwise only)
            "use_best_model": True, "loss_function": "MultiClass",
            "random_seed": 42, "verbose": 500, "early_stopping_rounds": 200,
        },
        "xgboost": {
            "n_estimators": 50000, "max_depth": 4, "eta": 0.005,
            "objective": "multi:softprob", "num_class": 3,
            "lambda": 10.0, "alpha": 0.5, "min_child_weight": 50,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "tree_method": "hist", "device": "cpu",  # CPU — no GPU needed for tree models
            "nthread": 4, "seed": 42,
            "early_stopping_rounds": 200, "verbosity": 1,
        },
        "lightgbm": {
            "n_estimators": 50000, "num_leaves": 15, "max_depth": 4,
            "learning_rate": 0.003, "objective": "multiclass", "num_class": 3,
            "reg_lambda": 10.0, "reg_alpha": 0.5, "min_child_samples": 200,
            "min_gain_to_split": 0.01, "subsample": 0.7, "colsample_bytree": 0.5,
            "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 200, "verbose": -1,
        },
    }
    # fmt: on


def _eval_model(
    model: Any,
    X_valid: Any,
    y_valid: Any,
    train_time: float,
    init_scores_valid: Any | None = None,
) -> dict:
    """Evaluate model on validation, return {model, metrics, importance}."""
    from scripts.kaggle_diagnostics import _compute_metrics  # noqa: PLC0415
    from scripts.kaggle_metrics import predict_with_init  # noqa: PLC0415

    y_proba = predict_with_init(model, X_valid, init_scores_valid)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = _compute_metrics(y_valid.values, y_pred, y_proba)
    metrics["train_time_s"] = train_time
    if hasattr(model, "feature_importances_"):  # CatBoost, LightGBM
        importance = dict(zip(X_valid.columns, model.feature_importances_, strict=False))
    else:  # xgb.Booster
        importance = model.get_score(importance_type="gain") if hasattr(model, "get_score") else {}
    return {"model": model, "metrics": metrics, "importance": importance}


def _fail_result() -> dict:
    return {"model": None, "metrics": {}, "importance": {}}


def _train_catboost(
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    params: dict,
    init_scores_train: Any | None = None,
    init_scores_valid: Any | None = None,
) -> dict:
    """Train CatBoost with residual learning via Pool baseline."""
    try:
        from catboost import CatBoostClassifier, Pool  # noqa: PLC0415

        cb = CatBoostClassifier(**params, eval_metric="MultiClass")
        train_pool = Pool(X_train, y_train, baseline=init_scores_train)
        valid_pool = Pool(X_valid, y_valid, baseline=init_scores_valid)
        t0 = time.time()
        cb.fit(train_pool, eval_set=valid_pool)
        result = _eval_model(cb, X_valid, y_valid, time.time() - t0, init_scores_valid)
        del cb
        gc.collect()
        return result
    except Exception:
        logger.exception("CatBoost training failed")
        return _fail_result()


def _train_xgboost(
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    params: dict,
    init_scores_train: Any | None = None,
    init_scores_valid: Any | None = None,
) -> dict:
    """Train XGBoost via native xgb.train + DMatrix (sklearn API base_margin broken #5288)."""
    try:
        import xgboost as xgb  # noqa: PLC0415

        p = {k: v for k, v in params.items() if k not in ("n_estimators", "early_stopping_rounds")}
        n_rounds = params.get("n_estimators", 5000)
        es = params.get("early_stopping_rounds", 500)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        if init_scores_train is not None:
            dtrain.set_base_margin(init_scores_train.ravel())
        if init_scores_valid is not None:
            dvalid.set_base_margin(init_scores_valid.ravel())
        t0 = time.time()
        evals_log: dict = {}
        # fmt: off
        bst = xgb.train(p, dtrain, n_rounds, evals=[(dvalid, "val")],
                        early_stopping_rounds=es, verbose_eval=100, evals_result=evals_log)
        # fmt: on
        # Wrap Booster for sklearn-compatible pipeline (predict_proba, feature_importances_)
        from scripts.kaggle_metrics import XGBWrapper  # noqa: PLC0415

        wrapper = XGBWrapper(bst, X_train.columns, p.get("num_class", 3), evals_result=evals_log)
        result = _eval_model(wrapper, X_valid, y_valid, time.time() - t0, init_scores_valid)
        del bst
        gc.collect()
        return result
    except Exception:
        logger.exception("XGBoost training failed")
        return _fail_result()


def _train_lightgbm(
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    params: dict,
    init_scores_train: Any | None = None,
    init_scores_valid: Any | None = None,
) -> dict:
    """Train LightGBM with residual learning via init_score."""
    try:
        import lightgbm as lgb_lib  # noqa: PLC0415
        from lightgbm import LGBMClassifier  # noqa: PLC0415

        lgb_p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        lgb_p["device"] = "cpu"
        es = params.get("early_stopping_rounds", 50)
        cbs = [lgb_lib.early_stopping(es), lgb_lib.log_evaluation(100)]
        lgbm = LGBMClassifier(**lgb_p)
        fit_kw: dict = {
            "eval_set": [(X_valid, y_valid)],
            "eval_metric": "multi_logloss",
            "callbacks": cbs,
        }
        if init_scores_train is not None:
            fit_kw["init_score"] = init_scores_train
        if init_scores_valid is not None:
            fit_kw["eval_init_score"] = [init_scores_valid]
        t0 = time.time()
        lgbm.fit(X_train, y_train, **fit_kw)
        result = _eval_model(lgbm, X_valid, y_valid, time.time() - t0, init_scores_valid)
        del lgbm
        gc.collect()
        return result
    except Exception:
        logger.exception("LightGBM training failed")
        return _fail_result()


def train_all_sequential(
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    config: dict,
    init_scores_train: Any | None = None,
    init_scores_valid: Any | None = None,
    checkpoint_dir: Any | None = None,
    encoders: Any = None,
    model_extensions: dict | None = None,
    model_filter: str | None = None,
) -> dict:
    """Train models sequentially. Checkpoint after each. Filter with ALICE_MODEL env var."""
    all_trainers = [("CatBoost", _train_catboost), ("XGBoost", _train_xgboost), ("LightGBM", _train_lightgbm)]  # fmt: skip
    if model_filter:
        trainers = [(n, f) for n, f in all_trainers if n.lower() == model_filter.lower()]
        if not trainers:
            logger.error("Unknown model filter: %s", model_filter)
            trainers = all_trainers
    else:
        trainers = all_trainers
    logger.info("Training %d model(s): %s", len(trainers), [n for n, _ in trainers])
    results: dict = {}
    for name, fn in trainers:
        results[name] = fn(
            X_train,
            y_train,
            X_valid,
            y_valid,
            config[name.lower()],
            init_scores_train,
            init_scores_valid,
        )
        if checkpoint_dir and results[name]["model"] is not None:
            _checkpoint_model(name, results[name], checkpoint_dir, model_extensions)
        gc.collect()
    return results


def _checkpoint_model(
    name: str, result: dict, out_dir: Any, model_extensions: dict | None = None
) -> None:
    """Save a single model to disk immediately after training."""
    from pathlib import Path  # noqa: PLC0415

    out_dir = Path(out_dir)
    model = result["model"]
    ext = (model_extensions or {}).get(name, ".bin")
    path = out_dir / f"{name.lower()}_checkpoint{ext}"
    try:
        model.save_model(str(path))
        logger.info("Checkpoint: %s saved to %s", name, path)
    except Exception as e:
        logger.warning("Checkpoint failed for %s: %s", name, e)
