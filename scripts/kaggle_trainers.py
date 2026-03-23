"""ML trainers — CatBoost/XGBoost/LightGBM (ISO 5055/42001/24029)."""

from __future__ import annotations

import gc  # noqa: E401
import logging
import time
from typing import Any

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
    import numpy as np  # noqa: PLC0415
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

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
    from scripts.features.helpers import FORFAIT_RESULT  # noqa: PLC0415

    df = df[df[LABEL_COLUMN] != FORFAIT_RESULT].copy()  # exclude forfeits
    TARGET_MAP = {0.0: 0, 0.5: 1, 1.0: 2}  # loss=0, draw=1, win=2
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
    """Auto-detect GPU and set optimal hyperparameters for Kaggle."""
    gpu = _has_gpu()
    logger.info("GPU detected: %s", gpu)
    # fmt: off
    xgb_gpu = {"tree_method": "hist", "device": "cpu"}
    if gpu:
        try:
            from xgboost import __version__ as xv  # noqa: PLC0415
            xgb_gpu = {"tree_method": "hist", "device": "cuda"} if int(xv.split(".")[0]) >= 2 else {"tree_method": "gpu_hist"}
        except Exception:
            xgb_gpu = {"tree_method": "gpu_hist"}
    # V8v3: lower LR, shallower trees, stronger regularization (V8v2 diverged)
    return {
        "global": {"random_seed": 42, "early_stopping_rounds": 200, "eval_metric": "multi_logloss"},
        "catboost": {
            "iterations": 5000, "depth": 6, "border_count": 254,
            "learning_rate": 0.03, "l2_leaf_reg": 5, "min_data_in_leaf": 50,
            "thread_count": 4, "task_type": "GPU" if gpu else "CPU",
            "use_best_model": True, "loss_function": "MultiClass",
            "random_seed": 42, "verbose": 200, "early_stopping_rounds": 200,
        },
        "xgboost": {
            "n_estimators": 5000, "max_depth": 6, "learning_rate": 0.03,
            "objective": "multi:softprob", "num_class": 3,
            "reg_lambda": 3.0, "reg_alpha": 0.1, "min_child_weight": 10,
            "subsample": 0.8, "colsample_bytree": 0.8,
            **xgb_gpu, "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 200, "verbosity": 1,
        },
        "lightgbm": {
            "n_estimators": 5000, "num_leaves": 63, "max_depth": 6,
            "learning_rate": 0.03, "objective": "multiclass", "num_class": 3,
            "reg_lambda": 3.0, "reg_alpha": 0.1, "min_child_samples": 50,
            "subsample": 0.8, "colsample_bytree": 0.8,
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
    import numpy as np  # noqa: PLC0415

    from scripts.kaggle_diagnostics import _compute_metrics  # noqa: PLC0415
    from scripts.kaggle_metrics import predict_with_init  # noqa: PLC0415

    y_proba = predict_with_init(model, X_valid, init_scores_valid)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = _compute_metrics(y_valid.values, y_pred, y_proba)
    metrics["train_time_s"] = train_time
    importance = (
        dict(zip(X_valid.columns, model.feature_importances_, strict=False))
        if hasattr(model, "feature_importances_")
        else {}
    )
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
    """Train XGBoost with residual learning via base_margin."""
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415

        xgb = XGBClassifier(**params, eval_metric="mlogloss")
        fit_kw: dict = {"eval_set": [(X_valid, y_valid)], "verbose": 100}
        # XGBClassifier.fit() has no base_margin for eval_set; training still benefits
        if init_scores_train is not None:
            fit_kw["base_margin"] = init_scores_train  # (n, 3) for XGBoost >= 2.0
        t0 = time.time()
        xgb.fit(X_train, y_train, **fit_kw)
        result = _eval_model(xgb, X_valid, y_valid, time.time() - t0, init_scores_valid)
        del xgb
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
) -> dict:
    """CatBoost -> gc -> XGBoost -> gc -> LightGBM. Sequential memory management."""
    trainers = [
        ("CatBoost", _train_catboost),
        ("XGBoost", _train_xgboost),
        ("LightGBM", _train_lightgbm),
    ]
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
        gc.collect()
    return results


# Re-exported from kaggle_metrics for backward compatibility (ISO 5055 <300 lines)
from scripts.kaggle_metrics import check_quality_gates, evaluate_on_test  # noqa: F401, E402
