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
    # fmt: on
    # fmt: off
    return {
        "global": {"random_seed": 42, "early_stopping_rounds": 100, "eval_metric": "multi_logloss"},
        "catboost": {
            "iterations": 3000, "depth": 8, "border_count": 254,
            "l2_leaf_reg": 3, "min_data_in_leaf": 20, "thread_count": 4,
            "task_type": "GPU" if gpu else "CPU", "use_best_model": True,
            "loss_function": "MultiClass",
            "random_seed": 42, "verbose": 100, "early_stopping_rounds": 100,
        },
        "xgboost": {
            "n_estimators": 3000, "max_depth": 8,
            "objective": "multi:softprob", "num_class": 3,
            "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 1,
            **xgb_gpu, "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 100, "verbosity": 1,
        },
        "lightgbm": {
            "n_estimators": 3000, "num_leaves": 255,
            "objective": "multiclass", "num_class": 3,
            "max_depth": -1, "reg_lambda": 1.0, "reg_alpha": 0.0,
            "min_child_samples": 20, "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 100, "verbose": -1,
        },
    }
    # fmt: on


def compute_validation_metrics(y_true: Any, y_pred: Any, y_proba: Any) -> dict:
    """Delegate to kaggle_diagnostics (moved for ISO 5055 line limit)."""
    from scripts.kaggle_diagnostics import _compute_metrics  # noqa: PLC0415

    return _compute_metrics(y_true, y_pred, y_proba)


def _eval_model(model: Any, X_valid: Any, y_valid: Any, train_time: float) -> dict:
    """Evaluate model on validation, return {model, metrics, importance}."""
    import numpy as np  # noqa: PLC0415

    y_proba = model.predict_proba(X_valid)  # (n, 3)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = compute_validation_metrics(y_valid.values, y_pred, y_proba)
    metrics["train_time_s"] = train_time
    importance = (
        dict(zip(X_valid.columns, model.feature_importances_, strict=False))
        if hasattr(model, "feature_importances_")
        else {}
    )
    return {"model": model, "metrics": metrics, "importance": importance}


def _fail_result() -> dict:
    return {"model": None, "metrics": {}, "importance": {}}


def _train_catboost(X_train: Any, y_train: Any, X_valid: Any, y_valid: Any, params: dict) -> dict:
    """Train CatBoost — no cat_features since data is label-encoded (I1)."""
    try:
        from catboost import CatBoostClassifier  # noqa: PLC0415

        cb = CatBoostClassifier(**params, eval_metric="MultiClass")
        t0 = time.time()
        cb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
        result = _eval_model(cb, X_valid, y_valid, time.time() - t0)
        del cb
        gc.collect()
        return result
    except Exception:
        logger.exception("CatBoost training failed")
        return _fail_result()


def _train_xgboost(X_train: Any, y_train: Any, X_valid: Any, y_valid: Any, params: dict) -> dict:
    """Train XGBoost with partial-failure handling (I1)."""
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415

        xgb = XGBClassifier(**params, eval_metric="mlogloss")
        t0 = time.time()
        xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
        result = _eval_model(xgb, X_valid, y_valid, time.time() - t0)
        del xgb
        gc.collect()
        return result
    except Exception:
        logger.exception("XGBoost training failed")
        return _fail_result()


def _train_lightgbm(X_train: Any, y_train: Any, X_valid: Any, y_valid: Any, params: dict) -> dict:
    """Train LightGBM CPU only (GPU requires special OpenCL build, not on Kaggle)."""
    try:
        import lightgbm as lgb_lib  # noqa: PLC0415
        from lightgbm import LGBMClassifier  # noqa: PLC0415

        lgb_p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        lgb_p["device"] = "cpu"
        es = params.get("early_stopping_rounds", 50)
        cbs = [lgb_lib.early_stopping(es), lgb_lib.log_evaluation(100)]
        lgbm = LGBMClassifier(**lgb_p)
        t0 = time.time()
        lgbm.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            callbacks=cbs,
        )
        result = _eval_model(lgbm, X_valid, y_valid, time.time() - t0)
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
) -> dict:
    """CatBoost -> gc -> XGBoost -> gc -> LightGBM. Sequential memory management."""
    results: dict = {}
    results["CatBoost"] = _train_catboost(X_train, y_train, X_valid, y_valid, config["catboost"])
    gc.collect()
    results["XGBoost"] = _train_xgboost(X_train, y_train, X_valid, y_valid, config["xgboost"])
    gc.collect()
    results["LightGBM"] = _train_lightgbm(X_train, y_train, X_valid, y_valid, config["lightgbm"])
    gc.collect()
    return results


def evaluate_on_test(results: dict, X_test: Any, y_test: Any) -> None:
    """Compute test metrics for each model (multiclass). Mutates results."""
    import numpy as np  # noqa: PLC0415
    from sklearn.metrics import accuracy_score, f1_score, log_loss  # noqa: PLC0415

    from scripts.kaggle_metrics import (  # noqa: PLC0415
        compute_expected_score_mae,
        compute_multiclass_brier,
        compute_rps,
    )

    for _name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_test)  # (n, 3)
        y_pred = np.argmax(y_proba, axis=1)
        r["metrics"]["test_log_loss"] = float(log_loss(y_test, y_proba))
        r["metrics"]["test_accuracy"] = float(accuracy_score(y_test, y_pred))
        r["metrics"]["test_f1_macro"] = float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        )
        r["metrics"]["test_rps"] = float(compute_rps(y_test.values, y_proba))
        r["metrics"]["test_brier"] = float(compute_multiclass_brier(y_test.values, y_proba))
        r["metrics"]["test_es_mae"] = float(compute_expected_score_mae(y_test.values, y_proba))


def check_quality_gates(
    results: dict,
    baseline_metrics: dict | None = None,
    champion_ll: float | None = None,
) -> dict:
    """ISO 42001: 8-condition quality gate (baselines + calibration + champion)."""
    from scripts.kaggle_metrics import check_baseline_conditions  # noqa: PLC0415

    candidates = [(n, r) for n, r in results.items() if r["model"] is not None]
    if not candidates:
        return {"passed": False, "reason": "All models failed to train"}
    best_name, best_r = min(candidates, key=lambda x: x[1]["metrics"].get("test_log_loss", 999.0))
    m = best_r["metrics"]
    best_ll = m.get("test_log_loss", 999.0)
    if baseline_metrics:
        reason = check_baseline_conditions(m, baseline_metrics)
        if reason:
            return {"passed": False, "reason": reason}
    for cls in ("loss", "draw", "win"):
        if m.get(f"ece_class_{cls}", 1.0) >= 0.05:
            return {"passed": False, "reason": f"ece_{cls} >= 0.05"}
    if abs(m.get("draw_calibration_bias", 1.0)) >= 0.02:
        return {"passed": False, "reason": "draw_calibration_bias >= 0.02"}
    if champion_ll and champion_ll > 0:
        rise_pct = (best_ll - champion_ll) / champion_ll * 100
        if rise_pct > 5.0:
            return {"passed": False, "reason": f"Degradation {rise_pct:.1f}% > 5.0%"}
    return {"passed": True, "best_model": best_name, "best_log_loss": best_ll}
