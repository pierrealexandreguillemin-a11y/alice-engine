"""ML trainers — CatBoost/XGBoost/LightGBM (ISO 5055/42001/24029)."""

from __future__ import annotations

import gc  # noqa: E401
import logging
import time
from typing import Any

import pandas as pd  # noqa: TCH002 — used at runtime, not just type-checking

logger = logging.getLogger(__name__)

# fmt: off
CATEGORICAL_FEATURES = ["type_competition", "division", "ligue_code", "jour_semaine"]
CATBOOST_CAT_FEATURES = ["type_competition", "division", "ligue_code", "blanc_titre",
                         "noir_titre", "jour_semaine", "zone_enjeu_dom"]
ADVANCED_CAT_FEATURES = [
    "forme_tendance_blanc", "forme_tendance_noir", "couleur_preferee_blanc",
    "couleur_preferee_noir", "data_quality_blanc", "data_quality_noir", "zone_enjeu_ext",
    "elo_trajectory_blanc", "elo_trajectory_noir", "pressure_type_blanc",
    "pressure_type_noir", "categorie_blanc", "categorie_noir", "elo_type_blanc",
    "elo_type_noir", "phase_saison", "regularite_blanc", "regularite_noir",
    "role_type_blanc", "role_type_noir"]
BOOL_FEATURES = [
    "joueur_fantome_blanc", "joueur_fantome_noir", "ffe_multi_equipe_blanc",
    "ffe_multi_equipe_noir", "est_dans_noyau_blanc", "est_dans_noyau_noir",
    "match_important", "renforce_fin_saison_dom", "renforce_fin_saison_ext"]
# fmt: on
LABEL_COLUMN = "resultat_blanc"
AUC_FLOOR = 0.70
MODEL_EXTENSIONS = {"CatBoost": ".cbm", "XGBoost": ".ubj", "LightGBM": ".txt"}


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
    y = (df[LABEL_COLUMN] == 1.0).astype(int)
    for col in BOOL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    X = df.select_dtypes(include=["int64", "int32", "float64", "float32"])
    # Drop target + any outcome column (leakage guard)
    drop = [c for c in X.columns if c in (LABEL_COLUMN, "resultat_noir") or "resultat" in c.lower()]
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
        "global": {"random_seed": 42, "early_stopping_rounds": 50, "eval_metric": "auc"},
        "catboost": {
            "iterations": 1000, "learning_rate": 0.03, "depth": 6,
            "l2_leaf_reg": 3, "min_data_in_leaf": 20, "thread_count": 4,
            "task_type": "GPU" if gpu else "CPU",
            "random_seed": 42, "verbose": 100, "early_stopping_rounds": 50,
        },
        "xgboost": {
            "n_estimators": 1000, "learning_rate": 0.03, "max_depth": 6,
            "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 1,
            **xgb_gpu, "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 50, "verbosity": 1,
        },
        "lightgbm": {
            "n_estimators": 1000, "learning_rate": 0.03, "num_leaves": 63,
            "max_depth": -1, "reg_lambda": 1.0, "reg_alpha": 0.0,
            "min_child_samples": 20, "n_jobs": 4, "random_state": 42,
            "early_stopping_rounds": 50, "verbose": -1,
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

    y_proba = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_proba >= 0.5).astype(np.int64)
    metrics = compute_validation_metrics(y_valid.values, y_pred, y_proba)
    metrics["train_time_s"] = train_time
    importance = (
        dict(zip(X_valid.columns, model.feature_importances_, strict=False))
        if hasattr(model, "feature_importances_")
        else {}
    )
    return {"model": model, "metrics": metrics, "importance": importance}


def _fail_result() -> dict:
    """Sentinel result for a model that failed to train (I1)."""
    return {"model": None, "metrics": {}, "importance": {}}


def _train_catboost(X_train: Any, y_train: Any, X_valid: Any, y_valid: Any, params: dict) -> dict:
    """Train CatBoost — no cat_features since data is label-encoded (I1)."""
    try:
        from catboost import CatBoostClassifier  # noqa: PLC0415

        cb = CatBoostClassifier(**params, eval_metric="AUC")
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

        xgb = XGBClassifier(**params, eval_metric="auc")
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
    """Train LightGBM with GPU attempt + CPU fallback (I1)."""
    try:
        import lightgbm as lgb_lib  # noqa: PLC0415
        from lightgbm import LGBMClassifier  # noqa: PLC0415

        lgb_p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        es = params.get("early_stopping_rounds", 50)
        cbs = [lgb_lib.early_stopping(es), lgb_lib.log_evaluation(100)]
        # Try GPU, auto-fallback to CPU if LightGBM not compiled with GPU
        for device in ["gpu", "cpu"] if _has_gpu() else ["cpu"]:
            lgb_p["device"] = device
            try:
                lgbm = LGBMClassifier(**lgb_p)
                t0 = time.time()
                lgbm.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric="auc",
                    callbacks=cbs,
                )
                result = _eval_model(lgbm, X_valid, y_valid, time.time() - t0)
                del lgbm
                gc.collect()
                return result
            except Exception:
                if device == "gpu":
                    logger.warning("LightGBM GPU failed — falling back to CPU")
                else:
                    raise
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
    """Compute test_auc, test_accuracy, test_f1 for each model. Mutates results."""
    import numpy as np  # noqa: PLC0415
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: PLC0415

    for _name, r in results.items():
        if r["model"] is None:
            continue
        y_proba = r["model"].predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(np.int64)
        r["metrics"]["test_auc"] = float(roc_auc_score(y_test, y_proba))
        r["metrics"]["test_accuracy"] = float(accuracy_score(y_test, y_pred))
        r["metrics"]["test_f1"] = float(f1_score(y_test, y_pred, zero_division=0))


def check_quality_gates(results: dict, champion_auc: float | None = None) -> dict:
    """ISO 42001: AUC floor + relative degradation (2%) gate."""
    candidates = [(n, r) for n, r in results.items() if r["model"] is not None]
    if not candidates:
        return {"passed": False, "reason": "All models failed to train"}
    best_name, best_r = max(candidates, key=lambda x: x[1]["metrics"].get("test_auc", 0.0))
    best_auc = best_r["metrics"].get("test_auc", 0.0)
    if best_auc < AUC_FLOOR:
        return {"passed": False, "reason": f"AUC {best_auc:.4f} < {AUC_FLOOR}"}
    if champion_auc and champion_auc > 0:
        drop_pct = (champion_auc - best_auc) / champion_auc * 100
        if drop_pct > 2.0:
            return {"passed": False, "reason": f"Degradation {drop_pct:.1f}% > 2.0%"}
    return {"passed": True, "best_model": best_name, "best_auc": best_auc}
