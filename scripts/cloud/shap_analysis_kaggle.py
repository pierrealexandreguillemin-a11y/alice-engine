"""SHAP + Permutation Importance — ALICE V8 models (ISO 25059/5055).

Loads saved v10 models + FE parquets. Computes CatBoost native SHAP,
permutation importance on all 3 models, saves concordance matrix.
CPU-only kernel — no GPU needed.

Bugs fixed (audit 2026-03-25):
- CatBoost SHAP multiclass shape: auto-detect axis order
- LightGBM Booster vs LGBMClassifier: use isinstance, not class name
- Feature prep: reuse prepare_features from kaggle_trainers (exact same as training)
- Permutation importance: subsample 20K for tractable runtime on CPU
- RAM: don't load train.parquet (only needed for draw_lookup, which is in models dataset)
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working")) / "shap_results"
PERM_SUBSAMPLE = 20_000  # subsample test for permutation (221K×5 repeats too slow on CPU)


def _setup_imports() -> None:
    """Add alice-code to sys.path."""
    candidates = [
        Path("/kaggle/input/alice-code"),
        Path("/kaggle/input/datasets/pguillemin/alice-code"),
    ]
    found = next((c for c in candidates if c.exists()), None)
    if found:
        sys.path.insert(0, str(found))
        logger.info("sys.path += %s", found)


def _find_parquet_dir() -> Path:
    """Locate FE parquets from alice-fe-v8 kernel output."""
    candidates = [
        Path("/kaggle/input/notebooks/pguillemin/alice-fe-v8/features"),
        Path("/kaggle/input/notebooks/pguillemin/alice-fe-v8"),
        Path("/kaggle/input/alice-fe-v8/features"),
    ]
    found = next((c for c in candidates if (c / "test.parquet").exists()), None)
    if not found:
        msg = f"FE parquets not found in {candidates}"
        raise FileNotFoundError(msg)
    logger.info("Features dir: %s", found)
    return found


def _find_models_dir() -> Path:
    """Locate v10 model files."""
    candidates = [
        Path("/kaggle/input/alice-models-v10"),
        Path("/kaggle/input/datasets/pguillemin/alice-models-v10"),
    ]
    found = next((c for c in candidates if (c / "CatBoost.cbm").exists()), None)
    if not found:
        msg = f"Models not found in {candidates}"
        raise FileNotFoundError(msg)
    logger.info("Models dir: %s", found)
    return found


def _predict_with_init_safe(model: object, X: pd.DataFrame, init_scores: np.ndarray) -> np.ndarray:
    """Predict probas with init_scores. Handles CatBoost, XGBoost Booster, LightGBM Booster.

    Fixes naming collision: both XGBoost and LightGBM have a class called 'Booster'.
    Uses isinstance checks instead of class name strings.
    """
    import catboost  # noqa: PLC0415
    import lightgbm as lgb  # noqa: PLC0415
    import xgboost as xgb  # noqa: PLC0415

    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp_s = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    if isinstance(model, catboost.CatBoostClassifier):
        raw = np.asarray(model.predict(X, prediction_type="RawFormulaVal"))
        return _softmax(raw + init_scores)

    if isinstance(model, xgb.Booster):
        dm = xgb.DMatrix(X)
        dm.set_base_margin(init_scores.ravel())
        return np.asarray(model.predict(dm)).reshape(-1, 3)

    if isinstance(model, lgb.Booster):
        # LightGBM Booster.predict(raw_score=True) returns raw margin (no init_scores)
        raw = np.asarray(model.predict(X, raw_score=True)).reshape(-1, 3)
        return _softmax(raw + init_scores)

    msg = f"Unsupported model type: {type(model)}"
    raise TypeError(msg)


def _catboost_shap(
    model_dir: Path,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    init_scores: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """CatBoost native SHAP — fast on symmetric trees.

    Shape auto-detection: CatBoost multiclass SHAP can be either
    (n_samples, n_features+1, n_classes) or (n_samples, n_classes, n_features+1).
    We detect based on which dimension matches n_features+1.
    """
    from catboost import CatBoostClassifier, Pool  # noqa: PLC0415

    model = CatBoostClassifier()
    model.load_model(str(model_dir / "CatBoost.cbm"))
    pool = Pool(X_test, label=y_test, baseline=init_scores)

    t0 = time.time()
    shap_vals = model.get_feature_importance(type="ShapValues", data=pool)
    elapsed = time.time() - t0
    logger.info("CatBoost SHAP: %.1fs, shape=%s", elapsed, shap_vals.shape)

    n_feat = len(feature_names)
    if shap_vals.ndim == 3:
        # Auto-detect axis order
        if shap_vals.shape[1] == n_feat + 1:
            # (n_samples, n_features+1, n_classes)
            mean_shap = np.abs(shap_vals[:, :n_feat, :]).mean(axis=0).sum(axis=1)
            logger.info("SHAP shape: (samples, features+1, classes)")
        elif shap_vals.shape[2] == n_feat + 1:
            # (n_samples, n_classes, n_features+1)
            mean_shap = np.abs(shap_vals[:, :, :n_feat]).mean(axis=0).sum(axis=0)
            logger.info("SHAP shape: (samples, classes, features+1)")
        else:
            msg = f"Unexpected SHAP shape {shap_vals.shape} for {n_feat} features"
            raise ValueError(msg)
    else:
        # 2D: binary or single-class fallback
        mean_shap = np.abs(shap_vals[:, :n_feat]).mean(axis=0)
        logger.info("SHAP shape: 2D (samples, features+1)")

    result = pd.DataFrame({"feature": feature_names, "catboost_shap": mean_shap})
    result = result.sort_values("catboost_shap", ascending=False).reset_index(drop=True)
    del model
    gc.collect()
    return result


def _permutation_importance_all(
    model_dir: Path,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    init_scores: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Permutation importance on all 3 models. Subsampled for tractable runtime."""
    from sklearn.inspection import permutation_importance  # noqa: PLC0415
    from sklearn.metrics import log_loss  # noqa: PLC0415

    # Subsample for runtime (221K × 5 repeats × 3 models = too slow on CPU)
    n = min(PERM_SUBSAMPLE, len(X_test))
    rng = np.random.RandomState(42)  # noqa: NPY002
    idx = rng.choice(len(X_test), n, replace=False)
    X_sub = X_test.iloc[idx].reset_index(drop=True)
    y_sub = y_test[idx]
    init_sub = init_scores[idx]
    logger.info("Permutation subsample: %d / %d", n, len(X_test))

    models = {
        "CatBoost": _load_catboost(model_dir),
        "XGBoost": _load_xgboost(model_dir),
        "LightGBM": _load_lightgbm(model_dir),
    }

    result = pd.DataFrame({"feature": feature_names})
    for name, model in models.items():

        def _scorer(estimator: Any, X: Any, y: Any, _init: Any = init_sub) -> float:  # noqa: N803
            probas = _predict_with_init_safe(estimator, X, _init)
            return -log_loss(y, probas, labels=[0, 1, 2])

        t0 = time.time()
        perm = permutation_importance(
            model,
            X_sub,
            y_sub,
            scoring=_scorer,
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        logger.info("%s permutation: %.1fs", name, time.time() - t0)
        result[f"{name}_perm_mean"] = perm.importances_mean
        result[f"{name}_perm_std"] = perm.importances_std
        del model
        gc.collect()

    return result


def _load_catboost(model_dir: Path) -> Any:
    from catboost import CatBoostClassifier  # noqa: PLC0415

    m = CatBoostClassifier()
    m.load_model(str(model_dir / "CatBoost.cbm"))
    return m


def _load_xgboost(model_dir: Path) -> Any:
    import xgboost as xgb  # noqa: PLC0415

    m = xgb.Booster()
    m.load_model(str(model_dir / "XGBoost.ubj"))
    return m


def _load_lightgbm(model_dir: Path) -> Any:
    import lightgbm as lgb  # noqa: PLC0415

    return lgb.Booster(model_file=str(model_dir / "LightGBM.txt"))


def main() -> None:
    """SHAP + permutation importance pipeline."""
    _setup_imports()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feat_dir = _find_parquet_dir()
    model_dir = _find_models_dir()

    # Use EXACT same feature prep as training (kaggle_trainers.prepare_features)
    from scripts.kaggle_trainers import prepare_features  # noqa: PLC0415

    # Only load valid+test (train not needed — draw_lookup is in models dataset)
    # But prepare_features needs train for LabelEncoder fit → load train too
    train = pd.read_parquet(feat_dir / "train.parquet")
    valid = pd.read_parquet(feat_dir / "valid.parquet")
    test = pd.read_parquet(feat_dir / "test.parquet")
    logger.info("Raw: train=%d valid=%d test=%d", len(train), len(valid), len(test))

    _X_train, _y_train, _X_valid, _y_valid, X_test, y_test, _encoders = prepare_features(
        train,
        valid,
        test,
    )
    del train, valid, _X_train, _y_train, _X_valid, _y_valid
    gc.collect()

    feature_names = list(X_test.columns)
    logger.info("Features: %d, Test samples: %d", len(feature_names), len(y_test))

    # Compute init_scores (same as training)
    from scripts.baselines import compute_init_scores_from_features  # noqa: PLC0415

    draw_lookup = pd.read_parquet(
        next(
            p
            for p in [
                model_dir / "draw_rate_lookup.parquet",
            ]
            if p.exists()
        )
    )
    init_scores = compute_init_scores_from_features(X_test, draw_lookup)
    logger.info("init_scores: shape=%s, mean=%s", init_scores.shape, init_scores.mean(axis=0))

    # 1. CatBoost native SHAP (full test set — fast on symmetric trees)
    cb_shap = _catboost_shap(model_dir, X_test, y_test.values, init_scores, feature_names)
    cb_shap.to_csv(OUTPUT_DIR / "catboost_shap_importance.csv", index=False)
    logger.info("CatBoost SHAP top 15:\n%s", cb_shap.head(15).to_string())

    # 2. Permutation importance — all 3 models (subsampled)
    perm = _permutation_importance_all(model_dir, X_test, y_test.values, init_scores, feature_names)
    perm.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)

    # 3. Concordance matrix
    concordance = cb_shap[["feature", "catboost_shap"]].merge(perm, on="feature", how="left")
    perm_cols = [c for c in concordance.columns if c.endswith("_perm_mean")]
    concordance["n_models_perm_positive"] = (concordance[perm_cols] > 0).sum(axis=1)
    concordance["mean_perm"] = concordance[perm_cols].mean(axis=1)
    concordance = concordance.sort_values("mean_perm", ascending=False).reset_index(drop=True)

    concordance["verdict"] = "NOISE"
    concordance.loc[concordance["n_models_perm_positive"] >= 1, "verdict"] = "MARGINAL"
    concordance.loc[concordance["n_models_perm_positive"] >= 2, "verdict"] = "VALIDATED"

    concordance.to_csv(OUTPUT_DIR / "feature_concordance.csv", index=False)

    counts = concordance["verdict"].value_counts()
    logger.info("Feature concordance: %s", counts.to_dict())
    logger.info(
        "VALIDATED (%d):\n%s",
        (concordance["verdict"] == "VALIDATED").sum(),
        concordance[concordance["verdict"] == "VALIDATED"][
            ["feature", "catboost_shap", "mean_perm"]
        ]
        .head(30)
        .to_string(),
    )
    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
