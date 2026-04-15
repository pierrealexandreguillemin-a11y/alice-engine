"""OOF Stack Pipeline — 5-fold CV for meta-learner stacking (ISO 42001/5259).

Supports ALICE_MODEL (xgboost/lightgbm/catboost) and ALICE_FOLDS (0,1,2...) env vars.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
N_FOLDS = 5
ALL_MODELS = {"xgboost": "xgb", "lightgbm": "lgb", "catboost": "cb"}
CLASS_NAMES = ["loss", "draw", "win"]


def _setup_kaggle_imports() -> None:
    """Find alice-code dataset and add to sys.path."""
    candidates = [
        Path("/kaggle/input/alice-code"),
        Path("/kaggle/input/datasets/pguillemin/alice-code"),
    ]
    kaggle_input = next((c for c in candidates if c.exists()), None)
    if kaggle_input:
        sys.path.insert(0, str(kaggle_input))
        logger.info("sys.path += %s", kaggle_input)


def _create_folds(n: int, n_folds: int = 5, seed: int = 42) -> list[tuple]:
    """Create stratification-free fold indices. Returns [(train_idx, val_idx), ...]."""
    rng = np.random.RandomState(seed)  # noqa: NPY002
    indices = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, val_idx))
    return folds


def _calibrate_temperature(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Fit temperature scalar on validation probas (Guo 2017). Returns T."""
    from scipy.optimize import minimize_scalar  # noqa: PLC0415
    from sklearn.metrics import log_loss  # noqa: PLC0415

    def _nll(T: float) -> float:
        logits = np.log(np.clip(y_proba, 1e-7, 1.0))
        scaled = logits / T
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        return float(log_loss(y_true, probs))

    result = minimize_scalar(_nll, bounds=(0.5, 2.0), method="bounded")
    return float(result.x)


def _apply_temperature(y_proba: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to probabilities."""
    logits = np.log(np.clip(y_proba, 1e-7, 1.0))
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    return np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)


def main() -> None:
    """OOF pipeline for stacking. Supports ALICE_MODEL + ALICE_FOLDS env vars."""
    # Model selection: default XGB+LGB, override via ALICE_MODEL=catboost
    model_filter = os.environ.get("ALICE_MODEL")
    if model_filter:
        models = [model_filter.lower()]
    else:
        models = ["xgboost", "lightgbm"]
    model_prefixes = {m: ALL_MODELS[m] for m in models}

    # Fold selection: default all 5, override via ALICE_FOLDS=0,1
    fold_filter = os.environ.get("ALICE_FOLDS")
    if fold_filter:
        fold_indices = [int(x) for x in fold_filter.split(",")]
    else:
        fold_indices = list(range(N_FOLDS))

    logger.info("ALICE Engine — V9 OOF Stack (%s, folds %s)", models, fold_indices)
    _setup_kaggle_imports()

    from scripts.baselines import compute_init_scores_from_features  # noqa: PLC0415

    # --- Load features (same as Training Final) ---
    from scripts.cloud.train_kaggle import _load_features  # noqa: PLC0415
    from scripts.features.draw_priors import build_draw_rate_lookup  # noqa: PLC0415
    from scripts.kaggle_metrics import predict_with_init  # noqa: PLC0415
    from scripts.kaggle_trainers import (  # noqa: PLC0415
        default_hyperparameters,
        prepare_features,
    )

    train_raw, valid_raw, test_raw, features_dir = _load_features()
    logger.info("Loaded: train=%d valid=%d test=%d", len(train_raw), len(valid_raw), len(test_raw))

    # Combine train+valid for OOF (test stays separate)
    combined_raw = pd.concat([train_raw, valid_raw], ignore_index=True)
    logger.info("Combined train+valid: %d rows", len(combined_raw))

    # prepare_features fits encoder on splits[0] = combined_raw. Second arg is
    # a dummy (discarded as _). combined_raw used twice to avoid wasting memory.
    X_combined, y_combined, _, _, X_test, y_test, encoders = prepare_features(
        combined_raw,
        combined_raw,
        test_raw,
    )

    # NaN audit
    for name, df in [("combined", X_combined), ("test", X_test)]:
        dead = [c for c in df.columns if df[c].isna().mean() > 0.99]
        if dead:
            raise ValueError(f"{len(dead)} features >99% NaN on {name}")

    config = default_hyperparameters()

    # Draw lookup from TRAIN ONLY (not combined — avoids leakage via draw rates)
    draw_lookup = build_draw_rate_lookup(train_raw)

    # Init scores for full combined + test
    init_scores_combined = compute_init_scores_from_features(X_combined, draw_lookup)
    init_scores_test = compute_init_scores_from_features(X_test, draw_lookup)

    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(y_combined)
    folds = _create_folds(n, N_FOLDS)

    # Pre-allocate OOF arrays
    n_cols = len(models) * 3
    oof_preds = np.zeros((n, n_cols))
    test_preds_acc = np.zeros((len(y_test), n_cols))
    test_fold_counts = np.zeros(n_cols)

    alpha_map = {
        "xgboost": config["xgboost"].get("init_score_alpha", 0.5),
        "lightgbm": config["lightgbm"].get("init_score_alpha", 0.1),
        "catboost": config["catboost"].get("init_score_alpha", 0.3),
    }

    from scripts.kaggle_trainers import (  # noqa: PLC0415
        _train_catboost,
        _train_lightgbm,
        _train_xgboost,
    )

    train_fns = {
        "xgboost": _train_xgboost,
        "lightgbm": _train_lightgbm,
        "catboost": _train_catboost,
    }

    for fold_k, (train_idx, val_idx) in enumerate(folds):
        if fold_k not in fold_indices:
            continue
        logger.info("=" * 50)
        logger.info(
            "FOLD %d/%d (train=%d, val=%d)", fold_k + 1, N_FOLDS, len(train_idx), len(val_idx)
        )

        X_tr = X_combined.iloc[train_idx]
        y_tr = y_combined.iloc[train_idx]
        X_va = X_combined.iloc[val_idx]
        y_va = y_combined.iloc[val_idx]

        for m_idx, model_name in enumerate(models):
            alpha = alpha_map[model_name]
            init_tr = init_scores_combined[train_idx] * alpha
            init_va = init_scores_combined[val_idx] * alpha
            init_te = init_scores_test * alpha

            logger.info("  Training %s (alpha=%.2f)...", model_name, alpha)
            result = train_fns[model_name](
                X_tr,
                y_tr,
                X_va,
                y_va,
                config[model_name],
                init_scores_train=init_tr,
                init_scores_valid=init_va,
            )
            if result["model"] is None:
                logger.error("  %s fold %d FAILED", model_name, fold_k)
                continue

            # Raw predictions on validation fold
            y_proba_va = predict_with_init(result["model"], X_va, init_va)

            # Temperature calibration on validation fold
            T = _calibrate_temperature(y_va.values, y_proba_va)
            y_proba_va_cal = _apply_temperature(y_proba_va, T)
            logger.info(
                "  %s fold %d: T=%.4f, val_ll=%.6f",
                model_name,
                fold_k,
                T,
                float(
                    np.mean(
                        -np.log(np.clip(y_proba_va_cal[np.arange(len(y_va)), y_va.values], 1e-7, 1))
                    )
                ),
            )

            # Store OOF predictions
            col_start = m_idx * 3
            oof_preds[val_idx, col_start : col_start + 3] = y_proba_va_cal

            # Test predictions (accumulate, average by actual success count later)
            y_proba_te = predict_with_init(result["model"], X_test, init_te)
            y_proba_te_cal = _apply_temperature(y_proba_te, T)
            test_preds_acc[:, col_start : col_start + 3] += y_proba_te_cal
            test_fold_counts[col_start : col_start + 3] += 1

            del result
            gc.collect()

        # Checkpoint after each fold
        _save_oof_checkpoint(oof_preds, y_combined.values, fold_k, out_dir)

    # Average test predictions by actual successful fold count (not N_FOLDS)
    for col_start in range(0, n_cols, 3):
        n_success = test_fold_counts[col_start]
        if n_success > 0:
            test_preds_acc[:, col_start : col_start + 3] /= n_success
        else:
            model_name = models[col_start // 3]
            raise RuntimeError(f"{model_name} has zero successful folds — cannot produce OOF")

    # Build output DataFrames with explicit prefix map
    col_names = []
    for model_name in models:
        prefix = model_prefixes[model_name]
        for cls in CLASS_NAMES:
            col_names.append(f"{prefix}_p_{cls}")

    oof_df = pd.DataFrame(oof_preds, columns=col_names)
    oof_df.insert(0, "y_true", y_combined.values)

    test_df = pd.DataFrame(test_preds_acc, columns=col_names)
    test_df.insert(0, "y_true", y_test.values)

    oof_df.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    test_df.to_parquet(out_dir / "test_predictions_stack.parquet", index=False)
    logger.info("Saved OOF (%d rows) + test (%d rows)", len(oof_df), len(test_df))

    # Quality gates on OOF (sanity)
    _log_oof_metrics(oof_df, test_df)

    logger.info("Done. OOF stack complete.")


def _save_oof_checkpoint(oof: np.ndarray, y: np.ndarray, fold_k: int, out_dir: Path) -> None:
    """Save partial OOF after each fold — survives timeout."""
    import json  # noqa: PLC0415

    np.save(out_dir / "oof_checkpoint.npy", oof)
    with open(out_dir / "oof_status.json", "w") as f:
        json.dump({"last_fold_complete": fold_k, "n_rows": len(y)}, f)
    logger.info("  Checkpoint saved: fold %d complete", fold_k)


def _log_oof_metrics(oof_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Log quality metrics on OOF and test predictions."""
    from sklearn.metrics import log_loss  # noqa: PLC0415

    y_oof = oof_df["y_true"].values
    y_test = test_df["y_true"].values

    # Detect which models are in the DataFrame from column prefixes
    all_prefixes = {"xgb": "XGBoost", "lgb": "LightGBM", "cb": "CatBoost"}
    for prefix, name in all_prefixes.items():
        cols = [f"{prefix}_p_loss", f"{prefix}_p_draw", f"{prefix}_p_win"]
        if cols[0] not in oof_df.columns:
            continue
        oof_probas = oof_df[cols].values
        test_probas = test_df[cols].values
        oof_ll = float(log_loss(y_oof, oof_probas))
        test_ll = float(log_loss(y_test, test_probas))
        logger.info("  %s OOF logloss=%.6f, test logloss=%.6f", name, oof_ll, test_ll)


if __name__ == "__main__":
    main()
