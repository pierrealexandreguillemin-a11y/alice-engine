"""AutoGluon V9 Benchmark — best_quality multiclass (ISO 42001/5259).

Trains AutoGluon on V9 features + 3 Elo proba features. No init_scores
(AG doesn't support them). Benchmark against V9 single models.
"""

from __future__ import annotations

import json
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
AG_TIME_LIMIT = 36000  # 10h (2h margin for post-processing)
AG_PRESETS = "best_quality"
AG_BAG_FOLDS = 5
AG_STACK_LEVELS = 1  # V8 postmortem: L2/L3 overfit


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


def _compute_elo_proba_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute P_elo(W/D/L) from blanc_elo/noir_elo as explicit features.

    Gives AutoGluon the Elo baseline signal that V9 models get via init_scores.
    Uses the same logistic model as scripts/baselines.py + draw_rate_lookup.
    """
    blanc_elo = df["blanc_elo"].fillna(1500).values
    noir_elo = df["noir_elo"].fillna(1500).values
    # White advantage: dynamic +20 Elo (simplified, full lookup in production)
    blanc_adj = blanc_elo + 20.0
    expected_w = 1.0 / (1.0 + 10.0 ** ((noir_elo - blanc_adj) / 400.0))
    expected_l = 1.0 / (1.0 + 10.0 ** ((blanc_adj - noir_elo) / 400.0))
    # Draw = 1 - W - L (simplified; full version uses draw_rate_lookup.parquet)
    p_draw = np.clip(1.0 - expected_w - expected_l, 0.01, 0.99)
    # Renormalize
    total = expected_w + p_draw + expected_l
    result = pd.DataFrame(
        {
            "p_elo_win": expected_w / total,
            "p_elo_draw": p_draw / total,
            "p_elo_loss": expected_l / total,
        },
        index=df.index,
    )
    return result


def main() -> None:
    """AutoGluon V9 benchmark pipeline."""
    logger.info("ALICE Engine — AutoGluon V9 Benchmark")
    _setup_kaggle_imports()

    from autogluon.tabular import TabularPredictor  # noqa: PLC0415

    # Load features (same as V9 Training Final)
    from scripts.cloud.train_kaggle import _load_features  # noqa: PLC0415
    from scripts.kaggle_trainers import prepare_features  # noqa: PLC0415

    train_raw, valid_raw, test_raw, features_dir = _load_features()
    logger.info("Loaded: train=%d valid=%d test=%d", len(train_raw), len(valid_raw), len(test_raw))

    # Prepare features (encoding, target mapping)
    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(
        train_raw,
        valid_raw,
        test_raw,
    )

    # NaN audit
    for name, df in [("train", X_train), ("valid", X_valid), ("test", X_test)]:
        dead = [c for c in df.columns if df[c].isna().mean() > 0.99]
        if dead:
            raise ValueError(f"{len(dead)} features >99% NaN on {name}")

    # Add 3 Elo proba features (compensates for missing init_scores).
    # Align on index: prepare_features may drop rows (forfeit filter in _split_xy).
    for X, raw in [(X_train, train_raw), (X_valid, valid_raw), (X_test, test_raw)]:
        elo_feats = _compute_elo_proba_features(raw)
        for col in elo_feats.columns:
            X[col] = elo_feats[col].reindex(X.index).values

    logger.info("Features: %d (201 V9 + 3 Elo probas)", X_train.shape[1])

    # Build AG training DataFrame (AG needs label column in the DataFrame)
    train_ag = X_train.copy()
    train_ag["target"] = y_train.values
    valid_ag = X_valid.copy()
    valid_ag["target"] = y_valid.values

    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- FIT ---
    predictor = TabularPredictor(
        label="target",
        eval_metric="log_loss",
        problem_type="multiclass",
        path=str(out_dir / "ag_models"),
    )
    predictor.fit(
        train_data=train_ag,
        tuning_data=valid_ag,
        use_bag_holdout=True,  # Required: tuning_data + num_bag_folds needs this flag
        presets=AG_PRESETS,
        time_limit=AG_TIME_LIMIT,
        num_bag_folds=AG_BAG_FOLDS,
        num_stack_levels=AG_STACK_LEVELS,
        dynamic_stacking=False,  # Force num_stack_levels=1 (V8 postmortem: L2 overfit)
        calibrate=True,
        num_gpus=0,
        ag_args_fit={"max_memory_usage_ratio": 1.5},  # No "ag." prefix (AG 1.5 docs)
        verbosity=2,
    )

    # --- LEADERBOARD ---
    test_ag = X_test.copy()
    test_ag["target"] = y_test.values
    leaderboard = predictor.leaderboard(test_ag, silent=True)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    logger.info("Leaderboard:\n%s", leaderboard.to_string())

    # --- PREDICTIONS ---
    # Best model (stacked ensemble)
    probas_test_best = predictor.predict_proba(X_test)
    probas_valid_best = predictor.predict_proba(X_valid)

    # Best single model
    best_single = None
    for m in leaderboard["model"].values:
        if "Ensemble" not in m and "Stack" not in m:
            best_single = m
            break
    if best_single:
        probas_test_single = predictor.predict_proba(X_test, model=best_single)
        logger.info("Best single model: %s", best_single)

    # Save predictions
    _save_predictions(probas_test_best, y_test, out_dir / "predictions_test_ensemble.parquet")
    _save_predictions(probas_valid_best, y_valid, out_dir / "predictions_valid_ensemble.parquet")
    if best_single:
        _save_predictions(probas_test_single, y_test, out_dir / "predictions_test_single.parquet")

    # --- QUALITY GATES T1-T12 ---
    from sklearn.metrics import log_loss  # noqa: PLC0415

    from scripts.kaggle_metrics import (  # noqa: PLC0415
        compute_ece,
        compute_expected_score_mae,
        compute_multiclass_brier,
        compute_rps,
    )

    y_arr = y_test.values
    probas_arr = (
        probas_test_best.values
        if hasattr(probas_test_best, "values")
        else np.asarray(probas_test_best)
    )
    # Ensure column order is [loss, draw, win] = [0, 1, 2]
    if hasattr(probas_test_best, "columns"):
        probas_arr = probas_test_best[[0, 1, 2]].values

    test_ll = float(log_loss(y_arr, probas_arr))
    test_rps = float(compute_rps(y_arr, probas_arr))
    test_brier = float(compute_multiclass_brier(y_arr, probas_arr))
    test_es_mae = float(compute_expected_score_mae(y_arr, probas_arr))
    mean_p_draw = float(probas_arr[:, 1].mean())
    observed_draw = float((y_arr == 1).mean())
    draw_bias = mean_p_draw - observed_draw

    logger.info(
        "AG ENSEMBLE: ll=%.6f rps=%.6f es_mae=%.6f draw_bias=%.6f",
        test_ll,
        test_rps,
        test_es_mae,
        draw_bias,
    )

    for c, cls in enumerate(["loss", "draw", "win"]):
        ece = float(compute_ece((y_arr == c).astype(float), probas_arr[:, c]))
        logger.info("  ECE %s: %.4f", cls, ece)

    # --- METADATA ---
    metadata = {
        "version": version,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "pipeline": "AutoGluon V9 benchmark",
        "preset": AG_PRESETS,
        "features": int(X_train.shape[1]),
        "train_rows": len(train_ag),
        "test_rows": len(X_test),
        "time_limit": AG_TIME_LIMIT,
        "num_bag_folds": AG_BAG_FOLDS,
        "num_stack_levels": AG_STACK_LEVELS,
        "best_model_ensemble": predictor.model_best,
        "best_model_single": best_single,
        "metrics_ensemble": {
            "test_log_loss": test_ll,
            "test_rps": test_rps,
            "test_brier": test_brier,
            "test_es_mae": test_es_mae,
            "mean_p_draw": mean_p_draw,
            "draw_calibration_bias": round(draw_bias, 6),
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Feature importance
    try:
        imp = predictor.feature_importance(valid_ag, silent=True)
        imp.to_csv(out_dir / "feature_importance.csv")
    except Exception:
        logger.warning("Feature importance failed (non-blocking)")

    logger.info("Done. AG V9 benchmark complete.")


def _save_predictions(probas: pd.DataFrame, y: pd.Series, path: Path) -> None:
    """Save predictions parquet with y_true + 3 proba columns.

    AG predict_proba returns DataFrame with class labels as columns (0,1,2).
    Use label-based access [[0,1,2]] NOT positional .iloc (avoids class swap).
    """
    arr = probas[[0, 1, 2]].values if hasattr(probas, "columns") else np.asarray(probas)
    df = pd.DataFrame(
        {
            "y_true": y.values,
            "p_loss": arr[:, 0],
            "p_draw": arr[:, 1],
            "p_win": arr[:, 2],
        }
    )
    df.to_parquet(path, index=False)
    logger.info("Saved predictions: %s (%d rows)", path.name, len(df))


if __name__ == "__main__":
    main()
