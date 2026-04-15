"""AutoGluon V9 Benchmark — best_quality multiclass (ISO 42001/5259).

Trains AutoGluon on V9 features + 3 Elo proba features. No init_scores
(AG doesn't support them). Benchmark against V9 single models.
Time guard: fit 7h, post-processing checkpointed after each artifact.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time as _time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
AG_TIME_LIMIT = 25200  # 7h fit (2h margin for post-processing on 9h GPU)
AG_PRESETS = "best_quality"
AG_BAG_FOLDS = 5
AG_STACK_LEVELS = 1  # V8 postmortem: L2/L3 overfit
SESSION_HARD_LIMIT = 32400  # 9h GPU absolute max


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


def _compute_elo_proba_features(
    df: pd.DataFrame,
    draw_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Compute P_elo(W/D/L) using the SAME baseline as V9 models.

    Uses compute_elo_baseline from baselines.py (dynamic white advantage
    +8.5 to +32.4 per Elo level, draw_rate_lookup per band). NOT simplified.
    """
    from scripts.baselines import compute_elo_baseline  # noqa: PLC0415

    blanc_elo = df["blanc_elo"].fillna(1500).values
    noir_elo = df["noir_elo"].fillna(1500).values
    probas = compute_elo_baseline(blanc_elo, noir_elo, draw_lookup)
    return pd.DataFrame(
        {"p_elo_win": probas[:, 2], "p_elo_draw": probas[:, 1], "p_elo_loss": probas[:, 0]},
        index=df.index,
    )


def _save_predictions(probas: pd.DataFrame, y: pd.Series, path: Path) -> None:
    """Save predictions parquet. Label-based [[0,1,2]] NOT positional .iloc."""
    arr = probas[[0, 1, 2]].values if hasattr(probas, "columns") else np.asarray(probas)
    pd.DataFrame(
        {"y_true": y.values, "p_loss": arr[:, 0], "p_draw": arr[:, 1], "p_win": arr[:, 2]}
    ).to_parquet(path, index=False)
    logger.info("CHECKPOINT: %s (%d rows)", path.name, len(y))


def _time_left(t0: float) -> float:
    """Seconds remaining before 9h session hard limit."""
    return SESSION_HARD_LIMIT - (_time.time() - t0)


def main() -> None:  # noqa: PLR0915
    """AutoGluon V9 benchmark with time-guarded checkpoints."""
    t0 = _time.time()
    logger.info("ALICE Engine — AutoGluon V9 Benchmark")
    _setup_kaggle_imports()

    from autogluon.tabular import TabularPredictor  # noqa: PLC0415

    from scripts.cloud.train_kaggle import _load_features  # noqa: PLC0415
    from scripts.kaggle_trainers import prepare_features  # noqa: PLC0415

    train_raw, valid_raw, test_raw, features_dir = _load_features()
    logger.info("Loaded: train=%d valid=%d test=%d", len(train_raw), len(valid_raw), len(test_raw))

    X_train, y_train, X_valid, y_valid, X_test, y_test, encoders = prepare_features(
        train_raw,
        valid_raw,
        test_raw,
    )

    for name, df in [("train", X_train), ("valid", X_valid), ("test", X_test)]:
        dead = [c for c in df.columns if df[c].isna().mean() > 0.99]
        if dead:
            raise ValueError(f"{len(dead)} features >99% NaN on {name}")

    from scripts.features.draw_priors import build_draw_rate_lookup  # noqa: PLC0415

    draw_lookup = build_draw_rate_lookup(train_raw)

    for X, raw in [(X_train, train_raw), (X_valid, valid_raw), (X_test, test_raw)]:
        elo_feats = _compute_elo_proba_features(raw, draw_lookup)
        for col in elo_feats.columns:
            X[col] = elo_feats[col].reindex(X.index).values

    logger.info("Features: %d (201 V9 + 3 Elo probas)", X_train.shape[1])

    train_ag = X_train.copy()
    train_ag["target"] = y_train.values
    valid_ag = X_valid.copy()
    valid_ag["target"] = y_valid.values

    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # === FIT (time-limited, AG handles its own checkpoints) ===
    predictor = TabularPredictor(
        label="target",
        eval_metric="log_loss",
        problem_type="multiclass",
        path=str(out_dir / "ag_models"),
    )
    predictor.fit(
        train_data=train_ag,
        tuning_data=valid_ag,
        use_bag_holdout=True,
        presets=AG_PRESETS,
        time_limit=AG_TIME_LIMIT,
        num_bag_folds=AG_BAG_FOLDS,
        num_stack_levels=AG_STACK_LEVELS,
        dynamic_stacking=False,
        calibrate=True,
        num_gpus=1,
        ag_args_fit={"max_memory_usage_ratio": 1.5},
        verbosity=2,
    )
    logger.info("Fit complete. Time left: %.0fs", _time_left(t0))

    # === POST-PROCESSING: each step checkpointed, time-guarded ===

    # CHECKPOINT 1: leaderboard
    test_ag = X_test.copy()
    test_ag["target"] = y_test.values
    leaderboard = predictor.leaderboard(test_ag, silent=True)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    logger.info(
        "CHECKPOINT: leaderboard.csv (%d models). Time left: %.0fs",
        len(leaderboard),
        _time_left(t0),
    )

    # CHECKPOINT 2: ensemble predictions
    if _time_left(t0) < 300:
        logger.warning("TIME GUARD: <5min left, skipping remaining artifacts")
        return
    probas_test_best = predictor.predict_proba(X_test)
    _save_predictions(probas_test_best, y_test, out_dir / "predictions_test_ensemble.parquet")

    # CHECKPOINT 3: valid predictions
    if _time_left(t0) < 300:
        logger.warning("TIME GUARD: <5min left, skipping remaining artifacts")
        return
    probas_valid_best = predictor.predict_proba(X_valid)
    _save_predictions(probas_valid_best, y_valid, out_dir / "predictions_valid_ensemble.parquet")

    # CHECKPOINT 4: best single model predictions
    best_single = None
    for m in leaderboard["model"].values:
        if "Ensemble" not in m and "Stack" not in m:
            best_single = m
            break
    if best_single and _time_left(t0) > 300:
        probas_test_single = predictor.predict_proba(X_test, model=best_single)
        _save_predictions(probas_test_single, y_test, out_dir / "predictions_test_single.parquet")
        logger.info("Best single: %s", best_single)

    # CHECKPOINT 5: quality metrics
    if _time_left(t0) < 120:
        logger.warning("TIME GUARD: <2min left, skipping metrics+metadata")
        return
    from sklearn.metrics import log_loss  # noqa: PLC0415

    from scripts.kaggle_metrics import (  # noqa: PLC0415
        compute_ece,
        compute_expected_score_mae,
        compute_multiclass_brier,
        compute_rps,
    )

    y_arr = y_test.values
    probas_arr = (
        probas_test_best[[0, 1, 2]].values
        if hasattr(probas_test_best, "columns")
        else np.asarray(probas_test_best)
    )
    test_ll = float(log_loss(y_arr, probas_arr))
    test_rps = float(compute_rps(y_arr, probas_arr))
    test_brier = float(compute_multiclass_brier(y_arr, probas_arr))
    test_es_mae = float(compute_expected_score_mae(y_arr, probas_arr))
    mean_p_draw = float(probas_arr[:, 1].mean())
    draw_bias = mean_p_draw - float((y_arr == 1).mean())

    logger.info(
        "AG ENSEMBLE: ll=%.6f rps=%.6f es_mae=%.6f draw_bias=%.6f",
        test_ll,
        test_rps,
        test_es_mae,
        draw_bias,
    )
    for c, cls in enumerate(["loss", "draw", "win"]):
        logger.info(
            "  ECE %s: %.4f", cls, float(compute_ece((y_arr == c).astype(float), probas_arr[:, c]))
        )

    # CHECKPOINT 6: metadata
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
    logger.info("CHECKPOINT: metadata.json. Time left: %.0fs", _time_left(t0))

    # CHECKPOINT 7: feature importance (non-blocking)
    if _time_left(t0) > 300:
        try:
            imp = predictor.feature_importance(valid_ag, silent=True)
            imp.to_csv(out_dir / "feature_importance.csv")
            logger.info("CHECKPOINT: feature_importance.csv")
        except Exception:
            logger.warning("Feature importance failed (non-blocking)")

    logger.info("Done. AG V9 benchmark complete. Total: %.0fs", _time.time() - t0)


if __name__ == "__main__":
    main()
