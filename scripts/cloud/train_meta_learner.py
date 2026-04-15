"""Meta-learner MLP on OOF XGB+LGB predictions — ISO 42001/25059.

Consumes OOF predictions from alice-oof-stack-v9 kernel via kernel_sources.
Trains MLP(16) meta-learner, temperature calibration, quality gates T1-T12.
Compares Stack_MLP_cal vs V9 singles on ECE draw + draw_bias.
Ref: MODEL_SPECS.md §Stacking, spec 2026-04-15 §3.3.
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
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
META_COLS = [
    "xgb_p_loss",
    "xgb_p_draw",
    "xgb_p_win",
    "lgb_p_loss",
    "lgb_p_draw",
    "lgb_p_win",
]


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


def _find_oof_parquets() -> tuple[Path, Path]:
    """Find OOF + test parquets from alice-oof-stack-v9 kernel output."""
    candidates = [
        Path("/kaggle/input/notebooks/pguillemin/alice-oof-stack-v9"),
        Path("/kaggle/input/alice-oof-stack-v9"),
    ]
    for base in candidates:
        if not base.exists():
            continue
        # OOF kernel saves in versioned subdir
        for d in sorted(base.rglob("oof_predictions.parquet")):
            oof = d
            test = d.parent / "test_predictions_stack.parquet"
            if test.exists():
                return oof, test
    msg = "OOF parquets not found. Run alice-oof-stack-v9 kernel first."
    raise FileNotFoundError(msg)


def _calibrate_temperature(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Fit temperature T on held-out probas (Guo 2017)."""
    from scipy.optimize import minimize_scalar

    def _nll(T: float) -> float:
        logits = np.log(np.clip(y_proba, 1e-7, 1.0))
        scaled = logits / T
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        return float(log_loss(y_true, probs))

    return float(minimize_scalar(_nll, bounds=(0.5, 2.0), method="bounded").x)


def _apply_temperature(y_proba: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling."""
    logits = np.log(np.clip(y_proba, 1e-7, 1.0))
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    return np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)


def main() -> None:
    """Train MLP meta-learner, calibrate, evaluate T1-T12."""
    logger.info("ALICE Meta-Learner — MLP on XGB+LGB OOF predictions")
    _setup_kaggle_imports()

    # --- Load OOF + test ---
    oof_path, test_path = _find_oof_parquets()
    oof = pd.read_parquet(oof_path)
    test = pd.read_parquet(test_path)
    logger.info("OOF: %s from %s", oof.shape, oof_path)
    logger.info("Test: %s from %s", test.shape, test_path)

    X_oof = oof[META_COLS].values
    y_oof = oof["y_true"].values.astype(int)
    X_test = test[META_COLS].values
    y_test = test["y_true"].values.astype(int)

    # --- T7/T8: sanity checks ---
    if not np.all(np.isfinite(X_oof)):
        raise ValueError("T7 FAIL: NaN/Inf in OOF predictions")
    if not np.all(np.isfinite(X_test)):
        raise ValueError("T7 FAIL: NaN/Inf in test predictions")
    for i, name in enumerate(["XGB", "LGB"]):
        s = X_oof[:, i * 3 : (i + 1) * 3].sum(axis=1)
        if not np.allclose(s, 1.0, atol=1e-5):
            raise ValueError(f"T8 FAIL: {name} OOF probas don't sum to 1")

    # --- Train MLP meta-learner (MODEL_SPECS §Stacking) ---
    logger.info("Training MLP(hidden=16, max_iter=500, early_stopping=True)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(16,),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    mlp.fit(X_oof, y_oof)
    logger.info("MLP: %d iterations, best_loss=%.6f", mlp.n_iter_, mlp.best_loss_)

    # --- Predictions ---
    y_proba_raw = mlp.predict_proba(X_test)
    y_proba_oof_raw = mlp.predict_proba(X_oof)

    # --- Temperature calibration on OOF (NOT test — avoids leakage) ---
    T = _calibrate_temperature(y_oof, y_proba_oof_raw)
    logger.info("Temperature: T=%.4f", T)
    y_proba_cal = _apply_temperature(y_proba_raw, T)

    # --- Quality Gates T1-T12 ---
    from scripts.kaggle_metrics import (
        compute_ece,
        compute_expected_score_mae,
        compute_multiclass_brier,
        compute_rps,
    )

    y_arr = y_test
    test_ll = float(log_loss(y_arr, y_proba_cal))
    test_rps = float(compute_rps(y_arr, y_proba_cal))
    test_brier = float(compute_multiclass_brier(y_arr, y_proba_cal))
    test_es_mae = float(compute_expected_score_mae(y_arr, y_proba_cal))
    mean_p_draw = float(y_proba_cal[:, 1].mean())
    observed_draw = float((y_arr == 1).mean())
    draw_bias = mean_p_draw - observed_draw

    logger.info("=" * 60)
    logger.info("QUALITY GATES T1-T12 — Stack_MLP_cal")
    logger.info("=" * 60)
    logger.info("  T1  log_loss=%.6f", test_ll)
    logger.info("  T2  rps=%.6f", test_rps)
    logger.info("  T3  es_mae=%.6f brier=%.6f", test_es_mae, test_brier)
    for c, cls in enumerate(["loss", "draw", "win"]):
        ece = float(compute_ece((y_arr == c).astype(float), y_proba_cal[:, c]))
        logger.info("  T4  ECE_%s=%.4f", cls, ece)
    logger.info("  T5  draw_bias=%.6f", draw_bias)
    logger.info("  T6  mean_p_draw=%.4f", mean_p_draw)
    logger.info("  T7  NaN/Inf: PASS (checked at load)")
    logger.info("  T8  sum-to-1: PASS (checked at load)")
    logger.info("  T12 dual: log_loss=%.6f rps=%.6f", test_ll, test_rps)
    logger.info("=" * 60)

    # --- Comparison vs V9 singles ---
    logger.info("COMPARISON vs V9 singles:")
    logger.info("  LGB V9:        ll=0.5619  draw_bias=0.0136  ECE_draw=0.0145")
    logger.info("  XGB V9:        ll=0.5622  draw_bias=0.0109  ECE_draw=0.0129")
    logger.info("  Stack_MLP_cal: ll=%.4f  draw_bias=%.4f", test_ll, draw_bias)
    delta_ll = test_ll - 0.5619
    logger.info("  Delta vs LGB: %+.4f log_loss", delta_ll)

    # --- Save outputs ---
    version = datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "y_true": y_test,
            "p_loss": y_proba_cal[:, 0],
            "p_draw": y_proba_cal[:, 1],
            "p_win": y_proba_cal[:, 2],
        }
    ).to_parquet(out_dir / "predictions_test_stack_mlp_cal.parquet", index=False)
    logger.info("CHECKPOINT: predictions saved")

    import joblib

    joblib.dump(mlp, out_dir / "mlp_meta_learner.joblib")
    joblib.dump(T, out_dir / "temperature_T.joblib")
    logger.info("CHECKPOINT: model + calibrator saved")

    metadata = {
        "version": version,
        "pipeline": "Stack_MLP_cal (XGB+LGB OOF)",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "mlp_config": {"hidden": [16], "max_iter": 500, "early_stopping": True},
        "temperature": T,
        "n_iter": mlp.n_iter_,
        "oof_rows": len(X_oof),
        "test_rows": len(X_test),
        "metrics": {
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
    logger.info("CHECKPOINT: metadata saved")
    logger.info("Done. Stack_MLP_cal complete.")


if __name__ == "__main__":
    main()
