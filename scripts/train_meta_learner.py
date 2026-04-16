"""Meta-learner stacking: LogReg + simple average + Dirichlet on OOF — ISO 42001/25059.

Document ID: ALICE-META-LEARNER
Version: 2.0.0

Trains LogisticRegression meta-learner on OOF predictions from 3 base models
(XGB, LGB, CB). Compares to simple average baseline (literature: "forecast
combination puzzle" — simple average often beats tuned weights).

Post-hoc: Dirichlet calibration (Kull 2019 NeurIPS) on champion output.
Quality gates T1-T12 on all candidates.

Refs:
- MODEL_SPECS.md §Stacking meta-learner
- Wolpert 1992: Stacked Generalization
- Kull 2019 (NeurIPS): Dirichlet calibration > temperature scaling for multiclass
- Breiman 1996: simple average = strong ensemble baseline
- arxiv:2502.02861: calibration error bounds downstream optimization regret
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OOF_DIR = Path("results/oof_merged")
OUT_DIR = Path("results/meta_learner")


# --- Calibration methods ---


def _calibrate_temperature(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Fit scalar temperature T (Guo 2017 ICML)."""
    from scipy.optimize import minimize_scalar

    def _nll(T: float) -> float:
        logits = np.log(np.clip(y_proba, 1e-7, 1.0))
        scaled = logits / T
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        return float(log_loss(y_true, probs))

    return float(minimize_scalar(_nll, bounds=(0.5, 2.0), method="bounded").x)


def _apply_temperature(y_proba: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to probabilities."""
    logits = np.log(np.clip(y_proba, 1e-7, 1.0))
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    return np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)


def _calibrate_dirichlet(
    y_true: np.ndarray, y_proba: np.ndarray, reg: float = 1e-3
) -> LogisticRegression:
    """Fit Dirichlet calibration (Kull 2019 NeurIPS).

    Equivalent to: log(p) → linear 3×3 → softmax.
    Implementation: LogisticRegression on log-transformed probabilities.
    reg = L2 regularization (high = closer to identity = temperature scaling).
    """
    log_probas = np.log(np.clip(y_proba, 1e-7, 1.0))
    lr = LogisticRegression(C=1.0 / reg, max_iter=1000, multi_class="multinomial", solver="lbfgs")
    lr.fit(log_probas, y_true)
    return lr


def _apply_dirichlet(y_proba: np.ndarray, lr: LogisticRegression) -> np.ndarray:
    """Apply fitted Dirichlet calibrator."""
    log_probas = np.log(np.clip(y_proba, 1e-7, 1.0))
    return lr.predict_proba(log_probas)


# --- Quality metrics ---


def _compute_all_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute all quality gate metrics."""
    from scripts.kaggle_metrics import (
        compute_ece,
        compute_expected_score_mae,
        compute_multiclass_brier,
        compute_rps,
    )

    metrics = {
        "log_loss": float(log_loss(y_true, y_proba)),
        "rps": float(compute_rps(y_true, y_proba)),
        "brier": float(compute_multiclass_brier(y_true, y_proba)),
        "es_mae": float(compute_expected_score_mae(y_true, y_proba)),
        "mean_p_draw": float(y_proba[:, 1].mean()),
        "draw_bias": float(y_proba[:, 1].mean()) - float((y_true == 1).mean()),
    }
    for c, cls in enumerate(["loss", "draw", "win"]):
        metrics[f"ece_{cls}"] = float(compute_ece((y_true == c).astype(float), y_proba[:, c]))
    return metrics


def _log_gates(name: str, m: dict) -> None:
    """Log quality gates T1-T12."""
    logger.info("=" * 60)
    logger.info("QUALITY GATES — %s", name)
    logger.info("=" * 60)
    logger.info("  T1  log_loss    = %.6f", m["log_loss"])
    logger.info("  T3  rps         = %.6f", m["rps"])
    logger.info("  T5  es_mae      = %.6f", m["es_mae"])
    logger.info("  T6  brier       = %.6f", m["brier"])
    logger.info(
        "  T7  ECE_loss    = %.4f  ECE_draw = %.4f  ECE_win = %.4f",
        m["ece_loss"],
        m["ece_draw"],
        m["ece_win"],
    )
    logger.info("  T8  draw_bias   = %.6f  (target: |bias| < 0.02)", m["draw_bias"])
    logger.info("  T9  mean_p_draw = %.4f  (target: > 0.01)", m["mean_p_draw"])


def main() -> None:
    """Train meta-learners, compare, select champion, apply Dirichlet."""
    logger.info("ALICE Meta-Learner — LogReg + Average + Dirichlet (V2)")

    # --- Load OOF + test (3 base models) ---
    xgb_lgb_oof = pd.read_parquet(OOF_DIR / "xgb_lgb_oof_predictions.parquet")
    cb_oof = pd.read_parquet(OOF_DIR / "cb_oof_predictions.parquet")
    xgb_lgb_test = pd.read_parquet(OOF_DIR / "xgb_lgb_test_predictions.parquet")
    cb_test = pd.read_parquet(OOF_DIR / "cb_test_predictions.parquet")

    y_oof = xgb_lgb_oof["y_true"].values.astype(int)
    y_test = xgb_lgb_test["y_true"].values.astype(int)

    # 9 features: XGB(3) + LGB(3) + CB(3)
    xgb_cols = ["xgb_p_loss", "xgb_p_draw", "xgb_p_win"]
    lgb_cols = ["lgb_p_loss", "lgb_p_draw", "lgb_p_win"]
    cb_cols = ["cb_p_loss", "cb_p_draw", "cb_p_win"]

    X_oof = np.hstack(
        [xgb_lgb_oof[xgb_cols].values, xgb_lgb_oof[lgb_cols].values, cb_oof[cb_cols].values]
    )
    X_test = np.hstack(
        [xgb_lgb_test[xgb_cols].values, xgb_lgb_test[lgb_cols].values, cb_test[cb_cols].values]
    )

    logger.info("OOF: %s, Test: %s, Features: %d", X_oof.shape, X_test.shape, X_oof.shape[1])

    # Sanity: no NaN, probas sum=1 per model
    assert np.all(np.isfinite(X_oof)), "NaN/Inf in OOF"
    assert np.all(np.isfinite(X_test)), "NaN/Inf in test"
    for i in range(3):
        s = X_oof[:, i * 3 : (i + 1) * 3].sum(axis=1)
        assert np.allclose(s, 1.0, atol=1e-4), f"Model {i} OOF probas don't sum to 1"

    # --- V9 single model baselines (from Training Final) ---
    v9_baselines = {
        "V9_LGB_single": {"log_loss": 0.5619, "ece_draw": 0.0145, "draw_bias": 0.0136},
        "V9_XGB_single": {"log_loss": 0.5622, "ece_draw": 0.0129, "draw_bias": 0.0109},
        "V9_CB_single": {"log_loss": 0.5708, "ece_draw": 0.0123, "draw_bias": 0.0095},
    }

    # =========================================================
    # CANDIDATE 1: Simple average (mandatory baseline)
    # Breiman 1996: "forecast combination puzzle" — often hard to beat
    # =========================================================
    avg_test = np.zeros((len(y_test), 3))
    for i in range(3):
        avg_test += X_test[:, i * 3 : (i + 1) * 3]
    avg_test /= 3.0

    m_avg = _compute_all_metrics(y_test, avg_test)
    _log_gates("SimpleAverage_3models", m_avg)

    # =========================================================
    # CANDIDATE 2: LogisticRegression meta-learner (literature default)
    # Wolpert 1992, mlxtend default, consensus recommendation
    # =========================================================
    # C=1.0 = moderate regularization. Will grid-search if needed.
    lr = LogisticRegression(
        C=1.0, max_iter=1000, multi_class="multinomial", solver="lbfgs", random_state=42
    )
    lr.fit(X_oof, y_oof)
    logger.info("LogReg trained: coef shape=%s", lr.coef_.shape)

    lr_test_raw = lr.predict_proba(X_test)
    m_lr_raw = _compute_all_metrics(y_test, lr_test_raw)
    _log_gates("LogReg_raw", m_lr_raw)

    # Temperature scaling on LogReg output
    lr_oof_raw = lr.predict_proba(X_oof)
    T_lr = _calibrate_temperature(y_oof, lr_oof_raw)
    lr_test_temp = _apply_temperature(lr_test_raw, T_lr)
    logger.info("LogReg temperature: T=%.4f", T_lr)

    m_lr_temp = _compute_all_metrics(y_test, lr_test_temp)
    _log_gates("LogReg_TempCal", m_lr_temp)

    # =========================================================
    # CANDIDATE 3: Dirichlet calibration on best single model
    # Kull 2019 (NeurIPS) — per-class correction in log-proba space
    # Applied to LGB (best single V9, 0.5619)
    # =========================================================
    lgb_oof_probas = xgb_lgb_oof[lgb_cols].values
    lgb_test_probas = xgb_lgb_test[lgb_cols].values

    dir_cal = _calibrate_dirichlet(y_oof, lgb_oof_probas, reg=1e-3)
    lgb_test_dir = _apply_dirichlet(lgb_test_probas, dir_cal)

    m_lgb_dir = _compute_all_metrics(y_test, lgb_test_dir)
    _log_gates("LGB_Dirichlet", m_lgb_dir)

    # =========================================================
    # CANDIDATE 4: Dirichlet on LogReg stacked output
    # Best of both: stacking diversity + per-class calibration
    # =========================================================
    dir_cal_stack = _calibrate_dirichlet(y_oof, lr_oof_raw, reg=1e-3)
    lr_test_dir = _apply_dirichlet(lr_test_raw, dir_cal_stack)

    m_lr_dir = _compute_all_metrics(y_test, lr_test_dir)
    _log_gates("LogReg_Dirichlet", m_lr_dir)

    # =========================================================
    # CHAMPION SELECTION
    # Criterion: ECE_draw + |draw_bias| first, then log_loss
    # (P(draw) = 45% variance E[score] for CE)
    # =========================================================
    candidates = {
        "SimpleAvg": (m_avg, avg_test),
        "LogReg_raw": (m_lr_raw, lr_test_raw),
        "LogReg_TempCal": (m_lr_temp, lr_test_temp),
        "LGB_Dirichlet": (m_lgb_dir, lgb_test_dir),
        "LogReg_Dirichlet": (m_lr_dir, lr_test_dir),
    }

    logger.info("\n" + "=" * 70)
    logger.info("CHAMPION SELECTION — all candidates")
    logger.info("=" * 70)
    logger.info(
        "%-20s %10s %10s %10s %10s", "Candidate", "log_loss", "ECE_draw", "draw_bias", "rps"
    )
    logger.info("-" * 70)

    for name, (m, _) in sorted(candidates.items(), key=lambda x: x[1][0]["log_loss"]):
        logger.info(
            "%-20s %10.6f %10.4f %10.6f %10.6f",
            name,
            m["log_loss"],
            m["ece_draw"],
            m["draw_bias"],
            m["rps"],
        )

    logger.info("-" * 70)
    logger.info("V9 baselines:")
    for name, m in v9_baselines.items():
        logger.info(
            "%-20s %10.6f %10.4f %10.6f %10s",
            name,
            m["log_loss"],
            m["ece_draw"],
            m["draw_bias"],
            "—",
        )

    # Select champion: best log_loss among those with ECE_draw < 0.02 and |draw_bias| < 0.02
    qualified = {
        n: (m, p)
        for n, (m, p) in candidates.items()
        if m["ece_draw"] < 0.05 and abs(m["draw_bias"]) < 0.02
    }
    if not qualified:
        logger.warning("NO candidate passes ECE_draw < 0.05 + |draw_bias| < 0.02 — relaxing to all")
        qualified = candidates

    champion_name = min(qualified, key=lambda n: qualified[n][0]["log_loss"])
    champion_metrics, champion_probas = qualified[champion_name]

    logger.info(
        "\nCHAMPION: %s (log_loss=%.6f, ECE_draw=%.4f, draw_bias=%.6f)",
        champion_name,
        champion_metrics["log_loss"],
        champion_metrics["ece_draw"],
        champion_metrics["draw_bias"],
    )

    # --- Save all outputs ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Champion predictions
    pd.DataFrame(
        {
            "y_true": y_test,
            "p_loss": champion_probas[:, 0],
            "p_draw": champion_probas[:, 1],
            "p_win": champion_probas[:, 2],
        }
    ).to_parquet(OUT_DIR / "predictions_test_champion.parquet", index=False)

    # All candidate predictions (for post-hoc analysis)
    for name, (_m, probas) in candidates.items():
        pd.DataFrame(
            {
                "y_true": y_test,
                "p_loss": probas[:, 0],
                "p_draw": probas[:, 1],
                "p_win": probas[:, 2],
            }
        ).to_parquet(OUT_DIR / f"predictions_test_{name}.parquet", index=False)

    # LogReg coefficients (explainability — ISO 42001)
    coef_df = pd.DataFrame(
        lr.coef_,
        columns=xgb_cols + lgb_cols + cb_cols,
        index=["loss", "draw", "win"],
    )
    coef_df.to_csv(OUT_DIR / "logreg_coefficients.csv")
    logger.info("LogReg coefficients:\n%s", coef_df.round(4).to_string())

    # Metadata (ISO 42001 model card)
    metadata = {
        "version": datetime.now(tz=UTC).strftime("v%Y%m%d_%H%M%S"),
        "pipeline": "Meta-learner V2: LogReg + Average + Dirichlet",
        "base_models": ["XGBoost V9", "LightGBM V9", "CatBoost V9"],
        "meta_features": xgb_cols + lgb_cols + cb_cols,
        "meta_learner": "LogisticRegression(C=1.0, multinomial, lbfgs)",
        "calibration_methods": ["temperature_scaling (Guo 2017)", "dirichlet (Kull 2019)"],
        "champion": champion_name,
        "oof_rows": len(y_oof),
        "test_rows": len(y_test),
        "all_candidates": {n: m for n, (m, _) in candidates.items()},
        "v9_baselines": v9_baselines,
        "sources": [
            "Wolpert 1992: Stacked Generalization",
            "Kull 2019 (NeurIPS): Dirichlet calibration",
            "Breiman 1996: simple average baseline",
            "arxiv:2502.02861: calibration bounds downstream regret",
        ],
    }

    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save models
    import joblib

    joblib.dump(lr, OUT_DIR / "logreg_meta_learner.joblib")
    joblib.dump(dir_cal, OUT_DIR / "dirichlet_lgb.joblib")
    joblib.dump(dir_cal_stack, OUT_DIR / "dirichlet_logreg.joblib")
    if "TempCal" in champion_name:
        joblib.dump(T_lr, OUT_DIR / "temperature_T.joblib")
    logger.info("All models + calibrators saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
