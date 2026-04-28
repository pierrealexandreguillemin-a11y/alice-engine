"""Refit MLP(32,16) champion meta-learner from OOF predictions (D-P3-13 résorption).

Plan 3 V2 T22.0 : reproduit le pipeline `meta_learner_sweep` best config :
- features: 18 (9 raw probas + 3 std + 3 max + 3 entropy) via build_meta_features
- arch: MLP(32, 16), early_stopping=True, alpha=1e-4
- post-calibration: temperature scaling (Guo 2017)
- expected: log_loss ~0.5530, ECE_draw ~0.0016 sur test set

Persiste 2 artefacts dans `models/cache/` (path attendu par
`scripts.serving.model_loader`) :
- mlp_meta_learner.joblib : sklearn.MLPClassifier fit
- temperature_T.joblib : float scalar T

Usage:
    python -m scripts.cloud.refit_mlp_champion

Sources SOTA
------------
- Mitchell M. et al. 2019. "Model Cards for Model Reporting." FAccT 220-229.
- Guo C. et al. 2017. "On Calibration of Modern Neural Networks." ICML.
- Kull M. et al. 2019. "Beyond temperature scaling : Obtaining well-calibrated
  multi-class probabilities with Dirichlet calibration." NeurIPS.
- ISO/IEC 42001:2023 Annex A.6 (Lifecycle traceability).

Document ID: ALICE-MLP-REFIT
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from scripts.serving.meta_features import build_meta_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OOF_DIR = Path("results/oof_merged")
OUT_DIR = Path("models/cache")

# Reproduit best config du sweep (commit Apr 16 2026, results/meta_learner_sweep)
ARCH_LAYERS: tuple[int, ...] = (32, 16)
ALPHA: float = 1e-4
MAX_ITER: int = 500
VAL_FRAC: float = 0.1
SEED: int = 42


def _load_oof() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load OOF predictions and stack 18 meta-features.

    @returns (X_oof_18, y_oof, X_test_18, y_test).
    @raises FileNotFoundError: OOF parquets absent.
    """
    xgb_lgb_oof = pd.read_parquet(OOF_DIR / "xgb_lgb_oof_predictions.parquet")
    cb_oof = pd.read_parquet(OOF_DIR / "cb_oof_predictions.parquet")
    xgb_lgb_test = pd.read_parquet(OOF_DIR / "xgb_lgb_test_predictions.parquet")
    cb_test = pd.read_parquet(OOF_DIR / "cb_test_predictions.parquet")

    y_oof = xgb_lgb_oof["y_true"].to_numpy().astype(int)
    y_test = xgb_lgb_test["y_true"].to_numpy().astype(int)

    p_xgb_oof = xgb_lgb_oof[["xgb_p_loss", "xgb_p_draw", "xgb_p_win"]].to_numpy()
    p_lgb_oof = xgb_lgb_oof[["lgb_p_loss", "lgb_p_draw", "lgb_p_win"]].to_numpy()
    p_cb_oof = cb_oof[["cb_p_loss", "cb_p_draw", "cb_p_win"]].to_numpy()

    p_xgb_test = xgb_lgb_test[["xgb_p_loss", "xgb_p_draw", "xgb_p_win"]].to_numpy()
    p_lgb_test = xgb_lgb_test[["lgb_p_loss", "lgb_p_draw", "lgb_p_win"]].to_numpy()
    p_cb_test = cb_test[["cb_p_loss", "cb_p_draw", "cb_p_win"]].to_numpy()

    x_oof_18 = build_meta_features(p_xgb_oof, p_lgb_oof, p_cb_oof)
    x_test_18 = build_meta_features(p_xgb_test, p_lgb_test, p_cb_test)
    logger.info("OOF: %d rows × %d feats | TEST: %d rows", *x_oof_18.shape, x_test_18.shape[0])
    return x_oof_18, y_oof, x_test_18, y_test


def _fit_mlp(x_oof: np.ndarray, y_oof: np.ndarray) -> MLPClassifier:
    """Fit MLP(32,16) early-stopping (sweep best config)."""
    mlp = MLPClassifier(
        hidden_layer_sizes=ARCH_LAYERS,
        max_iter=MAX_ITER,
        early_stopping=True,
        validation_fraction=VAL_FRAC,
        random_state=SEED,
        alpha=ALPHA,
    )
    t0 = time.time()
    mlp.fit(x_oof, y_oof)
    logger.info("MLP fit in %.1fs (n_iter=%d)", time.time() - t0, mlp.n_iter_)
    return mlp


def _calibrate_temperature(p_oof_raw: np.ndarray, y_oof: np.ndarray) -> float:
    """Tune scalar T minimizing OOF log-loss after softmax(logits / T).

    Guo 2017 §3 : single scalar maintains argmax, only sharpens/flattens
    distribution. Search range [0.5, 2.0] standard.
    """

    def _nll(temp: float) -> float:
        logits = np.log(np.clip(p_oof_raw, 1e-7, 1.0)) / temp
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return float(log_loss(y_oof, p))

    result = minimize_scalar(_nll, bounds=(0.5, 2.0), method="bounded")
    return float(result.x)


def _apply_temperature(p_raw: np.ndarray, temp: float) -> np.ndarray:
    """Softmax(log(p_raw) / T) avec stabilisation max-substraction."""
    logits = np.log(np.clip(p_raw, 1e-7, 1.0)) / temp
    logits -= logits.max(axis=1, keepdims=True)
    return np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)


def _ece(y_true: np.ndarray, p_class: np.ndarray, *, n_bins: int = 10) -> float:
    """Expected Calibration Error binary (Guo 2017 §3.2)."""
    y_bin = (y_true == 1).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p_class >= bins[i]) & (p_class < bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.mean()) * abs(p_class[mask].mean() - y_bin[mask].mean())
    return float(ece)


def _persist(mlp: MLPClassifier, temp: float, metrics: dict[str, float]) -> dict[str, str]:
    """Dump artefacts + metadata + sha256 hashes (ISO 42001 traceability)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mlp_path = OUT_DIR / "mlp_meta_learner.joblib"
    temp_path = OUT_DIR / "temperature_T.joblib"
    meta_path = OUT_DIR / "mlp_champion_metadata.json"

    joblib.dump(mlp, mlp_path, compress=3)
    joblib.dump(float(temp), temp_path)

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    hashes = {"mlp": _sha(mlp_path), "temperature": _sha(temp_path)}
    metadata: dict[str, Any] = {
        "document_id": "ALICE-MLP-CHAMPION",
        "version": "1.0.0",
        "fit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": list(ARCH_LAYERS),
        "alpha": ALPHA,
        "max_iter": MAX_ITER,
        "validation_fraction": VAL_FRAC,
        "random_state": SEED,
        "n_iter_actual": int(mlp.n_iter_),
        "feature_count": 18,
        "feature_pipeline": "scripts.serving.meta_features.build_meta_features",
        "temperature_T": float(temp),
        "metrics_test": metrics,
        "sha256": hashes,
        "iso_compliance": ["ISO/IEC 42001:2023 Annex A.6", "ISO/IEC 25059:2023"],
        "sources": {
            "sweep": "results/meta_learner_sweep/sweep_results.csv (best config)",
            "guo2017": "Guo C. et al. ICML 2017 — temperature scaling",
            "mitchell2019": "Mitchell M. et al. FAccT 2019 — model card §1",
        },
        "debt_resolved": "D-P3-13 (MLP champion artifact missing → fallback)",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote %s (sha %s...)", mlp_path, hashes["mlp"][:12])
    logger.info("Wrote %s (T=%.6f)", temp_path, temp)
    logger.info("Wrote %s", meta_path)
    return hashes


def main() -> None:
    """Refit MLP champion + persister artefacts."""
    logger.info("D-P3-13 résorption: refit MLP(32,16) champion from OOF Phase 2")

    x_oof, y_oof, x_test, y_test = _load_oof()

    mlp = _fit_mlp(x_oof, y_oof)
    p_oof_raw = mlp.predict_proba(x_oof)
    p_test_raw = mlp.predict_proba(x_test)

    temp = _calibrate_temperature(p_oof_raw, y_oof)
    p_test_cal = _apply_temperature(p_test_raw, temp)

    metrics = {
        "log_loss_test_raw": float(log_loss(y_test, p_test_raw)),
        "log_loss_test_calibrated": float(log_loss(y_test, p_test_cal)),
        "ece_draw_test_raw": _ece(y_test, p_test_raw[:, 1]),
        "ece_draw_test_calibrated": _ece(y_test, p_test_cal[:, 1]),
        "temperature_T": float(temp),
    }
    logger.info("METRICS test : %s", json.dumps(metrics, indent=2))

    _persist(mlp, temp, metrics)
    logger.info("D-P3-13 résorption COMPLETE — relancer harness, fallback désactivé.")


if __name__ == "__main__":
    main()
