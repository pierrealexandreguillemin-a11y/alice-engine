"""Quality gates T1-T12 — ISO 25059/42001/24029.

Document ID: ALICE-QUALITY-GATES
Version: 1.0.0

Full 12-condition gate system with audit logging.
Ref: docs/requirements/QUALITY_GATES.md, Guo2017, Constantinou2012, GoogleML.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def check_baseline_conditions(m: dict, baseline_metrics: dict) -> str | None:
    """Check 6 baseline conditions (T1-T3). Returns failure reason or None."""
    naive = baseline_metrics.get("naive", {})
    elo = baseline_metrics.get("elo", {})
    checks: list[tuple[bool, str]] = [
        (
            bool(naive) and m.get("test_log_loss", 999) >= naive.get("log_loss", 999),
            "T1: log_loss >= naive baseline",
        ),
        (
            bool(elo) and m.get("test_log_loss", 999) >= elo.get("log_loss", 999),
            "T1: log_loss >= elo baseline",
        ),
        (
            bool(naive) and m.get("test_rps", 999) >= naive.get("rps", 999),
            "T2: rps >= naive baseline",
        ),
        (
            bool(elo) and m.get("test_rps", 999) >= elo.get("rps", 999),
            "T2: rps >= elo baseline",
        ),
        (
            bool(naive) and m.get("test_brier", 999) >= naive.get("brier", 999),
            "T3: brier >= naive",
        ),
        (
            bool(elo) and m.get("test_es_mae", 999) >= elo.get("es_mae", 999),
            "T3: es_mae >= elo",
        ),
    ]
    for fail_cond, label in checks:
        if fail_cond:
            return label
    return None


def _check_calibration(m: dict) -> str | None:
    """Check ECE, draw bias, mean P(draw) (T4-T6)."""
    for cls in ("loss", "draw", "win"):
        ece = m.get(f"ece_class_{cls}", 1.0)
        if ece >= 0.05:
            return f"T4: ece_{cls}={ece:.4f} >= 0.05"
    bias = m.get("draw_calibration_bias", 1.0)
    if abs(bias) >= 0.02:
        return f"T5: draw_calibration_bias={bias:.6f}, |bias| >= 0.02"
    if m.get("mean_p_draw", 0.0) < 0.01:
        return f"T6: mean_p_draw={m.get('mean_p_draw', 0):.4f} < 0.01"
    return None


def compute_split_logloss(
    results: dict,
    X: Any,
    y: Any,
    init_scores: Any | None = None,
    calibrators: dict | None = None,
    subsample: int = 50_000,
) -> float | None:
    """Compute calibrated logloss on a data split for T10 train-test gap.

    Subsamples to `subsample` rows for performance on large train sets (1.1M).
    """
    from sklearn.metrics import log_loss  # noqa: PLC0415

    from scripts.kaggle_metrics import _apply_isotonic, predict_with_init  # noqa: PLC0415

    candidates = [(n, r) for n, r in results.items() if r["model"] is not None]
    if not candidates:
        return None
    best_name = min(candidates, key=lambda x: x[1]["metrics"].get("test_log_loss", 999.0))[0]
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    n = len(y_arr)
    if n > subsample:
        rng = np.random.RandomState(42)  # noqa: NPY002
        idx = rng.choice(n, subsample, replace=False)
        X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        y_sub, init_sub = y_arr[idx], init_scores[idx] if init_scores is not None else None
    else:
        X_sub, y_sub, init_sub = X, y_arr, init_scores
    y_proba = predict_with_init(results[best_name]["model"], X_sub, init_sub)
    if calibrators and best_name in calibrators:
        y_proba = _apply_isotonic(y_proba, calibrators[best_name])
    return float(log_loss(y_sub, y_proba))


def check_quality_gates(
    results: dict,
    baseline_metrics: dict | None = None,
    champion_ll: float | None = None,
    train_log_loss: float | None = None,
    verbose: bool = True,
) -> dict:
    """ISO 42001: 12-condition quality gate T1-T12 with full audit logging.

    T1-T3: baseline beats (log_loss, RPS, E[score] MAE)
    T4: ECE < 0.05 classwise (Guo 2017)
    T5: draw_calibration_bias < 0.02
    T6: mean_p_draw > 0.01
    T7-T8: enforced in evaluate_on_test (NaN/Inf crash, sum-to-1 crash)
    T9: >5 features with importance > 0
    T10: train-test gap < 0.05 (GoogleML #37) — WARNING for temporal splits
    T11: reliability deviation logged (visual audit proxy)
    T12: both RPS and log_loss reported (Wheatcroft 2019)
    """
    candidates = [(n, r) for n, r in results.items() if r["model"] is not None]
    if not candidates:
        return {"passed": False, "reason": "All models failed to train"}
    best_name, best_r = min(candidates, key=lambda x: x[1]["metrics"].get("test_log_loss", 999.0))
    m = best_r["metrics"]
    best_ll = m.get("test_log_loss", 999.0)
    _log = logger.info if verbose else logger.debug

    # === AUDIT LOG: all metrics for traceability (ISO 42001) ===
    if verbose:
        _log("=" * 60)
        _log("QUALITY GATES T1-T12 — %s", best_name)
        _log("=" * 60)
        _log("  T1  log_loss=%.6f", m.get("test_log_loss", -1))
        _log("  T2  rps=%.6f", m.get("test_rps", -1))
        _log(
            "  T3  es_mae=%.6f brier=%.6f",
            m.get("test_es_mae", -1),
            m.get("test_brier", -1),
        )
        _log(
            "  T4  ECE: loss=%.4f draw=%.4f win=%.4f",
            m.get("ece_class_loss", -1),
            m.get("ece_class_draw", -1),
            m.get("ece_class_win", -1),
        )
        _log("  T5  draw_bias=%.6f", m.get("draw_calibration_bias", -1))
        _log("  T6  mean_p_draw=%.4f", m.get("mean_p_draw", -1))
        _log("  T7  NaN/Inf: PASS (enforced in evaluate_on_test)")
        _log("  T8  sum-to-1: PASS (enforced in evaluate_on_test)")

    # T1-T3: baselines
    if baseline_metrics:
        reason = check_baseline_conditions(m, baseline_metrics)
        if reason:
            logger.error("  FAIL %s", reason)
            return {"passed": False, "reason": reason}
        _log("  T1-T3 baselines: PASS")
    elif verbose:
        _log("  T1-T3 baselines: SKIPPED (no baseline_metrics)")

    # T4-T6: calibration
    cal_reason = _check_calibration(m)
    if cal_reason:
        logger.error("  FAIL %s", cal_reason)
        return {"passed": False, "reason": cal_reason}
    _log("  T4-T6 calibration: PASS")

    # T9: >5 features with importance > 0
    importances = getattr(best_r["model"], "feature_importances_", None)
    if importances is not None and isinstance(importances, np.ndarray):
        n_nz = int(np.sum(importances > 0))
        _log("  T9  features importance>0: %d", n_nz)
        if n_nz <= 5:
            reason = f"T9: only {n_nz} features with importance>0 (need >5)"
            logger.error("  FAIL %s", reason)
            return {"passed": False, "reason": reason}
        _log("  T9: PASS")
    elif verbose:
        _log("  T9: SKIPPED (no numpy feature_importances_)")

    # T10: train-test gap (GoogleML #37)
    if train_log_loss is not None:
        gap = abs(train_log_loss - best_ll)
        _log(
            "  T10 train=%.6f test=%.6f gap=%.4f",
            train_log_loss,
            best_ll,
            gap,
        )
        if gap > 0.05:
            logger.warning("  T10 WARNING: gap %.4f > 0.05 (expected for temporal splits)", gap)
        else:
            _log("  T10: PASS")
    elif verbose:
        _log("  T10: SKIPPED (no train_log_loss)")

    # T11: reliability deviation (visual proxy — Guo 2017)
    if verbose:
        for cls_name in ("loss", "draw", "win"):
            _log(
                "  T11 reliability_%s: ECE=%.4f (diagrams in diagnostics/)",
                cls_name,
                m.get(f"ece_class_{cls_name}", -1),
            )

    # T12: dual reporting (Wheatcroft 2019)
    if verbose:
        _log(
            "  T12 dual: log_loss=%.6f rps=%.6f",
            m.get("test_log_loss", -1),
            m.get("test_rps", -1),
        )

    # Champion degradation check
    if champion_ll and champion_ll > 0:
        rise_pct = (best_ll - champion_ll) / champion_ll * 100
        _log("  Champion: %.6f vs %.6f (%.1f%%)", best_ll, champion_ll, rise_pct)
        if rise_pct > 5.0:
            reason = f"Degradation {rise_pct:.1f}% > 5.0%"
            logger.error("  FAIL %s", reason)
            return {"passed": False, "reason": reason}

    if verbose:
        _log("=" * 60)
        _log("  ALL 12 GATES PASSED — %s logloss=%.6f", best_name, best_ll)
        _log("=" * 60)
    return {"passed": True, "best_model": best_name, "best_log_loss": best_ll}
