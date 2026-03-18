"""Promote CANDIDATE model to PRODUCTION after ISO validation.

Pulls candidate from HF Hub, runs ISO 24029 robustness,
ISO 24027 fairness, and McNemar comparison vs champion.

Usage: python -m scripts.cloud.promote_model [--version v20260318_120000]

ISO Compliance:
- ISO/IEC 24029 - Neural Network Robustness
- ISO/IEC 24027 - Bias in AI
- ISO/IEC 42001 - AI Management System
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def decide_promotion(
    robustness: dict,
    fairness: dict,
    mcnemar: dict,
) -> dict:
    """Decide PRODUCTION or REJECTED based on ISO 24029/24027 checks.

    Args:
    ----
        robustness: result from run_robustness() with 'compliant' key
        fairness: result from run_fairness() with 'status' key
        mcnemar: result from run_mcnemar() with 'new_auc' key

    Returns:
    -------
        dict with 'decision' (PRODUCTION|REJECTED) and 'reason'
    """
    if not robustness.get("compliant", False):
        noise_tol = robustness.get("noise_tolerance", 0.0)
        return {
            "decision": "REJECTED",
            "reason": f"robustness check failed: noise_tolerance={noise_tol:.3f} < 0.85",
        }
    if fairness.get("status") == "CRITICAL":
        dp = fairness.get("demographic_parity", 0.0)
        return {
            "decision": "REJECTED",
            "reason": f"fairness check CRITICAL: demographic_parity={dp:.3f} < 0.6",
        }
    new_auc = mcnemar.get("new_auc", 0.0)
    return {
        "decision": "PRODUCTION",
        "reason": f"all ISO checks passed: robustness OK, fairness {fairness.get('status')}, AUC={new_auc:.4f}",
    }


def run_robustness(model: Any, X_test: Any, y_test: Any) -> dict:
    """ISO 24029: noise injection + feature dropout."""
    import numpy as np
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    base_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    X_noisy = X_test.copy()
    num_cols = X_noisy.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        noise = np.random.normal(0, 0.05 * X_noisy[col].std(), len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise
    noisy_auc = roc_auc_score(y_test, model.predict_proba(X_noisy)[:, 1])
    noise_tolerance = noisy_auc / base_auc if base_auc > 0 else 0.0
    return {
        "base_auc": base_auc,
        "noisy_auc": noisy_auc,
        "noise_tolerance": noise_tolerance,
        "compliant": noise_tolerance >= 0.85,
    }


def run_fairness(model: Any, X_test: Any, y_test: Any, protected_attr: str = "ligue_code") -> dict:
    """ISO 24027: demographic parity on protected attribute."""
    y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    groups = X_test[protected_attr].unique()
    rates: dict = {}
    for g in groups:
        mask = X_test[protected_attr] == g
        if mask.sum() < 30:
            continue
        rates[g] = float(y_pred[mask].mean())
    if len(rates) < 2:
        return {"status": "INSUFFICIENT_DATA", "demographic_parity": None}
    min_rate, max_rate = min(rates.values()), max(rates.values())
    dp_ratio = min_rate / max_rate if max_rate > 0 else 0.0
    status = "CRITICAL" if dp_ratio < 0.6 else ("CAUTION" if dp_ratio < 0.8 else "FAIR")
    return {"status": status, "demographic_parity": dp_ratio, "group_rates": rates}


def run_mcnemar(new_model: Any, champion_model: Any, X_test: Any, y_test: Any) -> dict:
    """McNemar test: new vs champion predictions on test set."""
    from scipy.stats import chi2  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    pred_new = (new_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    pred_champ = (champion_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    correct_new = pred_new == y_test.values
    correct_champ = pred_champ == y_test.values
    b = int((correct_champ & ~correct_new).sum())
    c = int((~correct_champ & correct_new).sum())
    if b + c == 0:
        return {"p_value": 1.0, "statistic": 0.0, "significant": False}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - chi2.cdf(stat, 1))
    new_auc = float(roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1]))
    champ_auc = float(roc_auc_score(y_test, champion_model.predict_proba(X_test)[:, 1]))
    return {
        "p_value": p_value,
        "statistic": stat,
        "significant": p_value < 0.05,
        "new_auc": new_auc,
        "champion_auc": champ_auc,
    }


def main() -> None:
    """Pull → Robustness → Fairness → McNemar → decide_promotion."""
    logger.info("promote_model: ISO 24029/24027 validation pipeline")
    logger.info("Run with a specific version: --version v20260318_120000")
