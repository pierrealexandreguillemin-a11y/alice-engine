"""Calculs internes pour le rapport fairness - ISO 24027.

Fonctions:
- build_group_details: construit GroupMetrics disaggreges
- bootstrap_dp_ci: bootstrap CI pour demographic parity ratio

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- NIST AI 100-1 MEASURE 2.11 - Confidence intervals
- ISO/IEC 5055:2021 - Code Quality (SRP, <300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-02-11
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scripts.fairness.auto_report.types import GroupMetrics

if TYPE_CHECKING:
    from scripts.fairness.types import BiasMetrics

# NIST AI 100-1: 1000 bootstrap samples pour CI 95% stables
_BOOTSTRAP_N = 1000
_BOOTSTRAP_CI = 0.95


def build_group_details(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    bias_metrics: list[BiasMetrics],
) -> list[GroupMetrics]:
    """Construit les GroupMetrics a partir des BiasMetrics existants."""
    details: list[GroupMetrics] = []
    for bm in bias_metrics:
        mask = groups == bm.group_name
        yt, yp = y_true[mask], y_pred[mask]
        negatives = int(np.sum(yt == 0))
        pred_pos = int(np.sum(yp == 1))
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fpr = float(np.sum((yt == 0) & (yp == 1)) / negatives) if negatives > 0 else 0.0
        precision = float(tp / pred_pos) if pred_pos > 0 else 0.0
        accuracy = float(np.sum(yt == yp) / len(yt)) if len(yt) > 0 else 0.0
        actual_rate = float(np.mean(yt)) if len(yt) > 0 else 0.0
        cal_gap = abs(bm.positive_rate - actual_rate)
        group_name = str(bm.group_name) if str(bm.group_name) else "(empty)"

        details.append(
            GroupMetrics(
                group_name=group_name,
                sample_count=bm.group_size,
                positive_rate=bm.positive_rate,
                tpr=bm.true_positive_rate,
                fpr=round(fpr, 4),
                precision=round(precision, 4),
                accuracy=round(accuracy, 4),
                calibration_gap=round(cal_gap, 4),
            )
        )
    return details


def bootstrap_dp_ci(
    y_pred: np.ndarray,
    groups: np.ndarray,
    unique_groups: np.ndarray,
    *,
    seed: int | None = 42,
) -> dict[str, list[float]]:
    """Bootstrap CI pour le demographic parity ratio (NIST AI 100-1)."""
    rng = np.random.default_rng(seed)
    n = len(y_pred)
    dp_ratios: list[float] = []

    for _ in range(_BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        bp, bg = y_pred[idx], groups[idx]
        rates = []
        for g in unique_groups:
            mask = bg == g
            if np.sum(mask) > 0:
                rates.append(float(np.mean(bp[mask])))
        if rates and max(rates) > 0:
            dp_ratios.append(min(rates) / max(rates))

    if not dp_ratios:
        return {}

    alpha = (1 - _BOOTSTRAP_CI) / 2
    lo = float(np.percentile(dp_ratios, alpha * 100))
    hi = float(np.percentile(dp_ratios, (1 - alpha) * 100))
    return {"demographic_parity_ratio": [round(lo, 4), round(hi, 4)]}
