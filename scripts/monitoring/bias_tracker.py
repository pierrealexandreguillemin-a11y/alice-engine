"""Bias Tracker - ISO 24027 (Bias in AI).

Module de monitoring continu du biais en production ML.

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias Detection in AI
- ISO/IEC 5055:2021 - Code Quality (<200 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.monitoring.bias_types import (
    CALIBRATION_THRESHOLD,
    DEMOGRAPHIC_PARITY_THRESHOLD,
    DISPARATE_IMPACT_THRESHOLD,
    EQUALIZED_ODDS_THRESHOLD,
    BiasAlert,
    BiasAlertLevel,
    BiasMetrics,
    BiasMonitorConfig,
    BiasMonitorResult,
    FairnessStatus,
)

logger = logging.getLogger(__name__)


def _compute_rates(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict[str, float | None]:
    """Calcule les taux pour un groupe."""
    n = len(y_true)
    if n == 0:
        return {"positive_rate": 0.0, "tpr": None, "fpr": None, "calibration": None}

    positive_rate = float(np.mean(y_pred == 1))
    positives, negatives = y_true == 1, y_true == 0
    tpr = float(np.mean(y_pred[positives] == 1)) if positives.sum() > 0 else None
    fpr = float(np.mean(y_pred[negatives] == 1)) if negatives.sum() > 0 else None
    calibration = float(abs(np.mean(y_prob) - np.mean(y_true == 1))) if y_prob is not None else None
    return {"positive_rate": positive_rate, "tpr": tpr, "fpr": fpr, "calibration": calibration}


def compute_bias_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    attribute_name: str,
    y_prob: np.ndarray | None = None,
    min_group_size: int = 30,
) -> list[BiasMetrics]:
    """Calcule les métriques de biais par groupe."""
    metrics = []
    for group in np.unique(protected):
        mask = protected == group
        if mask.sum() < min_group_size:
            continue
        rates = _compute_rates(
            y_true[mask], y_pred[mask], y_prob[mask] if y_prob is not None else None
        )
        metrics.append(
            BiasMetrics(
                group_name=attribute_name,
                group_value=str(group),
                n_samples=int(mask.sum()),
                positive_rate=rates["positive_rate"],
                true_positive_rate=rates["tpr"],
                false_positive_rate=rates["fpr"],
                calibration=rates["calibration"],
            )
        )
    return metrics


def check_bias_alerts(result: BiasMonitorResult, config: BiasMonitorConfig) -> list[BiasAlert]:
    """Vérifie les alertes de biais selon les seuils."""
    alerts = []
    if result.demographic_parity < config.demographic_parity_threshold:
        level = (
            BiasAlertLevel.CRITICAL if result.demographic_parity < 0.6 else BiasAlertLevel.WARNING
        )
        rates = [(m.group_value, m.positive_rate) for m in result.group_metrics]
        min_g, max_g = min(rates, key=lambda x: x[1])[0], max(rates, key=lambda x: x[1])[0]
        alerts.append(
            BiasAlert(
                metric_name="demographic_parity",
                level=level,
                message=f"DP {result.demographic_parity:.2f} < {config.demographic_parity_threshold}",
                value=result.demographic_parity,
                threshold=config.demographic_parity_threshold,
                affected_groups=[min_g, max_g],
            )
        )
    if result.equalized_odds_tpr and result.equalized_odds_tpr > config.equalized_odds_threshold:
        alerts.append(
            BiasAlert(
                metric_name="equalized_odds_tpr",
                level=BiasAlertLevel.WARNING,
                message=f"TPR diff {result.equalized_odds_tpr:.2f} > {config.equalized_odds_threshold}",
                value=result.equalized_odds_tpr,
                threshold=config.equalized_odds_threshold,
            )
        )
    return alerts


def _determine_status(alerts: list[BiasAlert]) -> FairnessStatus:
    """Détermine le statut global."""
    if not alerts:
        return FairnessStatus.FAIR
    levels = [a.level for a in alerts]
    if BiasAlertLevel.CRITICAL in levels:
        return FairnessStatus.CRITICAL
    if BiasAlertLevel.WARNING in levels:
        return FairnessStatus.BIASED
    return FairnessStatus.CAUTION


def monitor_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    protected_attribute: str,
    config: BiasMonitorConfig | None = None,
    y_prob: np.ndarray | None = None,
) -> BiasMonitorResult:
    """Monitore le biais en production (ISO 24027)."""
    config = config or BiasMonitorConfig()
    y_true, y_pred, protected = np.asarray(y_true), np.asarray(y_pred), np.asarray(protected)

    group_metrics = compute_bias_metrics(
        y_true, y_pred, protected, protected_attribute, y_prob, config.min_group_size
    )
    rates = [m.positive_rate for m in group_metrics]
    dp = min(rates) / max(rates) if rates and max(rates) > 0 else 1.0
    tprs = [m.true_positive_rate for m in group_metrics if m.true_positive_rate is not None]
    fprs = [m.false_positive_rate for m in group_metrics if m.false_positive_rate is not None]

    result = BiasMonitorResult(
        timestamp=datetime.now().isoformat(),
        protected_attribute=protected_attribute,
        n_total_samples=len(y_true),
        group_metrics=group_metrics,
        demographic_parity=dp,
        disparate_impact=dp,
        equalized_odds_tpr=max(tprs) - min(tprs) if len(tprs) >= 2 else None,
        equalized_odds_fpr=max(fprs) - min(fprs) if len(fprs) >= 2 else None,
    )

    result.alerts = check_bias_alerts(result, config)
    result.overall_status = _determine_status(result.alerts)
    if result.overall_status in (FairnessStatus.BIASED, FairnessStatus.CRITICAL):
        result.recommendations.append("ISO 24027: Investiguer les causes du biais détecté.")
    return result


def save_bias_config(config: BiasMonitorConfig, path: Path) -> None:
    """Sauvegarde la configuration en JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def load_bias_config(path: Path) -> BiasMonitorConfig | None:
    """Charge la configuration depuis JSON."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return BiasMonitorConfig.from_dict(json.load(f))


__all__ = [
    "DEMOGRAPHIC_PARITY_THRESHOLD",
    "DISPARATE_IMPACT_THRESHOLD",
    "EQUALIZED_ODDS_THRESHOLD",
    "CALIBRATION_THRESHOLD",
    "FairnessStatus",
    "BiasAlertLevel",
    "BiasMetrics",
    "BiasAlert",
    "BiasMonitorResult",
    "BiasMonitorConfig",
    "compute_bias_metrics",
    "check_bias_alerts",
    "monitor_bias",
    "save_bias_config",
    "load_bias_config",
]
