"""Drift Monitor - ISO 23894 (AI Risk Management).

Module principal de monitoring du drift pour ML en production.

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<200 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from scripts.model_registry.drift_stats import (
    compute_chi2_test,
    compute_js_divergence,
    compute_ks_test,
    compute_psi,
)
from scripts.model_registry.drift_types import (
    KS_PVALUE_CRITICAL,
    KS_PVALUE_OK,
    KS_PVALUE_WARNING,
    PSI_THRESHOLD_CRITICAL,
    PSI_THRESHOLD_OK,
    PSI_THRESHOLD_WARNING,
    DriftMonitorResult,
    DriftSeverity,
    DriftType,
    FeatureDriftResult,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


def _determine_categorical_severity(chi2_pvalue: float) -> DriftSeverity:
    """Détermine la sévérité pour les features catégorielles."""
    thresholds = [
        (0.001, DriftSeverity.CRITICAL),
        (0.01, DriftSeverity.HIGH),
        (0.05, DriftSeverity.MEDIUM),
    ]
    return next((sev for thresh, sev in thresholds if chi2_pvalue < thresh), DriftSeverity.NONE)


def _determine_numeric_severity(psi: float, ks_pvalue: float | None) -> DriftSeverity:
    """Détermine la sévérité pour les features numériques."""
    if psi >= PSI_THRESHOLD_CRITICAL:
        return DriftSeverity.CRITICAL
    if psi >= PSI_THRESHOLD_WARNING:
        return DriftSeverity.HIGH
    if psi >= PSI_THRESHOLD_OK:
        return (
            DriftSeverity.MEDIUM
            if ks_pvalue and ks_pvalue < KS_PVALUE_WARNING
            else DriftSeverity.LOW
        )
    return DriftSeverity.LOW if ks_pvalue and ks_pvalue < KS_PVALUE_OK else DriftSeverity.NONE


def _determine_severity(
    psi: float,
    ks_pvalue: float | None = None,
    is_categorical: bool = False,
    chi2_pvalue: float | None = None,
) -> DriftSeverity:
    """Détermine la sévérité du drift."""
    if is_categorical and chi2_pvalue is not None:
        return _determine_categorical_severity(chi2_pvalue)
    return _determine_numeric_severity(psi, ks_pvalue)


def analyze_feature_drift(
    feature_name: str,
    baseline: np.ndarray,
    current: np.ndarray,
    is_categorical: bool = False,
) -> FeatureDriftResult:
    """Analyse le drift pour une feature."""
    result = FeatureDriftResult(
        feature_name=feature_name, psi_score=0.0, is_categorical=is_categorical
    )

    if is_categorical:
        chi2, chi2_pvalue = compute_chi2_test(baseline, current)
        result.chi2_statistic = chi2
        result.chi2_pvalue = chi2_pvalue
    else:
        result.psi_score = compute_psi(baseline, current)
        ks_stat, ks_pvalue = compute_ks_test(baseline, current)
        result.ks_statistic = ks_stat
        result.ks_pvalue = ks_pvalue
        result.js_divergence = compute_js_divergence(baseline, current)

    result.severity = _determine_severity(
        psi=result.psi_score,
        ks_pvalue=result.ks_pvalue,
        is_categorical=is_categorical,
        chi2_pvalue=result.chi2_pvalue,
    )
    return result


def _generate_recommendations(result: DriftMonitorResult) -> list[str]:
    """Génère des recommandations basées sur le drift."""
    messages = {
        DriftSeverity.CRITICAL: "URGENT: Drift critique. Retraining immédiat recommandé.",
        DriftSeverity.HIGH: "WARNING: Drift significatif. Planifier un retraining.",
        DriftSeverity.MEDIUM: "NOTICE: Drift modéré. Surveiller l'évolution.",
    }
    recs = [messages[result.overall_severity]] if result.overall_severity in messages else []
    if result.drifted_features:
        recs.append(f"Features impactées: {', '.join(result.drifted_features[:5])}")
    return recs


def _compute_overall_severity(severities: list[DriftSeverity]) -> DriftSeverity:
    """Calcule la sévérité globale depuis la liste des sévérités.

    Retourne la sévérité la plus haute présente dans la liste,
    en suivant l'ordre: CRITICAL > HIGH > MEDIUM > LOW > NONE.

    Args:
    ----
        severities: Liste des sévérités de drift par feature.

    Returns:
    -------
        DriftSeverity: La sévérité maximale, ou NONE si liste vide.
    """
    severity_order = [
        DriftSeverity.CRITICAL,
        DriftSeverity.HIGH,
        DriftSeverity.MEDIUM,
        DriftSeverity.LOW,
    ]
    return next((s for s in severity_order if s in severities), DriftSeverity.NONE)


def _is_significant_drift(severity: DriftSeverity) -> bool:
    """Vérifie si la sévérité est significative.

    Une sévérité est considérée significative si elle nécessite
    une attention ou une action (MEDIUM, HIGH, CRITICAL).

    Args:
    ----
        severity: Niveau de sévérité du drift à évaluer.

    Returns:
    -------
        bool: True si MEDIUM, HIGH ou CRITICAL; False sinon.
    """
    return severity in (DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL)


def monitor_drift(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    model_version: str,
    categorical_features: list[str] | None = None,
    features_to_monitor: list[str] | None = None,
) -> DriftMonitorResult:
    """Monitore le drift sur l'ensemble des features (ISO 23894)."""
    categorical_features = categorical_features or []
    features = features_to_monitor or [
        c for c in baseline_data.columns if c in current_data.columns
    ]

    result = DriftMonitorResult(
        timestamp=datetime.now().isoformat(),
        model_version=model_version,
        baseline_samples=len(baseline_data),
        current_samples=len(current_data),
    )

    for feature in features:
        if feature not in current_data.columns:
            continue
        feature_result = analyze_feature_drift(
            feature,
            baseline_data[feature].values,
            current_data[feature].values,
            is_categorical=feature in categorical_features,
        )
        result.feature_results.append(feature_result)
        if _is_significant_drift(feature_result.severity):
            result.drifted_features.append(feature)

    if result.feature_results:
        severities = [f.severity for f in result.feature_results]
        result.overall_severity = _compute_overall_severity(severities)

    result.drift_detected = result.overall_severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)
    result.recommendations = _generate_recommendations(result)
    return result


__all__ = [
    "PSI_THRESHOLD_OK",
    "PSI_THRESHOLD_WARNING",
    "PSI_THRESHOLD_CRITICAL",
    "KS_PVALUE_OK",
    "KS_PVALUE_WARNING",
    "KS_PVALUE_CRITICAL",
    "DriftSeverity",
    "DriftType",
    "FeatureDriftResult",
    "DriftMonitorResult",
    "compute_psi",
    "compute_ks_test",
    "compute_chi2_test",
    "compute_js_divergence",
    "analyze_feature_drift",
    "monitor_drift",
]
