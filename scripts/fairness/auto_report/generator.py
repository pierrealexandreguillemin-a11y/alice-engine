"""Generateur de rapport fairness complet - ISO 24027.

Fonctions:
- generate_comprehensive_report: rapport multi-attributs
- _analyze_attribute: analyse par attribut (reutilise fairness.metrics)
- _determine_overall_status: status global
- _generate_recommendations: recommandations actionnables

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- NIST AI 100-1 MEASURE 2.11 - Fairness evaluation + calibration
- EU AI Act Art.13 - Transparency (disaggregated metrics)
- ISO/IEC 5055:2021 - Code Quality (SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-11
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from scripts.fairness.auto_report.computations import (
    bootstrap_dp_ci,
    build_group_details,
)
from scripts.fairness.auto_report.thresholds import (
    CAL_GAP_CAUTION,
    CAL_GAP_CRITICAL,
    CAUTION_DP_THRESHOLD,
    EEOC_THRESHOLD,
    FPR_DIFF_CAUTION,
    FPR_DIFF_CRITICAL,
    MIN_ACC_CAUTION,
    MIN_ACC_CRITICAL,
    PP_DIFF_CAUTION,
    PP_DIFF_CRITICAL,
    TPR_DIFF_CAUTION,
    TPR_DIFF_CRITICAL,
)
from scripts.fairness.auto_report.types import (
    AttributeAnalysis,
    ComprehensiveFairnessReport,
    GroupMetrics,
)
from scripts.fairness.metrics import compute_bias_metrics_by_group

if TYPE_CHECKING:
    import pandas as pd

    from scripts.fairness.protected.types import ProtectedAttribute

logger = logging.getLogger(__name__)


def generate_comprehensive_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_data: pd.DataFrame,
    model_name: str,
    model_version: str,
    protected_attributes: list[ProtectedAttribute],
) -> ComprehensiveFairnessReport:
    """Genere un rapport fairness complet multi-attributs.

    Model-agnostic: accepte y_true/y_pred numpy arrays.
    Reutilise compute_bias_metrics_by_group (ISO 5055 - pas de duplication).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        msg = f"y_true/y_pred length mismatch: {len(y_true)} vs {len(y_pred)}"
        raise ValueError(msg)

    analyses: list[AttributeAnalysis] = []
    for attr in protected_attributes:
        if attr.name not in test_data.columns:
            logger.warning("Attribute '%s' not in test_data", attr.name)
            continue
        groups = test_data[attr.name].values
        analysis = _analyze_attribute(y_true, y_pred, groups, attr.name)
        analyses.append(analysis)

    return ComprehensiveFairnessReport(
        model_name=model_name,
        model_version=model_version,
        timestamp=datetime.now(tz=UTC).isoformat(),
        total_samples=len(y_true),
        analyses=analyses,
        overall_status=_determine_overall_status(analyses),
        recommendations=_generate_recommendations(analyses),
        iso_compliance=_build_iso_compliance(analyses),
    )


def _analyze_attribute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    attr_name: str,
) -> AttributeAnalysis:
    """Analyse fairness pour un attribut protege."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    if len(unique_groups) <= 1:
        return _single_group_analysis(y_true, y_pred, attr_name, len(unique_groups))

    # Reutilise le module fairness.metrics existant (ISO 5055)
    bias_metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)
    group_details = build_group_details(y_true, y_pred, groups, bias_metrics)
    dp_ratio, tpr_diff, fpr_diff, pp_diff, min_acc, max_cal = _aggregate_from_groups(
        group_details,
    )

    status = _status_from_metrics(dp_ratio, tpr_diff, fpr_diff, pp_diff, min_acc, max_cal)
    ci = bootstrap_dp_ci(y_pred, groups, unique_groups)

    return AttributeAnalysis(
        attribute_name=attr_name,
        sample_count=len(y_true),
        group_count=len(unique_groups),
        demographic_parity_ratio=round(dp_ratio, 4),
        equalized_odds_tpr_diff=round(tpr_diff, 4),
        equalized_odds_fpr_diff=round(fpr_diff, 4),
        predictive_parity_diff=round(pp_diff, 4),
        min_group_accuracy=round(float(min_acc), 4),
        max_calibration_gap=round(float(max_cal), 4),
        status=status,
        group_details=group_details,
        confidence_intervals=ci,
    )


def _single_group_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attr_name: str,
    n_groups: int,
) -> AttributeAnalysis:
    """Analyse pour un attribut avec un seul groupe (edge case)."""
    acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    actual_rate = float(np.mean(y_true)) if len(y_true) > 0 else 0.0
    pred_rate = float(np.mean(y_pred)) if len(y_pred) > 0 else 0.0
    return AttributeAnalysis(
        attribute_name=attr_name,
        sample_count=len(y_true),
        group_count=n_groups,
        demographic_parity_ratio=1.0,
        equalized_odds_tpr_diff=0.0,
        equalized_odds_fpr_diff=0.0,
        predictive_parity_diff=0.0,
        min_group_accuracy=round(acc, 4),
        max_calibration_gap=round(abs(pred_rate - actual_rate), 4),
        status="fair",
    )


def _aggregate_from_groups(
    group_details: list[GroupMetrics],
) -> tuple[float, float, float, float, float, float]:
    """Calcule les metriques aggregees depuis les GroupMetrics."""
    rates = [g.positive_rate for g in group_details]
    tprs = [g.tpr for g in group_details]
    fprs = [g.fpr for g in group_details]
    precisions = [g.precision for g in group_details]
    accuracies = [g.accuracy for g in group_details]
    cal_gaps = [g.calibration_gap for g in group_details]

    dp_ratio = min(rates) / max(rates) if max(rates) > 0 else 1.0
    tpr_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)
    pp_diff = max(precisions) - min(precisions) if precisions else 0.0
    min_acc = min(accuracies) if accuracies else 0.0
    max_cal = max(cal_gaps) if cal_gaps else 0.0
    return dp_ratio, tpr_diff, fpr_diff, pp_diff, min_acc, max_cal


def _status_from_metrics(
    dp_ratio: float,
    tpr_diff: float,
    fpr_diff: float,
    pp_diff: float,
    min_acc: float,
    cal_gap: float = 0.0,
) -> str:
    """Determine le status a partir de TOUTES les metriques (NIST AI 100-1)."""
    # Critical: any metric exceeds critical threshold
    if (
        dp_ratio < EEOC_THRESHOLD
        or tpr_diff > TPR_DIFF_CRITICAL
        or fpr_diff > FPR_DIFF_CRITICAL
        or pp_diff > PP_DIFF_CRITICAL
        or min_acc < MIN_ACC_CRITICAL
        or cal_gap > CAL_GAP_CRITICAL
    ):
        return "critical"
    # Caution: any metric in caution zone
    if (
        dp_ratio < CAUTION_DP_THRESHOLD
        or tpr_diff > TPR_DIFF_CAUTION
        or fpr_diff > FPR_DIFF_CAUTION
        or pp_diff > PP_DIFF_CAUTION
        or min_acc < MIN_ACC_CAUTION
        or cal_gap > CAL_GAP_CAUTION
    ):
        return "caution"
    return "fair"


def _determine_overall_status(analyses: list[AttributeAnalysis]) -> str:
    """Determine le status global depuis les analyses."""
    if not analyses:
        return "fair"
    statuses = [a.status for a in analyses]
    if "critical" in statuses:
        return "critical"
    if "caution" in statuses:
        return "caution"
    return "fair"


def _generate_recommendations(analyses: list[AttributeAnalysis]) -> list[str]:
    """Genere des recommandations actionnables et specifiques."""
    recs: list[str] = []
    for a in analyses:
        if a.status == "critical":
            details = _build_rec_details(a)
            recs.append(
                f"URGENT: '{a.attribute_name}' shows critical bias. {details} "
                "Apply pre/in/post-processing mitigation."
            )
        elif a.status == "caution":
            details = _build_rec_details(a)
            recs.append(
                f"MONITOR: '{a.attribute_name}' shows moderate bias. {details} "
                "Increase monitoring frequency."
            )
    if not recs:
        recs.append("No significant bias detected. Maintain regular monitoring.")
    return recs


def _build_rec_details(a: AttributeAnalysis) -> str:
    """Construit les details des metriques pour une recommandation."""
    parts = [f"DP={a.demographic_parity_ratio:.2f}"]
    if a.equalized_odds_tpr_diff > 0.1:
        parts.append(f"TPR_diff={a.equalized_odds_tpr_diff:.2f}")
    if a.equalized_odds_fpr_diff > 0.1:
        parts.append(f"FPR_diff={a.equalized_odds_fpr_diff:.2f}")
    if a.predictive_parity_diff > 0.1:
        parts.append(f"PP_diff={a.predictive_parity_diff:.2f}")
    if a.min_group_accuracy < 0.6:
        parts.append(f"min_acc={a.min_group_accuracy:.2f}")
    return f"({', '.join(parts)})"


def _build_iso_compliance(
    analyses: list[AttributeAnalysis],
) -> dict[str, bool]:
    """Construit la section ISO compliance du rapport."""
    has_analyses = len(analyses) > 0
    return {
        "iso_24027": has_analyses,
        "nist_ai_100_1": has_analyses and all(a.group_count >= 2 for a in analyses),
        "eu_ai_act_art13": has_analyses,
        "eeoc_80_percent": (
            has_analyses and all(a.demographic_parity_ratio >= EEOC_THRESHOLD for a in analyses)
        ),
    }
