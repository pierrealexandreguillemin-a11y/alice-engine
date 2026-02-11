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
Last Updated: 2026-02-10
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from scripts.fairness.auto_report.types import (
    AttributeAnalysis,
    ComprehensiveFairnessReport,
    GroupMetrics,
)
from scripts.fairness.metrics import compute_bias_metrics_by_group

if TYPE_CHECKING:
    import pandas as pd

    from scripts.fairness.protected.types import ProtectedAttribute
    from scripts.fairness.types import BiasMetrics

logger = logging.getLogger(__name__)

# Seuils EEOC 80% rule
_EEOC_THRESHOLD = 0.80
_CAUTION_THRESHOLD = 0.85
_BOOTSTRAP_N = 1000
_BOOTSTRAP_CI = 0.95
# Seuils pour toutes les metriques (utilises par _status_from_metrics)
_FPR_DIFF_CRITICAL = 0.20
_FPR_DIFF_CAUTION = 0.10
_PP_DIFF_CRITICAL = 0.20
_PP_DIFF_CAUTION = 0.10
_TPR_DIFF_CRITICAL = 0.20
_TPR_DIFF_CAUTION = 0.10
_MIN_ACC_CRITICAL = 0.50
_MIN_ACC_CAUTION = 0.60
_CAL_GAP_CRITICAL = 0.20
_CAL_GAP_CAUTION = 0.10


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

    overall_status = _determine_overall_status(analyses)
    recommendations = _generate_recommendations(analyses)
    has_analyses = len(analyses) > 0

    return ComprehensiveFairnessReport(
        model_name=model_name,
        model_version=model_version,
        timestamp=datetime.now(tz=UTC).isoformat(),
        total_samples=len(y_true),
        analyses=analyses,
        overall_status=overall_status,
        recommendations=recommendations,
        iso_compliance={
            "iso_24027": has_analyses,
            "nist_ai_100_1": has_analyses and all(a.group_count >= 2 for a in analyses),
            "eu_ai_act_art13": has_analyses,
            "eeoc_80_percent": (
                has_analyses
                and all(a.demographic_parity_ratio >= _EEOC_THRESHOLD for a in analyses)
            ),
        },
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
    n_groups = len(unique_groups)

    if n_groups <= 1:
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

    # Reutilise le module fairness.metrics existant (ISO 5055)
    bias_metrics = compute_bias_metrics_by_group(y_true, y_pred, groups)

    # Construire GroupMetrics disaggreges + metriques supplementaires
    group_details = _build_group_details(y_true, y_pred, groups, bias_metrics)

    # Deriver metriques aggregees
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

    status = _status_from_metrics(dp_ratio, tpr_diff, fpr_diff, pp_diff, min_acc, max_cal)
    ci = _bootstrap_dp_ci(y_pred, groups, unique_groups)

    return AttributeAnalysis(
        attribute_name=attr_name,
        sample_count=len(y_true),
        group_count=n_groups,
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


def _build_group_details(
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


def _bootstrap_dp_ci(
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
        dp_ratio < _EEOC_THRESHOLD
        or tpr_diff > _TPR_DIFF_CRITICAL
        or fpr_diff > _FPR_DIFF_CRITICAL
        or pp_diff > _PP_DIFF_CRITICAL
        or min_acc < _MIN_ACC_CRITICAL
        or cal_gap > _CAL_GAP_CRITICAL
    ):
        return "critical"
    # Caution: any metric in caution zone
    if (
        dp_ratio < _CAUTION_THRESHOLD
        or tpr_diff > _TPR_DIFF_CAUTION
        or fpr_diff > _FPR_DIFF_CAUTION
        or pp_diff > _PP_DIFF_CAUTION
        or min_acc < _MIN_ACC_CAUTION
        or cal_gap > _CAL_GAP_CAUTION
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
