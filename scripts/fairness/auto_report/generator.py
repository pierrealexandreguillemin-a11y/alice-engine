"""Generateur de rapport fairness complet - ISO 24027.

Fonctions:
- generate_comprehensive_report: rapport multi-attributs
- _analyze_attribute: analyse par attribut (reutilise fairness.metrics)
- _determine_overall_status: status global
- _generate_recommendations: recommandations actionnables

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- NIST AI 100-1 MEASURE 2.11 - Fairness evaluation
- EU AI Act Art.13 - Transparency (disaggregated metrics)
- ISO/IEC 5055:2021 - Code Quality (<250 lignes, SRP)

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
_BOOTSTRAP_N = 200
_BOOTSTRAP_CI = 0.95


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

    Args:
    ----
        y_true: Labels reels (0/1)
        y_pred: Predictions (0/1)
        test_data: DataFrame avec colonnes des attributs proteges
        model_name: Nom du modele
        model_version: Version du modele
        protected_attributes: Liste des attributs a analyser

    Returns:
    -------
        ComprehensiveFairnessReport avec analyses et recommandations
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

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

    return ComprehensiveFairnessReport(
        model_name=model_name,
        model_version=model_version,
        timestamp=datetime.now(tz=UTC).isoformat(),
        total_samples=len(y_true),
        analyses=analyses,
        overall_status=overall_status,
        recommendations=recommendations,
        iso_compliance={
            "iso_24027": len(analyses) > 0,
            "nist_ai_100_1": len(analyses) > 0 and all(a.group_count >= 2 for a in analyses),
            "eu_ai_act_art13": len(analyses) > 0,
            "eeoc_80_percent": all(a.demographic_parity_ratio >= _EEOC_THRESHOLD for a in analyses),
        },
    )


def _analyze_attribute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    attr_name: str,
) -> AttributeAnalysis:
    """Analyse fairness pour un attribut protege.

    Reutilise compute_bias_metrics_by_group() pour les metriques par groupe,
    puis derive les metriques aggregees (DP ratio, equalized odds, etc.).
    """
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups <= 1:
        return AttributeAnalysis(
            attribute_name=attr_name,
            sample_count=len(y_true),
            group_count=n_groups,
            demographic_parity_ratio=1.0,
            equalized_odds_tpr_diff=0.0,
            equalized_odds_fpr_diff=0.0,
            predictive_parity_diff=0.0,
            min_group_accuracy=1.0,
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

    dp_ratio = min(rates) / max(rates) if max(rates) > 0 else 1.0
    tpr_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)
    pp_diff = max(precisions) - min(precisions) if precisions else 0.0
    min_acc = min(accuracies) if accuracies else 1.0

    status = _status_from_metrics(dp_ratio, tpr_diff)
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
    """Construit les GroupMetrics a partir des BiasMetrics existants.

    Enrichit avec FPR, precision, accuracy non fournis par BiasMetrics.
    """
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

        details.append(
            GroupMetrics(
                group_name=str(bm.group_name),
                sample_count=bm.group_size,
                positive_rate=bm.positive_rate,
                tpr=bm.true_positive_rate,
                fpr=round(fpr, 4),
                precision=round(precision, 4),
                accuracy=round(accuracy, 4),
            )
        )
    return details


def _bootstrap_dp_ci(
    y_pred: np.ndarray,
    groups: np.ndarray,
    unique_groups: np.ndarray,
) -> dict[str, list[float]]:
    """Bootstrap CI pour le demographic parity ratio (NIST AI 100-1)."""
    rng = np.random.default_rng(42)
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


def _status_from_metrics(dp_ratio: float, tpr_diff: float) -> str:
    """Determine le status a partir des metriques."""
    if dp_ratio < _EEOC_THRESHOLD or tpr_diff > 0.2:
        return "critical"
    if dp_ratio < _CAUTION_THRESHOLD or tpr_diff > 0.1:
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
    """Genere des recommandations actionnables."""
    recs: list[str] = []
    for a in analyses:
        if a.status == "critical":
            recs.append(
                f"URGENT: '{a.attribute_name}' shows critical bias "
                f"(DP={a.demographic_parity_ratio:.2f}). "
                "Apply pre/in/post-processing mitigation."
            )
        elif a.status == "caution":
            recs.append(
                f"MONITOR: '{a.attribute_name}' shows moderate bias "
                f"(DP={a.demographic_parity_ratio:.2f}). "
                "Increase monitoring frequency."
            )
    if not recs:
        recs.append("Maintenir monitoring regulier. Aucun biais significatif.")
    return recs
