"""Generation de rapport de biais - ISO 24027.

Ce module contient les fonctions de generation de rapport:
- generate_fairness_report: Rapport complet ISO 24027
- _generate_recommendations: Recommandations d'action

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 42001:2023 - AI Management System (tracabilite)
- ISO/IEC 5055:2021 - Code Quality (<150 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd  # noqa: TCH002 - Used at runtime for type coercion

from scripts.fairness.checks import check_bias_thresholds
from scripts.fairness.metrics import compute_bias_metrics_by_group
from scripts.fairness.thresholds import BiasThresholds
from scripts.fairness.types import BiasLevel, BiasMetrics, BiasReport

logger = logging.getLogger(__name__)


def _generate_recommendations(
    metrics: list[BiasMetrics],
    overall_level: BiasLevel,
) -> list[str]:
    """Genere des recommandations basees sur les metriques."""
    recommendations = []

    if overall_level == BiasLevel.CRITICAL:
        recommendations.append(
            "URGENT: Biais critique detecte. Suspendre deploiement et investiguer."
        )

    # Analyser les groupes problematiques
    critical_groups = [m for m in metrics if m.level == BiasLevel.CRITICAL]
    warning_groups = [m for m in metrics if m.level == BiasLevel.WARNING]

    if critical_groups:
        group_names = ", ".join(m.group_name for m in critical_groups)
        recommendations.append(
            f"Groupes critiques: {group_names}. Reequilibrer donnees d'entrainement."
        )

    if warning_groups:
        group_names = ", ".join(m.group_name for m in warning_groups)
        recommendations.append(
            f"Groupes a surveiller: {group_names}. Monitoring renforce recommande."
        )

    # Recommandations specifiques
    recommendations.extend(_generate_dir_recommendations(metrics))

    if overall_level == BiasLevel.ACCEPTABLE and not recommendations:
        recommendations.append("Aucun biais significatif detecte. Maintenir monitoring regulier.")

    return recommendations


def _generate_dir_recommendations(metrics: list[BiasMetrics]) -> list[str]:
    """Genere recommandations specifiques pour DIR."""
    recommendations = []

    for m in metrics:
        if m.dir < 0.8:
            recommendations.append(
                f"Groupe '{m.group_name}': sous-represente (DIR={m.dir:.2f}). "
                "Considerer oversampling ou ajustement des seuils."
            )
        elif m.dir > 1.25:
            recommendations.append(
                f"Groupe '{m.group_name}': sur-represente (DIR={m.dir:.2f}). "
                "Considerer undersampling ou post-processing."
            )

    return recommendations


def generate_fairness_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    model_name: str = "Unknown",
    feature_name: str = "Unknown",
    reference_group: str | None = None,
    thresholds: BiasThresholds | None = None,
) -> BiasReport:
    """Genere un rapport de fairness complet ISO 24027.

    Args:
    ----
        y_true: Labels reels
        y_pred: Predictions
        groups: Groupes d'appartenance
        model_name: Nom du modele
        feature_name: Nom de la feature analysee
        reference_group: Groupe de reference
        thresholds: Seuils de biais

    Returns:
    -------
        BiasReport complet avec metriques et recommandations

    ISO 24027: Rapport de biais tracable.
    ISO 42001: Documentation des decisions AI.
    """
    logger.info(f"Generation rapport fairness: {model_name}, feature={feature_name}")

    # Calculer metriques
    metrics = compute_bias_metrics_by_group(
        y_true=y_true,
        y_pred=y_pred,
        groups=groups,
        reference_group=reference_group,
        thresholds=thresholds,
    )

    # Verifier seuils
    overall_level, alerts = check_bias_thresholds(metrics, thresholds)

    # Generer recommandations
    recommendations = _generate_recommendations(metrics, overall_level)

    # Determiner le groupe de reference utilise
    if reference_group is None:
        groups_arr = np.asarray(groups)
        unique_groups = np.unique(groups_arr)
        group_sizes = {g: np.sum(groups_arr == g) for g in unique_groups}
        reference_group = str(max(group_sizes, key=group_sizes.get))

    # Creer le rapport
    report = BiasReport(
        timestamp=datetime.now().isoformat(),
        model_name=model_name,
        total_samples=len(y_true),
        feature_analyzed=feature_name,
        reference_group=reference_group,
        metrics_by_group=metrics,
        overall_level=overall_level,
        recommendations=recommendations,
        iso_compliance={
            "iso_24027": True,
            "iso_42001": True,
            "eeoc_4_5_rule": all(0.8 <= m.dir <= 1.25 for m in metrics if m.dir != float("inf")),
            "alerts": alerts,
        },
    )

    logger.info(f"  Niveau global: {overall_level.value}")
    logger.info(f"  {len(alerts)} alertes generees")

    return report
