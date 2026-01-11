"""Generation de rapport de robustesse - ISO 24029.

Ce module contient les fonctions de generation de rapport:
- generate_robustness_report: Rapport complet ISO 24029
- _determine_overall_level: Niveau global
- _generate_recommendations: Recommandations d'action

ISO Compliance:
- ISO/IEC 24029-1:2021 - Neural Network Robustness Assessment
- ISO/IEC 42001:2023 - AI Management System (tracabilite)
- ISO/IEC 5055:2021 - Code Quality (<200 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd  # noqa: TCH002 - Used at runtime for type coercion

from scripts.robustness.metrics import compute_robustness_metrics
from scripts.robustness.thresholds import RobustnessThresholds
from scripts.robustness.types import RobustnessLevel, RobustnessMetrics, RobustnessReport

logger = logging.getLogger(__name__)


def _compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'accuracy."""
    return float(np.mean(y_true == y_pred))


def _determine_overall_level(metrics: list[RobustnessMetrics]) -> RobustnessLevel:
    """Determine le niveau global a partir de tous les tests."""
    if not metrics:
        return RobustnessLevel.ACCEPTABLE

    # Le niveau global est le pire niveau parmi tous les tests
    levels = [m.level for m in metrics]

    if RobustnessLevel.FRAGILE in levels:
        return RobustnessLevel.FRAGILE
    if RobustnessLevel.WARNING in levels:
        return RobustnessLevel.WARNING
    if RobustnessLevel.ACCEPTABLE in levels:
        return RobustnessLevel.ACCEPTABLE
    return RobustnessLevel.ROBUST


def _generate_recommendations(
    metrics: list[RobustnessMetrics],
    overall_level: RobustnessLevel,
) -> list[str]:
    """Genere des recommandations basees sur les metriques."""
    recommendations = []

    _add_fragile_alert(recommendations, overall_level)
    _add_test_recommendations(recommendations, metrics)
    recommendations.extend(_generate_specific_recommendations(metrics))
    _add_robust_recommendation(recommendations, overall_level)

    return recommendations


def _add_fragile_alert(recommendations: list[str], level: RobustnessLevel) -> None:
    """Ajoute alerte fragile si necessaire."""
    if level == RobustnessLevel.FRAGILE:
        recommendations.append(
            "CRITIQUE: Modele fragile detecte. Ne pas deployer en production "
            "sans renforcement de la robustesse."
        )


def _add_test_recommendations(recommendations: list[str], metrics: list[RobustnessMetrics]) -> None:
    """Ajoute recommandations par type de test."""
    fragile_tests = [m for m in metrics if m.level == RobustnessLevel.FRAGILE]
    warning_tests = [m for m in metrics if m.level == RobustnessLevel.WARNING]

    if fragile_tests:
        test_names = ", ".join(m.test_name for m in fragile_tests)
        recommendations.append(
            f"Tests fragiles: {test_names}. Implementer data augmentation "
            "et/ou regularisation renforcee."
        )

    if warning_tests:
        test_names = ", ".join(m.test_name for m in warning_tests)
        recommendations.append(f"Tests warning: {test_names}. Monitoring renforce recommande.")


def _add_robust_recommendation(recommendations: list[str], level: RobustnessLevel) -> None:
    """Ajoute recommandation si modele robuste."""
    if level == RobustnessLevel.ROBUST:
        recommendations.append(
            "Modele robuste. Maintenir monitoring regulier et reevaluer "
            "apres chaque reentrainement."
        )


def _generate_specific_recommendations(
    metrics: list[RobustnessMetrics],
) -> list[str]:
    """Genere recommandations specifiques par type de test."""
    recommendations = []

    for m in metrics:
        if m.level != RobustnessLevel.FRAGILE:
            continue

        if "noise" in m.test_name:
            recommendations.append(
                f"Bruit sensible ({m.test_name}): Considerer ajout de "
                "bruit gaussien durant l'entrainement (noise injection)."
            )
        elif "perturbation" in m.test_name:
            recommendations.append(
                f"Perturbation sensible ({m.test_name}): Considerer "
                "feature selection ou dropout renforce."
            )
        elif "distribution" in m.test_name:
            recommendations.append(
                f"OOD sensible ({m.test_name}): Implementer detection OOD " "et mecanisme de rejet."
            )
        elif "extreme" in m.test_name:
            recommendations.append(
                f"Valeurs extremes ({m.test_name}): Considerer clipping "
                "des features ou normalisation robuste."
            )

    return recommendations


def generate_robustness_report(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    model_name: str = "Unknown",
    thresholds: RobustnessThresholds | None = None,
) -> RobustnessReport:
    """Genere un rapport de robustesse complet ISO 24029.

    Args:
    ----
        model_predict: Fonction de prediction
        X: Features
        y_true: Labels reels
        model_name: Nom du modele
        thresholds: Seuils

    Returns:
    -------
        RobustnessReport complet avec metriques et recommandations

    ISO 24029: Rapport de robustesse tracable.
    ISO 42001: Documentation des decisions AI.
    """
    logger.info(f"Generation rapport robustesse: {model_name}")

    X = np.asarray(X)
    y_true = np.asarray(y_true)

    # Calcul accuracy originale
    y_pred = model_predict(X)
    original_accuracy = _compute_accuracy(y_true, y_pred)

    # Executer tous les tests
    metrics = compute_robustness_metrics(model_predict, X, y_true, thresholds)

    # Determiner niveau global
    overall_level = _determine_overall_level(metrics)

    # Generer recommandations
    recommendations = _generate_recommendations(metrics, overall_level)

    report = RobustnessReport(
        timestamp=datetime.now().isoformat(),
        model_name=model_name,
        total_tests=len(metrics),
        original_accuracy=round(original_accuracy, 4),
        metrics=metrics,
        overall_level=overall_level,
        recommendations=recommendations,
        iso_compliance={
            "iso_24029_1": True,
            "iso_24029_2": True,
            "iso_42001": True,
            "all_tests_passed": all(
                m.level in (RobustnessLevel.ROBUST, RobustnessLevel.ACCEPTABLE) for m in metrics
            ),
            "fragile_tests": [m.test_name for m in metrics if m.level == RobustnessLevel.FRAGILE],
            "warning_tests": [m.test_name for m in metrics if m.level == RobustnessLevel.WARNING],
        },
    )

    logger.info(f"  Niveau global: {overall_level.value}")
    logger.info(
        f"  {len([m for m in metrics if m.level == RobustnessLevel.FRAGILE])} " "tests fragiles"
    )

    return report
