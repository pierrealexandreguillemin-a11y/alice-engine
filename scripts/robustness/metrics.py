"""Calcul des metriques de robustesse - ISO 24029.

Ce module contient la fonction compute_robustness_metrics qui execute
la suite complete des tests de robustesse.

ISO Compliance:
- ISO/IEC 24029-1:2021 - Neural Network Robustness Assessment
- ISO/IEC 24029-2:2023 - Robustness Testing Methodology
- ISO/IEC 5055:2021 - Code Quality (<100 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd  # noqa: TCH002 - Used at runtime for type coercion

from scripts.robustness.perturbations import (
    run_extreme_values_test,
    run_feature_perturbation_test,
    run_noise_test,
    run_out_of_distribution_test,
)
from scripts.robustness.thresholds import DEFAULT_THRESHOLDS, RobustnessThresholds
from scripts.robustness.types import RobustnessMetrics

logger = logging.getLogger(__name__)


def compute_robustness_metrics(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    thresholds: RobustnessThresholds | None = None,
) -> list[RobustnessMetrics]:
    """Execute la suite complete de tests de robustesse.

    Args:
    ----
        model_predict: Fonction de prediction du modele
        X: Features d'entree
        y_true: Labels reels
        thresholds: Seuils de robustesse

    Returns:
    -------
        Liste de RobustnessMetrics pour chaque test

    ISO 24029: Suite complete de tests de robustesse.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    X = np.asarray(X)
    y_true = np.asarray(y_true)

    logger.info("Execution tests de robustesse ISO 24029...")

    metrics = []

    # Test 1: Bruit gaussien
    for noise_level in thresholds.noise_levels:
        result = run_noise_test(
            model_predict, X, y_true, noise_std=noise_level, thresholds=thresholds
        )
        metrics.append(result)

    # Test 2: Perturbation de features
    for perturb_level in thresholds.perturbation_levels:
        result = run_feature_perturbation_test(
            model_predict, X, y_true, perturbation=perturb_level, thresholds=thresholds
        )
        metrics.append(result)

    # Test 3: Out-of-Distribution
    result = run_out_of_distribution_test(model_predict, X, y_true, thresholds=thresholds)
    metrics.append(result)

    # Test 4: Valeurs extremes
    result = run_extreme_values_test(model_predict, X, y_true, thresholds=thresholds)
    metrics.append(result)

    logger.info(f"  {len(metrics)} tests executes")
    return metrics
