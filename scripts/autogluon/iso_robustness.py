"""Validation de robustesse - ISO 24029.

Ce module contient la validation de robustesse du modele.

ISO Compliance:
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC 5055:2021 - Code Quality (<80 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.autogluon.iso_types import ISO24029RobustnessReport

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor


def validate_robustness(
    predictor: TabularPredictor,
    test_data: Any,
    noise_level: float = 0.1,
) -> ISO24029RobustnessReport:
    """Valide la robustesse du modele selon ISO 24029.

    Args:
    ----
        predictor: TabularPredictor a evaluer
        test_data: Donnees de test
        noise_level: Niveau de bruit a appliquer (defaut 10%)

    Returns:
    -------
        ISO24029RobustnessReport

    ISO 24029: Tests de robustesse obligatoires.
    """
    model_id = str(predictor.path)

    # Evaluation baseline
    label = predictor.label
    y_true = test_data[label]
    X_test = test_data.drop(columns=[label])
    y_pred_baseline = predictor.predict(X_test)
    baseline_acc = (y_pred_baseline == y_true).mean()

    # Test avec bruit gaussien
    noisy_acc = _evaluate_noisy_accuracy(predictor, X_test, y_true, noise_level)

    # Calculer les metriques
    noise_tolerance = noisy_acc / baseline_acc if baseline_acc > 0 else 0
    status = _determine_robustness_status(noise_tolerance)

    return ISO24029RobustnessReport(
        model_id=model_id,
        noise_tolerance=float(noise_tolerance),
        adversarial_robustness=0.0,
        distribution_shift_score=0.0,
        confidence_calibration=0.0,
        status=status,
    )


def _evaluate_noisy_accuracy(predictor: Any, X_test: Any, y_true: Any, noise_level: float) -> float:
    """Evalue l'accuracy avec bruit gaussien."""
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    X_noisy = X_test.copy()

    for col in numeric_cols:
        noise = np.random.normal(0, noise_level * X_noisy[col].std(), len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise

    y_pred_noisy = predictor.predict(X_noisy)
    return (y_pred_noisy == y_true).mean()


def _determine_robustness_status(noise_tolerance: float) -> str:
    """Determine le statut de robustesse."""
    if noise_tolerance >= 0.95:
        return "ROBUST"
    if noise_tolerance >= 0.85:
        return "SENSITIVE"
    return "FRAGILE"
