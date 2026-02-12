"""Validation de robustesse améliorée - ISO 24029.

Ce module implémente une analyse de robustesse complète selon ISO/IEC 24029:2021
avec tests formels, perturbations multiples, et analyse de stabilité.

ISO Compliance:
- ISO/IEC 24029-1:2021 - Robustness Overview
- ISO/IEC 24029-2:2023 - Formal Methods Methodology
- ISO/IEC 5055:2021 - Code Quality (<300 lignes, SRP)

Document ID: ALICE-SCRIPT-ISO24029-002
Version: 2.1.0
Author: ALICE Engine Team
Last Updated: 2026-02-12
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor

from scripts.autogluon.iso_robustness_enhanced_perturbations import (
    test_feature_dropout,
    test_monotonicity,
    test_noise_robustness,
    test_prediction_consistency,
)
from scripts.autogluon.iso_types import (
    FeatureDropoutTest,
    ISO24029EnhancedReport,
    MonotonicityTest,
    NoiseRobustnessTest,
    PredictionConsistencyTest,
)

# Re-export for backward compatibility
__all__ = [
    "FeatureDropoutTest",
    "ISO24029EnhancedReport",
    "MonotonicityTest",
    "NoiseRobustnessTest",
    "PredictionConsistencyTest",
    "validate_robustness_enhanced",
]


def validate_robustness_enhanced(
    predictor: TabularPredictor,
    test_data: Any,
    noise_levels: list[float] | None = None,
    n_consistency_samples: int = 100,
) -> ISO24029EnhancedReport:
    """Valide la robustesse avec analyse complète ISO 24029.

    Args:
    ----
        predictor: Modèle AutoGluon à évaluer
        test_data: Données de test avec label
        noise_levels: Niveaux de bruit à tester (défaut: [0.01, 0.05, 0.1])
        n_consistency_samples: Nombre d'échantillons pour test de consistance

    Returns:
    -------
        ISO24029EnhancedReport avec analyse complète

    ISO 24029-1: Tests de perturbation
    ISO 24029-2: Vérification formelle
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.1]

    label = predictor.label
    y_true = test_data[label].values
    X_test = test_data.drop(columns=[label])

    # Prédiction baseline
    y_pred_baseline = predictor.predict(X_test).values
    baseline_acc = (y_pred_baseline == y_true).mean()

    # Test 1: Robustesse au bruit (multi-niveaux)
    noise_tests = test_noise_robustness(predictor, X_test, y_true, baseline_acc, noise_levels)

    # Test 2: Feature dropout
    dropout_tests = test_feature_dropout(predictor, X_test, y_true, baseline_acc)

    # Test 3: Consistance des prédictions
    consistency_test = test_prediction_consistency(
        predictor, X_test, y_pred_baseline, n_consistency_samples
    )

    # Test 4: Monotonicité (features ELO)
    monotonicity_tests = test_monotonicity(predictor, X_test, y_true)

    # Métriques globales
    overall_noise = _compute_overall_noise_tolerance(noise_tests)
    stability_score = _compute_stability_score(dropout_tests, consistency_test)
    critical_features = [t.feature_name for t in dropout_tests if t.is_critical]

    # Vérification formelle (bounds)
    formal_verification = _formal_verification_bounds(noise_tests, monotonicity_tests)

    # Statut final
    status, compliant = _determine_status(overall_noise, stability_score, consistency_test)

    return ISO24029EnhancedReport(
        model_id=str(predictor.path),
        noise_tests=noise_tests,
        feature_dropout_tests=dropout_tests,
        consistency_test=consistency_test,
        monotonicity_tests=monotonicity_tests,
        overall_noise_tolerance=overall_noise,
        overall_stability_score=stability_score,
        critical_features=critical_features,
        status=status,
        compliant=compliant,
        formal_verification=formal_verification,
    )


def _compute_overall_noise_tolerance(noise_tests: list[NoiseRobustnessTest]) -> float:
    """Calcule la tolérance au bruit globale (moyenne pondérée)."""
    if not noise_tests:
        return 1.0

    # Pondération: plus de poids aux niveaux de bruit élevés
    weights = [t.noise_level for t in noise_tests]
    tolerances = [t.tolerance for t in noise_tests]

    return float(np.average(tolerances, weights=weights))


def _compute_stability_score(
    dropout_tests: list[FeatureDropoutTest],
    consistency_test: PredictionConsistencyTest,
) -> float:
    """Calcule le score de stabilité global."""
    # Score dropout: 1 - (impact moyen normalisé)
    if dropout_tests:
        avg_impact = np.mean([t.impact for t in dropout_tests])
        dropout_score = max(0, 1 - avg_impact * 5)  # Normaliser
    else:
        dropout_score = 1.0

    # Combiner avec consistance
    return float((dropout_score + consistency_test.consistency_rate) / 2)


def _formal_verification_bounds(
    noise_tests: list[NoiseRobustnessTest],
    monotonicity_tests: list[MonotonicityTest],
) -> dict[str, Any]:
    """Génère les bornes de vérification formelle ISO 24029-2."""
    return {
        "noise_tolerance_bounds": {
            "lower_bound": min(t.tolerance for t in noise_tests) if noise_tests else 1.0,
            "upper_bound": max(t.tolerance for t in noise_tests) if noise_tests else 1.0,
            "certified_noise_level": max(
                (t.noise_level for t in noise_tests if t.tolerance >= 0.95), default=0.0
            ),
        },
        "monotonicity_verified": all(t.is_monotonic for t in monotonicity_tests),
        "monotonicity_violations_total": sum(t.violations_count for t in monotonicity_tests),
        "formal_method": "statistical_verification",
        "confidence_level": 0.95,
    }


def _determine_status(
    noise_tolerance: float,
    stability_score: float,
    consistency: PredictionConsistencyTest,
) -> tuple[str, bool]:
    """Détermine le statut de conformité."""
    if noise_tolerance >= 0.95 and stability_score >= 0.90 and consistency.consistency_rate >= 0.99:
        return "ROBUST", True
    if noise_tolerance >= 0.85 and stability_score >= 0.80:
        return "ACCEPTABLE", True
    if noise_tolerance >= 0.75:
        return "SENSITIVE", False
    return "FRAGILE", False
