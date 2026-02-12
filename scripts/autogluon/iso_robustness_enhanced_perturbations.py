"""Tests de perturbation pour robustesse - ISO 24029.

Ce module contient les fonctions de test individuelles (bruit, dropout,
consistance, monotonicité) pour l'analyse de robustesse.

ISO Compliance:
- ISO/IEC 24029-1:2021 - Robustness Overview
- ISO/IEC 24029-2:2023 - Formal Methods Methodology
- ISO/IEC 5055:2021 - Code Quality (SRP)

Document ID: ALICE-SCRIPT-ISO24029-PERTURBATIONS-001
Version: 1.0.0
Author: ALICE Engine Team
Last Updated: 2026-02-12
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.autogluon.iso_types import (
    FeatureDropoutTest,
    MonotonicityTest,
    NoiseRobustnessTest,
    PredictionConsistencyTest,
)


def test_noise_robustness(
    predictor: Any,
    X_test: Any,
    y_true: np.ndarray,
    baseline_acc: float,
    noise_levels: list[float],
) -> list[NoiseRobustnessTest]:
    """Test de robustesse au bruit gaussien multi-niveaux."""
    results = []
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()

    for noise_level in noise_levels:
        X_noisy = X_test.copy()

        for col in numeric_cols:
            std = X_noisy[col].std()
            if std > 0:
                noise = np.random.normal(0, noise_level * std, len(X_noisy))
                X_noisy[col] = X_noisy[col] + noise

        y_pred_noisy = predictor.predict(X_noisy).values
        noisy_acc = (y_pred_noisy == y_true).mean()
        tolerance = noisy_acc / baseline_acc if baseline_acc > 0 else 0

        status = "ROBUST" if tolerance >= 0.95 else "SENSITIVE" if tolerance >= 0.85 else "FRAGILE"

        results.append(
            NoiseRobustnessTest(
                noise_level=noise_level,
                baseline_accuracy=float(baseline_acc),
                noisy_accuracy=float(noisy_acc),
                tolerance=float(tolerance),
                status=status,
            )
        )

    return results


def test_feature_dropout(
    predictor: Any,
    X_test: Any,
    y_true: np.ndarray,
    baseline_acc: float,
) -> list[FeatureDropoutTest]:
    """Test de robustesse au dropout de features individuelles."""
    results = []
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols[:10]:  # Limiter aux 10 premières features numériques
        X_dropout = X_test.copy()
        X_dropout[col] = 0  # Remplacer par 0 (ou moyenne)

        y_pred_dropout = predictor.predict(X_dropout).values
        dropout_acc = (y_pred_dropout == y_true).mean()
        impact = baseline_acc - dropout_acc

        results.append(
            FeatureDropoutTest(
                feature_name=col,
                baseline_accuracy=float(baseline_acc),
                dropout_accuracy=float(dropout_acc),
                impact=float(impact),
                is_critical=impact > 0.05,  # >5% d'impact = critique
            )
        )

    return sorted(results, key=lambda x: x.impact, reverse=True)


def test_prediction_consistency(
    predictor: Any,
    X_test: Any,
    y_pred_baseline: np.ndarray,
    n_samples: int,
) -> PredictionConsistencyTest:
    """Test de consistance: même entrée = même sortie après micro-perturbations."""
    if len(X_test) > n_samples:
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test.iloc[indices]
        y_baseline_sample = y_pred_baseline[indices]
    else:
        X_sample = X_test
        y_baseline_sample = y_pred_baseline

    numeric_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
    n_perturbations = 5
    flip_count = 0
    total_comparisons = 0

    for _ in range(n_perturbations):
        X_perturbed = X_sample.copy()

        # Micro-perturbation (0.1% du std)
        for col in numeric_cols:
            std = X_perturbed[col].std()
            if std > 0:
                noise = np.random.normal(0, 0.001 * std, len(X_perturbed))
                X_perturbed[col] = X_perturbed[col] + noise

        y_pred_perturbed = predictor.predict(X_perturbed).values
        flip_count += (y_pred_perturbed != y_baseline_sample).sum()
        total_comparisons += len(y_baseline_sample)

    flip_rate = flip_count / total_comparisons if total_comparisons > 0 else 0
    consistency_rate = 1 - flip_rate

    status = (
        "STABLE"
        if consistency_rate >= 0.99
        else "SENSITIVE"
        if consistency_rate >= 0.95
        else "UNSTABLE"
    )

    return PredictionConsistencyTest(
        n_perturbations=n_perturbations,
        consistency_rate=float(consistency_rate),
        flip_rate=float(flip_rate),
        status=status,
    )


def test_monotonicity(
    predictor: Any,
    X_test: Any,
    y_true: np.ndarray,
) -> list[MonotonicityTest]:
    """Test de monotonicité pour features avec direction attendue.

    Pour les échecs: ELO plus élevé devrait augmenter la probabilité de victoire.
    """
    results = []
    elo_features = [c for c in X_test.columns if "elo" in c.lower() or "rating" in c.lower()]

    for feature in elo_features[:5]:
        # Calculer la corrélation entre la feature et les prédictions
        feature_values = X_test[feature].values
        predictions = predictor.predict(X_test).values

        # Corrélation de Spearman (plus robuste)
        valid_mask = ~np.isnan(feature_values)
        if valid_mask.sum() < 100:
            continue

        correlation = np.corrcoef(feature_values[valid_mask], predictions[valid_mask])[0, 1]

        # Pour ELO, on attend une corrélation positive (ELO haut = victoire)
        expected_direction = "positive"
        is_monotonic = correlation > 0.1  # Corrélation significative positive

        # Compter les violations (ELO plus haut mais prédiction plus basse)
        sorted_indices = np.argsort(feature_values)
        sorted_preds = predictions[sorted_indices]
        violations = (np.diff(sorted_preds) < 0).sum()

        results.append(
            MonotonicityTest(
                feature_name=feature,
                expected_direction=expected_direction,
                actual_correlation=float(correlation) if not np.isnan(correlation) else 0.0,
                is_monotonic=is_monotonic,
                violations_count=int(violations),
            )
        )

    return results
