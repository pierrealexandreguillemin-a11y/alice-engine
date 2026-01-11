"""Tests de perturbation - ISO 24029.

Ce module implémente les tests de robustesse aux perturbations:
- run_noise_test: Test bruit gaussien
- run_feature_perturbation_test: Test perturbation features
- run_out_of_distribution_test: Test OOD
- run_extreme_values_test: Test valeurs extrêmes

ISO Compliance:
- ISO/IEC 24029-1:2021 - Neural Network Robustness Assessment
- ISO/IEC 24029-2:2023 - Robustness Testing Methodology
- ISO/IEC 5055:2021 - Code Quality (<300 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from scripts.robustness.thresholds import DEFAULT_THRESHOLDS, RobustnessThresholds
from scripts.robustness.types import RobustnessLevel, RobustnessMetrics


def _compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'accuracy."""
    return float(np.mean(y_true == y_pred))


def _determine_robustness_level(
    degradation: float,
    stability: float,
    thresholds: RobustnessThresholds,
) -> RobustnessLevel:
    """Détermine le niveau de robustesse."""
    if degradation >= thresholds.degradation_critical or stability < 0.7:
        return RobustnessLevel.FRAGILE
    if degradation >= thresholds.degradation_warning or stability < 0.85:
        return RobustnessLevel.WARNING
    if (
        degradation >= thresholds.degradation_acceptable
        or stability < thresholds.stability_threshold
    ):
        return RobustnessLevel.ACCEPTABLE
    return RobustnessLevel.ROBUST


def run_noise_test(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y_true: np.ndarray,
    noise_std: float = 0.05,
    thresholds: RobustnessThresholds | None = None,
) -> RobustnessMetrics:
    """Test de robustesse au bruit gaussien.

    ISO 24029: Évaluation de la robustesse au bruit aléatoire.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    X = np.asarray(X)
    y_true = np.asarray(y_true)

    y_pred_original = model_predict(X)
    original_acc = _compute_accuracy(y_true, y_pred_original)

    noise = np.random.normal(0, noise_std, X.shape)
    X_noisy = X + noise

    y_pred_noisy = model_predict(X_noisy)
    noisy_acc = _compute_accuracy(y_true, y_pred_noisy)

    degradation = (original_acc - noisy_acc) / original_acc if original_acc > 0 else 0
    stability = np.mean(y_pred_original == y_pred_noisy)
    level = _determine_robustness_level(degradation, stability, thresholds)

    return RobustnessMetrics(
        test_name=f"gaussian_noise_{noise_std}",
        original_score=round(original_acc, 4),
        perturbed_score=round(noisy_acc, 4),
        degradation=round(degradation, 4),
        stability_ratio=round(stability, 4),
        level=level,
        details={"noise_std": noise_std, "n_samples": len(X)},
    )


def run_feature_perturbation_test(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y_true: np.ndarray,
    perturbation: float = 0.10,
    n_features_to_perturb: int | None = None,
    thresholds: RobustnessThresholds | None = None,
) -> RobustnessMetrics:
    """Test de sensibilité aux perturbations de features.

    ISO 24029: Analyse de sensibilité aux features.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    X = np.asarray(X, dtype=np.float64)
    y_true = np.asarray(y_true)

    y_pred_original = model_predict(X)
    original_acc = _compute_accuracy(y_true, y_pred_original)

    X_perturbed = X.copy()
    n_features = X.shape[1] if len(X.shape) > 1 else 1

    if n_features_to_perturb is None:
        n_features_to_perturb = n_features

    feature_indices = np.random.choice(
        n_features, size=min(n_features_to_perturb, n_features), replace=False
    )

    for idx in feature_indices:
        if len(X.shape) > 1:
            feature_std = np.std(X[:, idx])
            perturbation_noise = np.random.uniform(-perturbation, perturbation, X.shape[0])
            X_perturbed[:, idx] += feature_std * perturbation_noise
        else:
            feature_std = np.std(X)
            perturbation_noise = np.random.uniform(-perturbation, perturbation, X.shape[0])
            X_perturbed += feature_std * perturbation_noise

    y_pred_perturbed = model_predict(X_perturbed)
    perturbed_acc = _compute_accuracy(y_true, y_pred_perturbed)

    degradation = (original_acc - perturbed_acc) / original_acc if original_acc > 0 else 0
    stability = np.mean(y_pred_original == y_pred_perturbed)
    level = _determine_robustness_level(degradation, stability, thresholds)

    return RobustnessMetrics(
        test_name=f"feature_perturbation_{perturbation}",
        original_score=round(original_acc, 4),
        perturbed_score=round(perturbed_acc, 4),
        degradation=round(degradation, 4),
        stability_ratio=round(stability, 4),
        level=level,
        details={
            "perturbation_amplitude": perturbation,
            "n_features_perturbed": len(feature_indices),
            "features_perturbed": feature_indices.tolist(),
        },
    )


def run_out_of_distribution_test(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y_true: np.ndarray,
    ood_multiplier: float = 3.0,
    thresholds: RobustnessThresholds | None = None,
) -> RobustnessMetrics:
    """Test de comportement sur données out-of-distribution.

    ISO 24029: Détection et gestion des entrées OOD.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    X = np.asarray(X, dtype=np.float64)
    y_true = np.asarray(y_true)

    y_pred_original = model_predict(X)
    original_acc = _compute_accuracy(y_true, y_pred_original)

    X_ood = X.copy()
    n_samples = len(X)
    n_ood = max(1, n_samples // 10)

    ood_indices = np.random.choice(n_samples, size=n_ood, replace=False)

    for idx in ood_indices:
        if len(X.shape) > 1:
            for col in range(X.shape[1]):
                col_std = np.std(X[:, col])
                col_mean = np.mean(X[:, col])
                direction = np.random.choice([-1, 1])
                X_ood[idx, col] = col_mean + direction * ood_multiplier * col_std
        else:
            x_std = np.std(X)
            x_mean = np.mean(X)
            direction = np.random.choice([-1, 1])
            X_ood[idx] = x_mean + direction * ood_multiplier * x_std

    y_pred_ood = model_predict(X_ood)
    ood_acc = _compute_accuracy(y_true, y_pred_ood)

    non_ood_mask = np.ones(n_samples, dtype=bool)
    non_ood_mask[ood_indices] = False
    stability = np.mean(y_pred_original[non_ood_mask] == y_pred_ood[non_ood_mask])

    degradation = (original_acc - ood_acc) / original_acc if original_acc > 0 else 0
    level = _determine_robustness_level(max(0, degradation - 0.1), stability, thresholds)

    return RobustnessMetrics(
        test_name="out_of_distribution",
        original_score=round(original_acc, 4),
        perturbed_score=round(ood_acc, 4),
        degradation=round(degradation, 4),
        stability_ratio=round(stability, 4),
        level=level,
        details={
            "ood_multiplier": ood_multiplier,
            "n_ood_samples": n_ood,
            "ood_indices": ood_indices.tolist()[:10],
        },
    )


def run_extreme_values_test(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y_true: np.ndarray,
    percentile: float = 99.0,
    thresholds: RobustnessThresholds | None = None,
) -> RobustnessMetrics:
    """Test de comportement sur valeurs extrêmes.

    ISO 24029: Stress testing avec valeurs extrêmes.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    X = np.asarray(X, dtype=np.float64)
    y_true = np.asarray(y_true)

    y_pred_original = model_predict(X)
    original_acc = _compute_accuracy(y_true, y_pred_original)

    X_extreme = X.copy()
    n_samples = len(X)
    n_extreme = max(1, n_samples // 20)

    extreme_indices = np.random.choice(n_samples, size=n_extreme, replace=False)

    for idx in extreme_indices:
        if len(X.shape) > 1:
            for col in range(X.shape[1]):
                extreme_val = np.percentile(X[:, col], percentile)
                X_extreme[idx, col] = extreme_val
        else:
            extreme_val = np.percentile(X, percentile)
            X_extreme[idx] = extreme_val

    y_pred_extreme = model_predict(X_extreme)
    extreme_acc = _compute_accuracy(y_true, y_pred_extreme)

    non_extreme_mask = np.ones(n_samples, dtype=bool)
    non_extreme_mask[extreme_indices] = False
    stability = np.mean(y_pred_original[non_extreme_mask] == y_pred_extreme[non_extreme_mask])

    degradation = (original_acc - extreme_acc) / original_acc if original_acc > 0 else 0
    level = _determine_robustness_level(degradation, stability, thresholds)

    return RobustnessMetrics(
        test_name=f"extreme_values_p{int(percentile)}",
        original_score=round(original_acc, 4),
        perturbed_score=round(extreme_acc, 4),
        degradation=round(degradation, 4),
        stability_ratio=round(stability, 4),
        level=level,
        details={
            "percentile": percentile,
            "n_extreme_samples": n_extreme,
        },
    )
