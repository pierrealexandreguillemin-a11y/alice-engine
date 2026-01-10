"""Module: scripts/robustness/adversarial_tests.py - Tests de robustesse ML.

Document ID: ALICE-MOD-ROBUST-001
Version: 1.0.0
Lines: ~520 (ISO 5055 compliant)

Ce module implémente les tests de robustesse pour évaluer
la stabilité des modèles ALICE face aux perturbations.

Tests implémentés (ISO 24029):
1. Bruit gaussien (run_noise_test):
   - Ajoute bruit N(0, sigma) aux features
   - Mesure dégradation accuracy et stabilité prédictions
   - Niveaux: sigma = 0.01, 0.05, 0.10

2. Perturbation features (run_feature_perturbation_test):
   - Modifie aléatoirement un subset de features
   - Mesure sensibilité aux variations
   - Niveaux: 5%, 10%, 20% de perturbation

3. Out-of-Distribution (run_out_of_distribution_test):
   - Génère échantillons hors distribution normale
   - Vérifie comportement sur données OOD
   - Multiplicateur: 3x écart-type

4. Valeurs extrêmes (run_extreme_values_test):
   - Remplace valeurs par percentiles extrêmes
   - Stress test du modèle
   - Percentile: 99ème

Classes:
- RobustnessLevel: Enum (ROBUST, ACCEPTABLE, WARNING, FRAGILE)
- RobustnessThresholds: Dataclass seuils configurables
- RobustnessMetrics: Dataclass métriques par test
- RobustnessReport: Dataclass rapport complet

Functions:
- compute_robustness_metrics(): Suite complète 8 tests
- run_noise_test(): Test bruit gaussien
- run_feature_perturbation_test(): Test perturbation features
- run_out_of_distribution_test(): Test OOD
- run_extreme_values_test(): Test valeurs extrêmes
- generate_robustness_report(): Génération rapport ISO

Seuils par défaut (ISO 24029-2 + littérature académique):
- Dégradation acceptable: < 3%
- Warning: 3-5%
- Critique: >= 10%
- Stabilité minimum: 95%

Sources académiques:
- ISO/IEC 24029-1:2021 - Assessment of robustness of neural networks
- ISO/IEC 24029-2:2023 - Methodology for robustness testing
- Goodfellow et al. "Explaining and Harnessing Adversarial Examples" (ICLR 2015)
- Hendrycks & Dietterich "Benchmarking Neural Network Robustness" (ICLR 2019)
- Szegedy et al. "Intriguing properties of neural networks" (ICLR 2014)

ISO Compliance:
- ISO/IEC 24029-1:2021 - Neural Network Robustness Assessment (primary)
- ISO/IEC 24029-2:2023 - Robustness Testing Methodology (primary)
- ISO/IEC 42001:2023 - AI Management System (traçabilité)
- ISO/IEC 25059:2023 - AI Quality Model (métriques)
- ISO/IEC 5055:2021 - Code Quality (module <600 lignes)

See Also
--------
- scripts/robustness/__init__.py - Package exports
- tests/test_robustness_adversarial.py - 29 tests unitaires
- docs/iso/AI_RISK_ASSESSMENT.md - Section R2: Model Performance Risks
- docs/iso/STATEMENT_OF_APPLICABILITY.md - Control B.4.5

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd  # noqa: TCH002 - Used at runtime for type coercion

logger = logging.getLogger(__name__)


class RobustnessLevel(Enum):
    """Niveau de robustesse selon ISO 24029."""

    ROBUST = "robust"
    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    FRAGILE = "fragile"


@dataclass
class RobustnessThresholds:
    """Seuils de robustesse ISO 24029.

    Sources:
    - ISO 24029-2 recommandations
    - Benchmarks industrie (5% dégradation acceptable)
    - Publications académiques sur adversarial robustness
    """

    # Dégradation de performance acceptable
    degradation_acceptable: float = 0.03  # 3%
    degradation_warning: float = 0.05  # 5%
    degradation_critical: float = 0.10  # 10%

    # Stabilité des prédictions
    stability_threshold: float = 0.95  # 95% de prédictions stables

    # Niveaux de bruit pour tests
    noise_levels: tuple[float, ...] = (0.01, 0.05, 0.10)

    # Perturbation features
    perturbation_levels: tuple[float, ...] = (0.05, 0.10, 0.20)


# Seuils par défaut (ISO 24029)
DEFAULT_THRESHOLDS = RobustnessThresholds()


@dataclass
class RobustnessMetrics:
    """Métriques de robustesse pour un test.

    Attributes
    ----------
        test_name: Nom du test
        original_score: Score sur données originales
        perturbed_score: Score après perturbation
        degradation: Dégradation relative (%)
        stability_ratio: % de prédictions inchangées
        level: Niveau de robustesse
        details: Détails additionnels
    """

    test_name: str
    original_score: float
    perturbed_score: float
    degradation: float
    stability_ratio: float
    level: RobustnessLevel = RobustnessLevel.ACCEPTABLE
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobustnessReport:
    """Rapport complet de robustesse ISO 24029.

    Attributes
    ----------
        timestamp: Date/heure du rapport
        model_name: Nom du modèle testé
        total_tests: Nombre de tests effectués
        metrics: Liste des métriques par test
        overall_level: Niveau de robustesse global
        recommendations: Recommandations d'action
    """

    timestamp: str
    model_name: str
    total_tests: int
    original_accuracy: float
    metrics: list[RobustnessMetrics] = field(default_factory=list)
    overall_level: RobustnessLevel = RobustnessLevel.ACCEPTABLE
    recommendations: list[str] = field(default_factory=list)
    iso_compliance: dict[str, Any] = field(default_factory=dict)


def compute_robustness_metrics(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    thresholds: RobustnessThresholds | None = None,
) -> list[RobustnessMetrics]:
    """Exécute la suite complète de tests de robustesse.

    Args:
    ----
        model_predict: Fonction de prédiction du modèle
        X: Features d'entrée
        y_true: Labels réels
        thresholds: Seuils de robustesse

    Returns:
    -------
        Liste de RobustnessMetrics pour chaque test

    ISO 24029: Suite complète de tests de robustesse.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    X = np.asarray(X)
    y_true = np.asarray(y_true)

    logger.info("Exécution tests de robustesse ISO 24029...")

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

    # Test 4: Valeurs extrêmes
    result = run_extreme_values_test(model_predict, X, y_true, thresholds=thresholds)
    metrics.append(result)

    logger.info(f"  {len(metrics)} tests exécutés")
    return metrics


def run_noise_test(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y_true: np.ndarray,
    noise_std: float = 0.05,
    thresholds: RobustnessThresholds | None = None,
) -> RobustnessMetrics:
    """Test de robustesse au bruit gaussien.

    Args:
    ----
        model_predict: Fonction de prédiction
        X: Features originales
        y_true: Labels réels
        noise_std: Écart-type du bruit gaussien
        thresholds: Seuils

    Returns:
    -------
        RobustnessMetrics pour ce test

    ISO 24029: Évaluation de la robustesse au bruit aléatoire.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    X = np.asarray(X)
    y_true = np.asarray(y_true)

    # Prédictions originales
    y_pred_original = model_predict(X)
    original_acc = _compute_accuracy(y_true, y_pred_original)

    # Ajouter bruit gaussien
    noise = np.random.normal(0, noise_std, X.shape)
    X_noisy = X + noise

    # Prédictions perturbées
    y_pred_noisy = model_predict(X_noisy)
    noisy_acc = _compute_accuracy(y_true, y_pred_noisy)

    # Calcul métriques
    degradation = (original_acc - noisy_acc) / original_acc if original_acc > 0 else 0
    stability = np.mean(y_pred_original == y_pred_noisy)

    # Déterminer niveau
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

    Args:
    ----
        model_predict: Fonction de prédiction
        X: Features originales
        y_true: Labels réels
        perturbation: Amplitude de la perturbation (%)
        n_features_to_perturb: Nombre de features à perturber (défaut: toutes)
        thresholds: Seuils

    Returns:
    -------
        RobustnessMetrics pour ce test

    ISO 24029: Analyse de sensibilité aux features.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    X = np.asarray(X, dtype=np.float64)
    y_true = np.asarray(y_true)

    # Prédictions originales
    y_pred_original = model_predict(X)
    original_acc = _compute_accuracy(y_true, y_pred_original)

    # Perturbation proportionnelle à chaque feature
    X_perturbed = X.copy()
    n_features = X.shape[1] if len(X.shape) > 1 else 1

    if n_features_to_perturb is None:
        n_features_to_perturb = n_features

    # Sélectionner features à perturber aléatoirement
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

    # Prédictions perturbées
    y_pred_perturbed = model_predict(X_perturbed)
    perturbed_acc = _compute_accuracy(y_true, y_pred_perturbed)

    # Calcul métriques
    degradation = (original_acc - perturbed_acc) / original_acc if original_acc > 0 else 0
    stability = np.mean(y_pred_original == y_pred_perturbed)

    # Déterminer niveau
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

    Crée des échantillons OOD en déplaçant les features
    au-delà de leur plage normale (x * ood_multiplier * std).

    Args:
    ----
        model_predict: Fonction de prédiction
        X: Features originales
        y_true: Labels réels
        ood_multiplier: Multiplicateur pour créer OOD
        thresholds: Seuils

    Returns:
    -------
        RobustnessMetrics pour ce test

    ISO 24029: Détection et gestion des entrées OOD.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    X = np.asarray(X, dtype=np.float64)
    y_true = np.asarray(y_true)

    # Prédictions originales
    y_pred_original = model_predict(X)
    original_acc = _compute_accuracy(y_true, y_pred_original)

    # Créer données OOD
    X_ood = X.copy()
    n_samples = len(X)
    n_ood = max(1, n_samples // 10)  # 10% d'échantillons OOD

    # Sélectionner échantillons à rendre OOD
    ood_indices = np.random.choice(n_samples, size=n_ood, replace=False)

    for idx in ood_indices:
        if len(X.shape) > 1:
            for col in range(X.shape[1]):
                col_std = np.std(X[:, col])
                col_mean = np.mean(X[:, col])
                # Valeur OOD: au-delà de la distribution normale
                direction = np.random.choice([-1, 1])
                X_ood[idx, col] = col_mean + direction * ood_multiplier * col_std
        else:
            x_std = np.std(X)
            x_mean = np.mean(X)
            direction = np.random.choice([-1, 1])
            X_ood[idx] = x_mean + direction * ood_multiplier * x_std

    # Prédictions sur données OOD
    y_pred_ood = model_predict(X_ood)
    ood_acc = _compute_accuracy(y_true, y_pred_ood)

    # Stabilité: combien de prédictions non-OOD sont inchangées
    non_ood_mask = np.ones(n_samples, dtype=bool)
    non_ood_mask[ood_indices] = False
    stability = np.mean(y_pred_original[non_ood_mask] == y_pred_ood[non_ood_mask])

    # Dégradation
    degradation = (original_acc - ood_acc) / original_acc if original_acc > 0 else 0

    # Pour OOD, on attend une certaine dégradation
    # Le niveau est "acceptable" si le modèle dégrade proprement sur OOD
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
            "ood_indices": ood_indices.tolist()[:10],  # Limiter pour lisibilité
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

    Remplace une partie des valeurs par des valeurs au percentile extrême.

    Args:
    ----
        model_predict: Fonction de prédiction
        X: Features originales
        y_true: Labels réels
        percentile: Percentile pour valeurs extrêmes
        thresholds: Seuils

    Returns:
    -------
        RobustnessMetrics pour ce test

    ISO 24029: Stress testing avec valeurs extrêmes.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    X = np.asarray(X, dtype=np.float64)
    y_true = np.asarray(y_true)

    # Prédictions originales
    y_pred_original = model_predict(X)
    original_acc = _compute_accuracy(y_true, y_pred_original)

    # Créer version avec valeurs extrêmes
    X_extreme = X.copy()
    n_samples = len(X)
    n_extreme = max(1, n_samples // 20)  # 5% d'échantillons extrêmes

    extreme_indices = np.random.choice(n_samples, size=n_extreme, replace=False)

    for idx in extreme_indices:
        if len(X.shape) > 1:
            for col in range(X.shape[1]):
                # Remplacer par valeur au percentile extrême
                extreme_val = np.percentile(X[:, col], percentile)
                X_extreme[idx, col] = extreme_val
        else:
            extreme_val = np.percentile(X, percentile)
            X_extreme[idx] = extreme_val

    # Prédictions sur données extrêmes
    y_pred_extreme = model_predict(X_extreme)
    extreme_acc = _compute_accuracy(y_true, y_pred_extreme)

    # Stabilité sur échantillons non-extrêmes
    non_extreme_mask = np.ones(n_samples, dtype=bool)
    non_extreme_mask[extreme_indices] = False
    stability = np.mean(y_pred_original[non_extreme_mask] == y_pred_extreme[non_extreme_mask])

    # Dégradation
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


def generate_robustness_report(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    model_name: str = "Unknown",
    thresholds: RobustnessThresholds | None = None,
) -> RobustnessReport:
    """Génère un rapport de robustesse complet ISO 24029.

    Args:
    ----
        model_predict: Fonction de prédiction
        X: Features
        y_true: Labels réels
        model_name: Nom du modèle
        thresholds: Seuils

    Returns:
    -------
        RobustnessReport complet avec métriques et recommandations

    ISO 24029: Rapport de robustesse traçable.
    ISO 42001: Documentation des décisions AI.
    """
    logger.info(f"Génération rapport robustesse: {model_name}")

    X = np.asarray(X)
    y_true = np.asarray(y_true)

    # Calcul accuracy originale
    y_pred = model_predict(X)
    original_accuracy = _compute_accuracy(y_true, y_pred)

    # Exécuter tous les tests
    metrics = compute_robustness_metrics(model_predict, X, y_true, thresholds)

    # Déterminer niveau global
    overall_level = _determine_overall_level(metrics)

    # Générer recommandations
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
            "iso_24029_1": True,  # Robustness assessment
            "iso_24029_2": True,  # Testing methodology
            "iso_42001": True,  # AI management
            "all_tests_passed": all(
                m.level in (RobustnessLevel.ROBUST, RobustnessLevel.ACCEPTABLE) for m in metrics
            ),
            "fragile_tests": [m.test_name for m in metrics if m.level == RobustnessLevel.FRAGILE],
            "warning_tests": [m.test_name for m in metrics if m.level == RobustnessLevel.WARNING],
        },
    )

    logger.info(f"  Niveau global: {overall_level.value}")
    logger.info(
        f"  {len([m for m in metrics if m.level == RobustnessLevel.FRAGILE])} tests fragiles"
    )

    return report


def _compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'accuracy."""
    return float(np.mean(y_true == y_pred))


def _determine_robustness_level(
    degradation: float,
    stability: float,
    thresholds: RobustnessThresholds,
) -> RobustnessLevel:
    """Détermine le niveau de robustesse."""
    # Fragile si dégradation critique ou stabilité très basse
    if degradation >= thresholds.degradation_critical or stability < 0.7:
        return RobustnessLevel.FRAGILE

    # Warning si dégradation warning ou stabilité moyenne
    if degradation >= thresholds.degradation_warning or stability < 0.85:
        return RobustnessLevel.WARNING

    # Acceptable si dégradation acceptable
    if (
        degradation >= thresholds.degradation_acceptable
        or stability < thresholds.stability_threshold
    ):
        return RobustnessLevel.ACCEPTABLE

    # Robust sinon
    return RobustnessLevel.ROBUST


def _determine_overall_level(metrics: list[RobustnessMetrics]) -> RobustnessLevel:
    """Détermine le niveau global à partir de tous les tests."""
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
    """Génère des recommandations basées sur les métriques."""
    recommendations = []

    if overall_level == RobustnessLevel.FRAGILE:
        recommendations.append(
            "CRITIQUE: Modèle fragile détecté. Ne pas déployer en production "
            "sans renforcement de la robustesse."
        )

    # Analyser les tests problématiques
    fragile_tests = [m for m in metrics if m.level == RobustnessLevel.FRAGILE]
    warning_tests = [m for m in metrics if m.level == RobustnessLevel.WARNING]

    if fragile_tests:
        test_names = ", ".join(m.test_name for m in fragile_tests)
        recommendations.append(
            f"Tests fragiles: {test_names}. Implémenter data augmentation "
            "et/ou régularisation renforcée."
        )

    if warning_tests:
        test_names = ", ".join(m.test_name for m in warning_tests)
        recommendations.append(f"Tests warning: {test_names}. Monitoring renforcé recommandé.")

    # Recommandations spécifiques
    for m in metrics:
        if m.level == RobustnessLevel.FRAGILE:
            if "noise" in m.test_name:
                recommendations.append(
                    f"Bruit sensible ({m.test_name}): Considérer ajout de "
                    "bruit gaussien durant l'entraînement (noise injection)."
                )
            elif "perturbation" in m.test_name:
                recommendations.append(
                    f"Perturbation sensible ({m.test_name}): Considérer "
                    "feature selection ou dropout renforcé."
                )
            elif "distribution" in m.test_name:
                recommendations.append(
                    f"OOD sensible ({m.test_name}): Implémenter détection OOD "
                    "et mécanisme de rejet."
                )
            elif "extreme" in m.test_name:
                recommendations.append(
                    f"Valeurs extrêmes ({m.test_name}): Considérer clipping "
                    "des features ou normalisation robuste."
                )

    if overall_level == RobustnessLevel.ROBUST:
        recommendations.append(
            "Modèle robuste. Maintenir monitoring régulier et réévaluer "
            "après chaque réentraînement."
        )

    return recommendations
