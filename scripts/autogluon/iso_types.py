"""Types ISO pour AutoGluon - ISO 42001/24029/24027.

Ce module contient les dataclasses pour les rapports ISO (base + enhanced).

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Model Card)
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC TR 24027:2021 - Bias Detection
- ISO/IEC 5055:2021 - Code Quality (<300 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ISO42001ModelCard:
    """Model Card conforme ISO 42001.

    Documentation complete du modele pour tracabilite et gouvernance AI.
    """

    model_id: str
    model_name: str
    version: str
    created_at: str
    training_data_hash: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    intended_use: str = "Prediction de resultats d'echecs (classification binaire)"
    limitations: str = "Donnees FFE France uniquement, competitions officielles"
    ethical_considerations: str = "Pas de donnees personnelles sensibles"
    best_model: str = ""
    num_models_trained: int = 0
    feature_importance: dict[str, float] = field(default_factory=dict)


@dataclass
class ISO24029RobustnessReport:
    """Rapport de robustesse conforme ISO 24029.

    Analyse de la robustesse du modele face aux perturbations.
    """

    model_id: str
    noise_tolerance: float
    adversarial_robustness: float
    distribution_shift_score: float
    confidence_calibration: float
    status: str = "NOT_EVALUATED"  # ROBUST, SENSITIVE, FRAGILE, NOT_EVALUATED


@dataclass
class ISO24027BiasReport:
    """Rapport de biais conforme ISO 24027.

    Analyse de l'equite du modele.
    """

    model_id: str
    demographic_parity: float
    equalized_odds: float
    calibration_by_group: dict[str, float] = field(default_factory=dict)
    status: str = "NOT_EVALUATED"  # FAIR, CAUTION, CRITICAL, NOT_EVALUATED


# --- Enhanced Fairness types (ISO 24027) ---


@dataclass
class FairnessMetrics:
    """Métriques de fairness ISO 24027."""

    demographic_parity_ratio: float
    equalized_odds_ratio: float
    predictive_parity_ratio: float
    calibration_score: float


@dataclass
class GroupAnalysis:
    """Analyse par groupe pour root cause."""

    group_name: str
    sample_count: int
    positive_rate: float
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    is_disadvantaged: bool
    deviation_from_mean: float


@dataclass
class ISO24027EnhancedReport:
    """Rapport de fairness amélioré ISO 24027."""

    model_id: str
    protected_attribute: str
    metrics: FairnessMetrics
    group_analyses: list[GroupAnalysis]
    root_cause: str
    mitigations: list[str]
    status: str
    compliant: bool
    data_quality_warnings: list[str] = field(default_factory=list)


# --- Enhanced Robustness types (ISO 24029) ---


@dataclass
class NoiseRobustnessTest:
    """Résultat test de robustesse au bruit."""

    noise_level: float
    baseline_accuracy: float
    noisy_accuracy: float
    tolerance: float
    status: str


@dataclass
class FeatureDropoutTest:
    """Résultat test de dropout de features."""

    feature_name: str
    baseline_accuracy: float
    dropout_accuracy: float
    impact: float
    is_critical: bool


@dataclass
class PredictionConsistencyTest:
    """Résultat test de consistance des prédictions."""

    n_perturbations: int
    consistency_rate: float
    flip_rate: float
    status: str


@dataclass
class MonotonicityTest:
    """Résultat test de monotonicité (corrélation ELO attendue)."""

    feature_name: str
    expected_direction: str
    actual_correlation: float
    is_monotonic: bool
    violations_count: int


@dataclass
class ISO24029EnhancedReport:
    """Rapport de robustesse amélioré ISO 24029."""

    model_id: str
    noise_tests: list[NoiseRobustnessTest]
    feature_dropout_tests: list[FeatureDropoutTest]
    consistency_test: PredictionConsistencyTest
    monotonicity_tests: list[MonotonicityTest]
    overall_noise_tolerance: float
    overall_stability_score: float
    critical_features: list[str]
    status: str
    compliant: bool
    formal_verification: dict[str, Any] = field(default_factory=dict)
