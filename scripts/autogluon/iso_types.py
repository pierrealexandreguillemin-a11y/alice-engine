"""Types ISO pour AutoGluon - ISO 42001/24029/24027.

Ce module contient les dataclasses pour les rapports ISO.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Model Card)
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC TR 24027:2021 - Bias Detection
- ISO/IEC 5055:2021 - Code Quality (<100 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
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
