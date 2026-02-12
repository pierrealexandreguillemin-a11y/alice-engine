"""Module: scripts/autogluon/__init__.py - AutoGluon Package.

Document ID: ALICE-MOD-AUTOGLUON-PKG-001
Version: 2.1.0

Package AutoGluon pour ALICE Engine - AutoML avec TabPFN-2.5.

Modules:
- config: Configuration AutoGluon (ISO 42001)
- trainer: Pipeline d'entrainement AutoGluon
- predictor_wrapper: Wrapper sklearn-compatible
- iso_types: Types ISO (dataclasses base + enhanced)
- iso_model_card: Generation Model Card ISO 42001
- iso_robustness: Validation robustesse ISO 24029
- iso_fairness: Validation equite ISO 24027
- iso_validator: Validation complete ISO
- iso_fairness_enhanced: Validation fairness amelioree ISO 24027
- iso_robustness_enhanced: Validation robustesse amelioree ISO 24029
- iso_impact_assessment_enhanced: Impact assessment ISO 42005

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Tracabilite)
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC TR 24027:2021 - Bias Detection
- ISO/IEC 5055:2021 - Code Quality (<100 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-12
"""

from scripts.autogluon.config import (
    AutoGluonConfig,
    load_autogluon_config,
)
from scripts.autogluon.iso_fairness import validate_fairness
from scripts.autogluon.iso_fairness_enhanced import validate_fairness_enhanced
from scripts.autogluon.iso_impact_assessment_enhanced import (
    assess_impact_enhanced,
    save_report,
)
from scripts.autogluon.iso_model_card import generate_model_card
from scripts.autogluon.iso_robustness import validate_robustness
from scripts.autogluon.iso_robustness_enhanced import validate_robustness_enhanced
from scripts.autogluon.iso_types import (
    ISO24027BiasReport,
    ISO24029RobustnessReport,
    ISO42001ModelCard,
)
from scripts.autogluon.iso_validator import validate_iso_compliance
from scripts.autogluon.predictor_wrapper import AutoGluonWrapper
from scripts.autogluon.trainer import (
    AutoGluonTrainer,
    AutoGluonTrainingResult,
    train_autogluon,
)

__all__ = [
    # Config
    "AutoGluonConfig",
    "load_autogluon_config",
    # Trainer
    "AutoGluonTrainer",
    "AutoGluonTrainingResult",
    "train_autogluon",
    # Wrapper
    "AutoGluonWrapper",
    # ISO Compliance (base)
    "ISO42001ModelCard",
    "ISO24029RobustnessReport",
    "ISO24027BiasReport",
    "generate_model_card",
    "validate_robustness",
    "validate_fairness",
    "validate_iso_compliance",
    # ISO Compliance (enhanced)
    "validate_fairness_enhanced",
    "validate_robustness_enhanced",
    "assess_impact_enhanced",
    "save_report",
]
