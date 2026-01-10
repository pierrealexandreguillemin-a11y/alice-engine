"""Module: scripts/autogluon/__init__.py - AutoGluon Package.

Document ID: ALICE-MOD-AUTOGLUON-PKG-001
Version: 1.0.0

Package AutoGluon pour ALICE Engine - AutoML avec TabPFN-2.5.

Modules:
- config: Configuration AutoGluon (ISO 42001)
- trainer: Pipeline d'entrainement AutoGluon
- predictor_wrapper: Wrapper sklearn-compatible
- iso_compliance: Verification conformite ISO 42001/24029/24027

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Tracabilite)
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC TR 24027:2021 - Bias Detection
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from scripts.autogluon.config import (
    AutoGluonConfig,
    load_autogluon_config,
)
from scripts.autogluon.iso_compliance import (
    ISO24027BiasReport,
    ISO24029RobustnessReport,
    ISO42001ModelCard,
    generate_model_card,
    validate_iso_compliance,
)
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
    # ISO Compliance
    "ISO42001ModelCard",
    "ISO24029RobustnessReport",
    "ISO24027BiasReport",
    "generate_model_card",
    "validate_iso_compliance",
]
