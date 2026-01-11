"""Module: scripts/autogluon/iso_compliance.py - ISO Compliance (Re-export).

Document ID: ALICE-MOD-AUTOGLUON-ISO-001
Version: 2.0.0

Re-export pour compatibilite. Voir modules SRP:
- iso_types.py: Dataclasses ISO
- iso_model_card.py: Generation Model Card
- iso_robustness.py: Validation robustesse
- iso_fairness.py: Validation equite
- iso_validator.py: Validation complete

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (<80 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from scripts.autogluon.iso_fairness import validate_fairness
from scripts.autogluon.iso_model_card import generate_model_card
from scripts.autogluon.iso_robustness import validate_robustness
from scripts.autogluon.iso_types import (
    ISO24027BiasReport,
    ISO24029RobustnessReport,
    ISO42001ModelCard,
)
from scripts.autogluon.iso_validator import validate_iso_compliance

__all__ = [
    "ISO42001ModelCard",
    "ISO24029RobustnessReport",
    "ISO24027BiasReport",
    "generate_model_card",
    "validate_robustness",
    "validate_fairness",
    "validate_iso_compliance",
]
