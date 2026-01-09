"""Module: services/__init__.py - Services Package ALICE.

Services Layer - Logique metier pure (SRP).
Les services sont testables en isolation.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management (inference, optimization)
- ISO/IEC 42010 - Architecture (Service layer, SRP)
- ISO/IEC 25010 - System Quality (maintenabilite)

Services:
- InferenceService: ALI (Adversarial Lineup Inference)
- ComposerService: CE (Composition Engine)
- DataLoader: Repository pattern

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from services.composer import ComposerService
from services.data_loader import DataLoader
from services.inference import InferenceService

__all__ = ["InferenceService", "ComposerService", "DataLoader"]
