"""Module: services/__init__.py - Services Package ALICE.

Services Layer - Logique metier pure (SRP).
Les services sont testables en isolation.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management (inference, optimization)
- ISO/IEC 42010 - Architecture (Service layer, SRP)
- ISO/IEC 25010 - System Quality (maintenabilite)

Services:
- StackingInferenceService: ML stacking pipeline (Phase 2)
- InferenceService: ALI legacy stub (backward compat)
- DataLoader: Repository pattern (MongoDB + Parquet)

Note : ComposerService legacy supprime 2026-04-28 (D5 resorbe). Le vrai
flow CE est `app/api/routes.py::/compose` qui invoque ScenarioGenerator
+ StackingInferenceService + aggregation. Voir ADR-011 (AG eliminé) +
ADR-014 (ALI MC hybride).
"""

from services.data_loader import DataLoader
from services.inference import InferenceService

__all__ = ["InferenceService", "DataLoader"]
