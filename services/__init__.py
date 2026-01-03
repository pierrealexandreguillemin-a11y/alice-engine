# services/__init__.py
"""
Services Layer - Logique metier pure (SRP)

Ce module contient la logique metier sans I/O direct.
Les services sont testables en isolation.

@see ISO 42010 - Architecture (Service layer)
"""

from services.composer import ComposerService
from services.data_loader import DataLoader
from services.inference import InferenceService

__all__ = ["InferenceService", "ComposerService", "DataLoader"]
