"""Types pour Model Rollback - ISO 23894.

Ce module contient les types pour la detection de degradation
et le rollback automatique:
- DegradationThresholds: seuils de degradation
- RollbackDecision: decision de rollback
- RollbackResult: resultat du rollback

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 27034 - Secure Coding (Pydantic validation)
- ISO/IEC 5055:2021 - Code Quality (SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

VERSION_PATTERN = re.compile(r"^v\d{8}_\d{6}$")


def validate_version_format(version: str) -> bool:
    """Verifie que la version matche le format vYYYYMMDD_HHMMSS (ISO 27034)."""
    return bool(VERSION_PATTERN.match(version))


class DegradationThresholds(BaseModel):
    """Seuils de degradation pour declenchement rollback.

    Ref: Uber ML Safety - AUC/accuracy drop thresholds.

    Attributes
    ----------
        auc_drop_pct: Drop AUC maximum tolere (%)
        accuracy_drop_pct: Drop accuracy maximum tolere (%)
    """

    auc_drop_pct: float = Field(default=2.0, gt=0.0, le=100.0)
    accuracy_drop_pct: float = Field(default=3.0, gt=0.0, le=100.0)


class RollbackDecision(BaseModel):
    """Decision de rollback avec justification.

    Attributes
    ----------
        should_rollback: True si rollback necessaire
        reason: Raison du rollback (ou "No degradation")
        current_version: Version courante
        target_version: Version cible du rollback
        metrics_comparison: Comparaison des metriques
        timestamp: Date/heure de la decision
    """

    should_rollback: bool
    reason: str
    current_version: str
    target_version: str | None = None
    metrics_comparison: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour serialisation."""
        return self.model_dump()


class RollbackResult(BaseModel):
    """Resultat d'execution du rollback.

    Attributes
    ----------
        success: True si rollback effectue
        rolled_back_from: Version source
        rolled_back_to: Version cible
        reason: Raison du rollback
        timestamp: Date/heure du rollback
        error_message: Message d'erreur si echec
    """

    success: bool
    rolled_back_from: str
    rolled_back_to: str
    reason: str
    timestamp: str = ""
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour serialisation."""
        return self.model_dump()
