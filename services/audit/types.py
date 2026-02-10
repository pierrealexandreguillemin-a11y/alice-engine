"""Types pour Audit Log - ISO 27001:2022 A.8.15.

Ce module contient les types pour le logging d'audit:
- OperationType: Enum des operations CRUD
- AuditEntry: Model Pydantic d'une entree d'audit
- AuditConfig: Configuration du logger

Chaque entry suit NIST SP 800-92:
timestamp, operation, collection, user, query, result, duration, success.

ISO Compliance:
- ISO/IEC 27001:2022 A.8.15 - Logging
- NIST SP 800-92 - Log Management
- OWASP Logging Cheat Sheet
- ISO/IEC 27034 - Secure Coding (Pydantic validation)
- ISO/IEC 5055:2021 - Code Quality (<100 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OperationType(str, Enum):
    """Types d'operations MongoDB (CRUD)."""

    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"


class AuditEntry(BaseModel):
    """Entree d'audit conforme NIST SP 800-92.

    Attributes
    ----------
        timestamp: UTC ISO 8601
        operation_type: CRUD operation
        collection: Collection MongoDB
        user_source: API client ID, "system", ou correlation ID
        query_summary: Cles uniquement (PAS de valeurs PII)
        result_count: Nombre de resultats
        duration_ms: Duree de l'operation en ms
        success: True si operation reussie
        error_message: Message d'erreur si echec

    """

    timestamp: str
    operation_type: OperationType
    collection: str
    user_source: str = "system"
    query_summary: str = ""
    result_count: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    success: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour insertion MongoDB."""
        data = self.model_dump()
        data["operation_type"] = self.operation_type.value
        return data


class AuditConfig(BaseModel):
    """Configuration du logger d'audit.

    Attributes
    ----------
        enabled: Active/desactive le logging
        collection_name: Nom de la collection MongoDB pour les logs
        batch_size: Nombre d'entrees avant flush
        flush_interval_s: Intervalle de flush en secondes

    """

    enabled: bool = True
    collection_name: str = "audit_logs"
    batch_size: int = Field(default=50, gt=0)
    flush_interval_s: float = Field(default=5.0, gt=0.0)
