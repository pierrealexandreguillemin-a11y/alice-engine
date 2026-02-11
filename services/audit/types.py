"""Types pour Audit Log - ISO 27001:2022 A.8.15.

Ce module contient les types pour le logging d'audit:
- OperationType: Enum des operations CRUD
- AuditEntry: Model Pydantic d'une entree d'audit (immutable)
- AuditConfig: Configuration du logger (immutable)

Chaque entry suit NIST SP 800-92:
timestamp, operation, collection, user, query, result, duration, success.

ISO Compliance:
- ISO/IEC 27001:2022 A.8.15 - Logging
- NIST SP 800-92 - Log Management
- OWASP Logging Cheat Sheet
- ISO/IEC 27034 - Secure Coding (Pydantic validation)
- ISO/IEC 5055:2021 - Code Quality (SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_ISO_8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
# Also validate with datetime.fromisoformat for semantic correctness


class OperationType(StrEnum):
    """Types d'operations MongoDB (CRUD)."""

    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"


class AuditEntry(BaseModel):
    """Entree d'audit conforme NIST SP 800-92 (immutable).

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

    model_config = ConfigDict(frozen=True)

    timestamp: str = Field(min_length=1)
    operation_type: OperationType
    collection: str = Field(min_length=1, max_length=256)
    user_source: str = Field(default="system", min_length=1, max_length=256)
    query_summary: str = Field(default="", max_length=2000)
    result_count: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    success: bool = True
    error_message: str | None = Field(default=None, max_length=5000)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: str) -> str:
        """Validate timestamp is ISO 8601 format with real date values."""
        if not _ISO_8601_RE.match(v):
            msg = f"timestamp must be ISO 8601, got: {v!r}"
            raise ValueError(msg)
        try:
            datetime.fromisoformat(v)
        except ValueError:
            msg = f"timestamp is not a valid date: {v!r}"
            raise  # noqa: B904
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour insertion MongoDB."""
        data = self.model_dump()
        data["operation_type"] = self.operation_type.value
        return data


class AuditConfig(BaseModel):
    """Configuration du logger d'audit (immutable).

    Attributes
    ----------
        enabled: Active/desactive le logging
        collection_name: Nom de la collection MongoDB pour les logs
        batch_size: Nombre d'entrees avant flush
        flush_interval_s: Intervalle de flush en secondes

    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    collection_name: str = Field(default="audit_logs", min_length=1, max_length=256)
    batch_size: int = Field(default=50, gt=0, le=10000)
    flush_interval_s: float = Field(default=5.0, gt=0.0, le=300.0)
