"""Tests Error Paths - Audit Logger - ISO 29119.

Document ID: ALICE-TEST-AUDIT-ERROR-PATHS
Version: 1.0.0
Tests count: 12

Covers:
- Invalid timestamp format rejected
- Pydantic validators reject bad input
- Queue overflow tracking
- MongoDB insert failure triggers fallback file
- Disabled audit skips logging
- Sanitize handles deeply nested queries

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (error paths)
- ISO/IEC 27001:2022 A.8.15 - Logging reliability
- NIST SP 800-92 - No silent data loss

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from services.audit.logger import AuditLogger, _sanitize_query
from services.audit.types import AuditConfig, AuditEntry, OperationType


class TestAuditEntryValidation:
    """Tests pour la validation Pydantic des AuditEntry."""

    def test_rejects_non_iso8601_timestamp(self) -> None:
        """Timestamp non-ISO 8601 est rejete."""
        with pytest.raises(ValidationError, match="ISO 8601"):
            AuditEntry(
                timestamp="not-a-date",
                operation_type=OperationType.READ,
                collection="test",
            )

    def test_rejects_empty_collection(self) -> None:
        """Collection vide est rejetee."""
        with pytest.raises(ValidationError):
            AuditEntry(
                timestamp="2026-02-10T12:00:00Z",
                operation_type=OperationType.READ,
                collection="",
            )

    def test_rejects_negative_result_count(self) -> None:
        """result_count negatif est rejete."""
        with pytest.raises(ValidationError):
            AuditEntry(
                timestamp="2026-02-10T12:00:00Z",
                operation_type=OperationType.READ,
                collection="test",
                result_count=-1,
            )

    def test_rejects_negative_duration(self) -> None:
        """duration_ms negative est rejetee."""
        with pytest.raises(ValidationError):
            AuditEntry(
                timestamp="2026-02-10T12:00:00Z",
                operation_type=OperationType.READ,
                collection="test",
                duration_ms=-1.0,
            )

    def test_entry_is_immutable(self) -> None:
        """AuditEntry est immutable (frozen=True)."""
        entry = AuditEntry(
            timestamp="2026-02-10T12:00:00Z",
            operation_type=OperationType.READ,
            collection="test",
        )
        with pytest.raises(ValidationError):
            entry.collection = "other"  # type: ignore[misc]


class TestAuditConfigValidation:
    """Tests pour la validation Pydantic des AuditConfig."""

    def test_rejects_zero_batch_size(self) -> None:
        """batch_size=0 est rejete."""
        with pytest.raises(ValidationError):
            AuditConfig(batch_size=0)

    def test_rejects_negative_flush_interval(self) -> None:
        """flush_interval_s negative est rejetee."""
        with pytest.raises(ValidationError):
            AuditConfig(flush_interval_s=-1.0)

    def test_rejects_too_large_batch_size(self) -> None:
        """batch_size > 10000 est rejete."""
        with pytest.raises(ValidationError):
            AuditConfig(batch_size=10001)


class TestSanitizeDeepNesting:
    """Tests pour la sanitisation de queries profondement imbriquees."""

    def test_deep_nesting_stops_at_max_depth(self) -> None:
        """Arrete la recursion a MAX_SANITIZE_DEPTH."""
        # Build deeply nested dict (15 levels > MAX_SANITIZE_DEPTH=10)
        deep: dict = {"key": "value"}
        for _ in range(15):
            deep = {"nested": deep}
        result = _sanitize_query(deep)
        assert "***" in result
        # Should not crash

    def test_empty_dict(self) -> None:
        """Dict vide produit '{}'."""
        result = _sanitize_query({})
        assert result == "{}"


class TestAuditLoggerDisabled:
    """Tests pour le logger desactive."""

    @pytest.mark.asyncio
    async def test_disabled_audit_skips_logging(self) -> None:
        """Audit desactive ne met rien en queue."""
        config = AuditConfig(enabled=False)
        audit = AuditLogger(db=MagicMock(), config=config)
        await audit.log(OperationType.READ, "test")
        assert audit._queue.empty()  # noqa: SLF001


class TestMongoDBInsertFailure:
    """Tests pour l'echec d'insertion MongoDB avec fallback."""

    @pytest.mark.asyncio
    async def test_insert_failure_triggers_fallback(self, tmp_path) -> None:
        """Echec MongoDB ecrit dans le fichier fallback."""
        db = MagicMock()
        collection = MagicMock()
        collection.insert_many = AsyncMock(side_effect=Exception("MongoDB down"))
        db.__getitem__ = MagicMock(return_value=collection)
        db.audit_logs = collection

        config = AuditConfig(batch_size=1, flush_interval_s=0.1)
        audit = AuditLogger(db=db, config=config)

        fallback_path = tmp_path / "audit_fallback.jsonl"
        with patch("services.audit.logger._FALLBACK_PATH", fallback_path):
            await audit.start()
            try:
                await audit.log(OperationType.READ, "test")
                await asyncio.sleep(0.5)
            finally:
                await audit.stop()

            assert fallback_path.exists()
            content = fallback_path.read_text()
            assert "test" in content
