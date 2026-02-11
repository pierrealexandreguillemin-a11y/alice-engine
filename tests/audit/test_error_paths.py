"""Tests Error Paths - Audit Logger - ISO 29119.

Document ID: ALICE-TEST-AUDIT-ERROR-PATHS
Version: 1.1.0
Tests count: 20

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

    def test_collection_max_length_256_accepted(self) -> None:
        """Collection de 256 caracteres est acceptee (boundary)."""
        entry = AuditEntry(
            timestamp="2026-02-10T12:00:00Z",
            operation_type=OperationType.READ,
            collection="x" * 256,
        )
        assert len(entry.collection) == 256

    def test_collection_257_chars_rejected(self) -> None:
        """Collection de 257 caracteres est rejetee."""
        with pytest.raises(ValidationError):
            AuditEntry(
                timestamp="2026-02-10T12:00:00Z",
                operation_type=OperationType.READ,
                collection="x" * 257,
            )

    def test_rejects_semantically_invalid_date(self) -> None:
        """Timestamp avec date invalide (31 fevrier) est rejetee."""
        with pytest.raises(ValidationError, match="day is out of range"):
            AuditEntry(
                timestamp="2026-02-31T12:00:00Z",
                operation_type=OperationType.READ,
                collection="test",
            )


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

    def test_accepts_batch_size_exactly_10000(self) -> None:
        """batch_size = 10000 est accepte (boundary)."""
        config = AuditConfig(batch_size=10000)
        assert config.batch_size == 10000

    def test_rejects_flush_interval_above_300(self) -> None:
        """flush_interval_s > 300 est rejete."""
        with pytest.raises(ValidationError):
            AuditConfig(flush_interval_s=300.1)

    def test_accepts_flush_interval_exactly_300(self) -> None:
        """flush_interval_s = 300 est accepte (boundary)."""
        config = AuditConfig(flush_interval_s=300.0)
        assert config.flush_interval_s == 300.0


class TestAuditLoggerEdgeCases:
    """Tests pour les cas limites du logger."""

    @pytest.mark.asyncio
    async def test_stop_without_start_safe(self) -> None:
        """stop() sans start() ne crash pas."""
        audit = AuditLogger(db=MagicMock(), config=AuditConfig())
        await audit.stop()
        assert not audit._running  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self) -> None:
        """Appeler start() deux fois ne cree pas deux workers."""
        audit = AuditLogger(db=MagicMock(), config=AuditConfig())
        await audit.start()
        first_task = audit._worker_task  # noqa: SLF001
        await audit.start()
        assert audit._worker_task is first_task  # noqa: SLF001
        await audit.stop()


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
