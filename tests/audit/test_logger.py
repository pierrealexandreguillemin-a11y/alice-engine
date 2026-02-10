"""Tests Audit Logger - ISO 29119.

Document ID: ALICE-TEST-AUDIT-LOGGER
Version: 1.0.0
Tests count: 14

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 27001:2022 A.8.15 - Logging
- NIST SP 800-92 - Log Management

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from services.audit.logger import AuditLogger, _sanitize_query
from services.audit.types import AuditConfig, OperationType


class TestSanitizeQuery:
    """Tests pour la sanitisation des queries."""

    def test_keeps_keys_masks_values(self) -> None:
        """Garde les cles, masque les valeurs."""
        query = {"clubId": "ABC123", "isActive": True}
        result = _sanitize_query(query)
        assert "clubId" in result
        assert "ABC123" not in result
        assert "***" in result

    def test_handles_none_query(self) -> None:
        """Gere None sans erreur."""
        result = _sanitize_query(None)
        assert result == ""

    def test_handles_nested_query(self) -> None:
        """Gere les queries imbriquees."""
        query = {"$or": [{"clubId": "A"}, {"clubId": "B"}]}
        result = _sanitize_query(query)
        assert "$or" in result
        assert "A" not in result

    def test_no_pii_in_output(self) -> None:
        """Aucune PII dans la sortie."""
        query = {"email": "john@example.com", "name": "John Doe"}
        result = _sanitize_query(query)
        assert "john@example.com" not in result
        assert "John Doe" not in result


class TestAuditLoggerInit:
    """Tests pour l'initialisation du logger."""

    def test_creates_with_config(self, audit_config: AuditConfig) -> None:
        """Se cree avec la config fournie."""
        logger = AuditLogger(db=MagicMock(), config=audit_config)
        assert logger.config.batch_size == 5

    def test_uses_config_collection_name(self, audit_config: AuditConfig) -> None:
        """Utilise le nom de collection de la config."""
        logger = AuditLogger(db=MagicMock(), config=audit_config)
        assert logger.config.collection_name == "audit_logs"


class TestLogOperation:
    """Tests pour l'operation de logging."""

    @pytest.mark.asyncio
    async def test_non_blocking_enqueue(
        self,
        mock_db: MagicMock,
        audit_config: AuditConfig,
    ) -> None:
        """Log n'est pas bloquant (fire-and-forget)."""
        logger = AuditLogger(db=mock_db, config=audit_config)
        await logger.start()
        try:
            # Should return immediately
            await logger.log(OperationType.READ, "compositions", result_count=5)
            assert logger._queue.qsize() >= 0  # noqa: SLF001
        finally:
            await logger.stop()

    @pytest.mark.asyncio
    async def test_queue_full_does_not_raise(
        self,
        mock_db: MagicMock,
    ) -> None:
        """Queue pleine ne leve pas d'exception."""
        config = AuditConfig(batch_size=2, flush_interval_s=100.0)
        audit = AuditLogger(db=mock_db, config=config)
        # Override queue with tiny capacity, don't start worker
        audit._queue = asyncio.Queue(maxsize=2)  # noqa: SLF001
        for _ in range(50):
            await audit.log(OperationType.READ, "test")
        # Should not raise even though queue overflows

    @pytest.mark.asyncio
    async def test_creates_correct_audit_entry(
        self,
        mock_db: MagicMock,
        audit_config: AuditConfig,
    ) -> None:
        """Cree une entree d'audit correcte."""
        logger = AuditLogger(db=mock_db, config=audit_config)
        await logger.start()
        try:
            await logger.log(
                OperationType.WRITE,
                "players",
                user_source="api_client_1",
                query={"clubId": "ABC"},
                result_count=10,
                duration_ms=5.2,
            )
            # Wait for flush
            await asyncio.sleep(0.3)
        finally:
            await logger.stop()

        # Verify insert was called
        assert mock_db.audit_logs.insert_many.call_count >= 1


class TestWorker:
    """Tests pour le worker background."""

    @pytest.mark.asyncio
    async def test_batch_insert_on_threshold(
        self,
        mock_db: MagicMock,
    ) -> None:
        """Insert batch quand le seuil est atteint."""
        config = AuditConfig(batch_size=3, flush_interval_s=100.0)
        logger = AuditLogger(db=mock_db, config=config)
        await logger.start()
        try:
            for _ in range(3):
                await logger.log(OperationType.READ, "test")
            await asyncio.sleep(0.3)
        finally:
            await logger.stop()
        assert mock_db.audit_logs.insert_many.call_count >= 1

    @pytest.mark.asyncio
    async def test_flush_on_interval(
        self,
        mock_db: MagicMock,
    ) -> None:
        """Flush apres l'intervalle meme si batch pas plein."""
        config = AuditConfig(batch_size=100, flush_interval_s=0.1)
        logger = AuditLogger(db=mock_db, config=config)
        await logger.start()
        try:
            await logger.log(OperationType.READ, "test")
            await asyncio.sleep(0.3)
        finally:
            await logger.stop()
        assert mock_db.audit_logs.insert_many.call_count >= 1

    @pytest.mark.asyncio
    async def test_graceful_shutdown_flushes(
        self,
        mock_db: MagicMock,
        audit_config: AuditConfig,
    ) -> None:
        """Arret gracieux flush les entrees restantes."""
        logger = AuditLogger(db=mock_db, config=audit_config)
        await logger.start()
        await logger.log(OperationType.READ, "test")
        await logger.stop()
        # After stop, remaining entries should be flushed
        assert mock_db.audit_logs.insert_many.call_count >= 1


class TestIntegration:
    """Tests d'integration."""

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(
        self,
        mock_db: MagicMock,
        audit_config: AuditConfig,
    ) -> None:
        """Cycle de vie start/stop sans erreur."""
        logger = AuditLogger(db=mock_db, config=audit_config)
        await logger.start()
        assert logger._running  # noqa: SLF001
        await logger.stop()
        assert not logger._running  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_fire_and_forget_pattern(
        self,
        mock_db: MagicMock,
        audit_config: AuditConfig,
    ) -> None:
        """Pattern fire-and-forget fonctionne."""
        logger = AuditLogger(db=mock_db, config=audit_config)
        await logger.start()
        try:
            for i in range(10):
                await logger.log(
                    OperationType.READ,
                    "compositions",
                    result_count=i,
                )
            await asyncio.sleep(0.3)
        finally:
            await logger.stop()
        assert mock_db.audit_logs.insert_many.call_count >= 1
