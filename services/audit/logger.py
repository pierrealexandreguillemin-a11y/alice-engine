"""Audit Logger asynchrone - ISO 27001:2022 A.8.15.

Logger d'audit non-bloquant avec batch insert MongoDB.

Architecture:
- asyncio.Queue fire-and-forget
- Worker background: batch insert toutes les N entries ou T secondes
- _sanitize_query: garde cles, remplace valeurs par "***"

ISO Compliance:
- ISO/IEC 27001:2022 A.8.15 - Logging
- NIST SP 800-92 - Structured log entries
- OWASP Logging Cheat Sheet - No PII in logs
- ISO/IEC 5055:2021 - Code Quality (<140 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog

from services.audit.types import AuditConfig, AuditEntry, OperationType

logger = structlog.get_logger(__name__)

_MAX_QUEUE_SIZE = 10000


def _sanitize_query(query: dict[str, Any] | None) -> str:
    """Sanitise une query MongoDB: garde cles, masque valeurs.

    Conforme OWASP: pas de PII dans les logs.
    """
    if query is None:
        return ""
    return _sanitize_value(query)


def _sanitize_value(value: Any) -> str:
    """Sanitise recursivement une valeur."""
    if isinstance(value, dict):
        parts = [f"{k}: {_sanitize_value(v)}" for k, v in value.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_sanitize_value(v) for v in value) + "]"
    return "***"


class AuditLogger:
    """Logger d'audit asynchrone avec batch insert.

    Pattern fire-and-forget: log() ne bloque jamais.
    Worker background flush les entries par batch.
    """

    def __init__(self, db: Any, config: AuditConfig | None = None) -> None:
        """Initialise le logger d'audit."""
        self.config = config or AuditConfig()
        self._db = db
        self._queue: asyncio.Queue[AuditEntry] = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._running = False
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Demarre le worker background.

        Recommande: creer un TTL index sur la collection pour rotation auto:
            db.audit_logs.createIndex({"timestamp": 1}, {expireAfterSeconds: 7776000})
        """
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("audit_logger_started", collection=self.config.collection_name)

    async def stop(self) -> None:
        """Arrete le worker et flush les entrees restantes."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await self._flush_remaining()
        logger.info("audit_logger_stopped")

    async def log(  # noqa: PLR0913
        self,
        operation_type: OperationType,
        collection: str,
        *,
        user_source: str = "system",
        query: dict[str, Any] | None = None,
        result_count: int = 0,
        duration_ms: float = 0.0,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Enregistre une operation (fire-and-forget)."""
        if not self.config.enabled:
            return
        entry = AuditEntry(
            timestamp=datetime.now(tz=UTC).isoformat(),
            operation_type=operation_type,
            collection=collection,
            user_source=user_source,
            query_summary=_sanitize_query(query),
            result_count=result_count,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
        )
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            logger.warning("audit_queue_full")

    async def _worker(self) -> None:
        """Worker: batch insert a intervalles reguliers."""
        batch: list[dict[str, Any]] = []
        collection = self._db[self.config.collection_name]

        while self._running:
            try:
                entry = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.config.flush_interval_s,
                )
                batch.append(entry.to_dict())
                if len(batch) >= self.config.batch_size:
                    await self._insert_batch(collection, batch)
                    batch = []
            except TimeoutError:
                if batch:
                    await self._insert_batch(collection, batch)
                    batch = []

    async def _flush_remaining(self) -> None:
        """Flush les entrees restantes dans la queue."""
        batch: list[dict[str, Any]] = []
        collection = self._db[self.config.collection_name]
        while not self._queue.empty():
            try:
                entry = self._queue.get_nowait()
                batch.append(entry.to_dict())
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._insert_batch(collection, batch)

    async def _insert_batch(
        self,
        collection: Any,
        batch: list[dict[str, Any]],
    ) -> None:
        """Insere un batch dans MongoDB."""
        try:
            await collection.insert_many(batch)
            logger.debug("audit_batch_inserted", count=len(batch))
        except Exception:
            logger.exception("audit_batch_insert_failed", count=len(batch))
