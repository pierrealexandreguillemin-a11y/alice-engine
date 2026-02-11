"""Audit Logger asynchrone - ISO 27001:2022 A.8.15.

Logger d'audit non-bloquant avec batch insert MongoDB.
Fallback fichier si MongoDB echoue (NIST SP 800-92).

Architecture:
- asyncio.Queue fire-and-forget
- Worker background: batch insert toutes les N entries ou T secondes
- _sanitize_query: garde cles, remplace valeurs par "***"
- Fallback: ecriture fichier JSONL si insert MongoDB echoue

ISO Compliance:
- ISO/IEC 27001:2022 A.8.15 - Logging
- NIST SP 800-92 - No silent data loss, structured log entries
- OWASP Logging Cheat Sheet - No PII in logs
- ISO/IEC 5055:2021 - Code Quality (SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from services.audit.types import AuditConfig, AuditEntry, OperationType

logger = structlog.get_logger(__name__)

_MAX_QUEUE_SIZE = 10000
_MAX_SANITIZE_DEPTH = 10
_FALLBACK_PATH = Path("logs/audit_fallback.jsonl")


def _sanitize_query(query: dict[str, Any] | None) -> str:
    """Sanitise une query MongoDB: garde cles, masque valeurs.

    Conforme OWASP: pas de PII dans les logs.
    """
    if query is None:
        return ""
    return _sanitize_value(query, depth=0)


def _sanitize_value(value: Any, *, depth: int = 0) -> str:
    """Sanitise recursivement une valeur (avec limite profondeur)."""
    if depth >= _MAX_SANITIZE_DEPTH:
        return "***"
    if isinstance(value, dict):
        parts = [f"{k}: {_sanitize_value(v, depth=depth + 1)}" for k, v in value.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_sanitize_value(v, depth=depth + 1) for v in value) + "]"
    return "***"


class AuditLogger:
    """Logger d'audit asynchrone avec batch insert.

    Pattern fire-and-forget: log() ne bloque jamais.
    Worker background flush les entries par batch.
    Fallback fichier JSONL si MongoDB insert echoue (NIST SP 800-92).
    """

    def __init__(self, db: Any, config: AuditConfig | None = None) -> None:
        """Initialise le logger d'audit."""
        self.config = config or AuditConfig()
        self._db = db
        self._queue: asyncio.Queue[AuditEntry] = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._running = False
        self._worker_task: asyncio.Task[None] | None = None
        self._dropped_count = 0

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
        if self._dropped_count > 0:
            logger.warning("audit_entries_dropped", count=self._dropped_count)
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
            self._dropped_count += 1
            logger.warning("audit_queue_full", dropped_total=self._dropped_count)

    async def _worker(self) -> None:
        """Worker: batch insert a intervalles reguliers."""
        batch: list[dict[str, Any]] = []
        collection = self._db[self.config.collection_name]

        try:
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
        except asyncio.CancelledError:
            pass  # Expected on shutdown
        finally:
            # Always flush local batch on exit (prevents data loss on cancel)
            if batch:
                await self._insert_batch(collection, batch)

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
        """Insere un batch dans MongoDB. Fallback fichier si echec."""
        try:
            await collection.insert_many(batch)
            logger.debug("audit_batch_inserted", count=len(batch))
        except Exception:
            logger.exception("audit_batch_insert_failed", count=len(batch))
            self._fallback_to_file(batch)

    def _fallback_to_file(self, batch: list[dict[str, Any]]) -> None:
        """Ecrit les entries dans un fichier JSONL en fallback (NIST SP 800-92)."""
        try:
            _FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _FALLBACK_PATH.open("a", encoding="utf-8") as f:
                for entry in batch:
                    f.write(json.dumps(entry) + "\n")
            logger.info("audit_fallback_written", count=len(batch), path=str(_FALLBACK_PATH))
        except Exception:
            logger.exception("audit_fallback_failed", count=len(batch))
