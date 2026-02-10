"""Fixtures Audit Log - ISO 29119.

Document ID: ALICE-TEST-AUDIT-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.audit.types import AuditConfig


@pytest.fixture
def mock_db() -> MagicMock:
    """Mock MongoDB database."""
    db = MagicMock()
    collection = MagicMock()
    collection.insert_many = AsyncMock(return_value=MagicMock(inserted_ids=["id1"]))
    db.__getitem__ = MagicMock(return_value=collection)
    db.audit_logs = collection
    return db


@pytest.fixture
def audit_config() -> AuditConfig:
    """Config d'audit pour tests."""
    return AuditConfig(
        enabled=True,
        collection_name="audit_logs",
        batch_size=5,
        flush_interval_s=0.1,
    )
