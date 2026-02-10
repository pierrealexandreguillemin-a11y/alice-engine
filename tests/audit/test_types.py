"""Tests Audit Types - ISO 29119.

Document ID: ALICE-TEST-AUDIT-TYPES
Version: 1.0.0
Tests count: 8

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 27001:2022 A.8.15 - Logging

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

from services.audit.types import AuditConfig, AuditEntry, OperationType


class TestOperationType:
    """Tests pour l'enum OperationType."""

    def test_values_are_strings(self) -> None:
        """Les valeurs sont des strings."""
        assert OperationType.READ.value == "read"
        assert isinstance(OperationType.WRITE.value, str)

    def test_all_crud_operations_covered(self) -> None:
        """Toutes les operations CRUD sont couvertes."""
        ops = {o.value for o in OperationType}
        assert ops == {"read", "write", "update", "delete"}


class TestAuditEntry:
    """Tests pour AuditEntry."""

    def test_creation_with_required_fields(self) -> None:
        """Creation avec champs requis."""
        entry = AuditEntry(
            timestamp="2026-02-10T12:00:00Z",
            operation_type=OperationType.READ,
            collection="compositions",
        )
        assert entry.collection == "compositions"

    def test_to_dict_returns_all_fields(self) -> None:
        """to_dict retourne tous les champs."""
        entry = AuditEntry(
            timestamp="2026-02-10T12:00:00Z",
            operation_type=OperationType.WRITE,
            collection="players",
            result_count=10,
        )
        d = entry.to_dict()
        assert "timestamp" in d
        assert "operation_type" in d
        assert "collection" in d
        assert "result_count" in d
        assert d["operation_type"] == "write"

    def test_timestamp_is_utc_iso_format(self) -> None:
        """Le timestamp est au format UTC ISO 8601."""
        entry = AuditEntry(
            timestamp="2026-02-10T12:00:00+00:00",
            operation_type=OperationType.READ,
            collection="test",
        )
        assert "2026" in entry.timestamp
        assert "T" in entry.timestamp

    def test_default_error_message_none(self) -> None:
        """Le message d'erreur est None par defaut."""
        entry = AuditEntry(
            timestamp="2026-02-10T12:00:00Z",
            operation_type=OperationType.READ,
            collection="test",
        )
        assert entry.error_message is None


class TestAuditConfig:
    """Tests pour AuditConfig."""

    def test_default_values(self) -> None:
        """Les valeurs par defaut sont correctes."""
        config = AuditConfig()
        assert config.enabled is True
        assert config.collection_name == "audit_logs"
        assert config.batch_size == 50
        assert config.flush_interval_s == 5.0

    def test_custom_values(self) -> None:
        """Les valeurs custom sont respectees."""
        config = AuditConfig(
            enabled=False,
            collection_name="custom_audit",
            batch_size=100,
            flush_interval_s=10.0,
        )
        assert config.enabled is False
        assert config.collection_name == "custom_audit"
        assert config.batch_size == 100
