"""Tests Types ISO Audit - ISO 29119.

Document ID: ALICE-TEST-ISO-AUDIT-TYPES
Version: 1.0.0

Tests pour ISOViolation et AuditResult.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from scripts.iso_audit.types import AuditResult, ISOViolation


class TestISOViolation:
    """Tests pour ISOViolation dataclass."""

    def test_create_violation(self) -> None:
        """Test création violation basique."""
        v = ISOViolation(
            norm="ISO 5055",
            severity="HIGH",
            file="test.py",
            message="Fichier trop long",
        )
        assert v.norm == "ISO 5055"
        assert v.severity == "HIGH"
        assert v.file == "test.py"
        assert v.message == "Fichier trop long"
        assert v.fix_suggestion == ""

    def test_violation_with_fix(self) -> None:
        """Test violation avec suggestion de correction."""
        v = ISOViolation(
            norm="ISO 29119",
            severity="CRITICAL",
            file="module.py",
            message="Couverture 0%",
            fix_suggestion="Créer tests unitaires",
        )
        assert v.fix_suggestion == "Créer tests unitaires"


class TestAuditResult:
    """Tests pour AuditResult dataclass."""

    def test_create_empty_result(self) -> None:
        """Test création résultat vide."""
        result = AuditResult()
        assert result.violations == []
        assert result.stats == {}
        assert result.compliant is False
        assert result.timestamp is not None

    def test_add_violation(self) -> None:
        """Test ajout de violation."""
        result = AuditResult()
        result.add_violation(
            norm="ISO 5055",
            severity="MEDIUM",
            file="test.py",
            message="300+ lignes",
            fix="Refactorer",
        )
        assert len(result.violations) == 1
        assert result.violations[0].norm == "ISO 5055"
        assert result.violations[0].severity == "MEDIUM"

    def test_to_dict(self) -> None:
        """Test conversion en dictionnaire."""
        result = AuditResult()
        result.add_violation("ISO 5055", "CRITICAL", "a.py", "msg1")
        result.add_violation("ISO 5055", "HIGH", "b.py", "msg2")
        result.add_violation("ISO 29119", "MEDIUM", "c.py", "msg3")
        result.stats = {"test": 123}
        result.compliant = True

        d = result.to_dict()
        assert d["compliant"] is True
        assert d["stats"] == {"test": 123}
        assert len(d["violations"]) == 3
        assert d["summary"]["total"] == 3
        assert d["summary"]["critical"] == 1
        assert d["summary"]["high"] == 1
        assert d["summary"]["medium"] == 1
        assert d["summary"]["low"] == 0

    def test_to_dict_empty(self) -> None:
        """Test conversion dictionnaire vide."""
        result = AuditResult()
        d = result.to_dict()
        assert d["summary"]["total"] == 0
