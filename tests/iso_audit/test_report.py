"""Tests Report ISO Audit - ISO 29119.

Document ID: ALICE-TEST-ISO-AUDIT-REPORT
Version: 1.0.0

Tests pour la génération de rapports.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import json
from pathlib import Path

from scripts.iso_audit.report import (
    _build_markdown_content,
    _build_violations_by_norm,
    _generate_json_report,
    _generate_markdown_report,
    generate_report,
)
from scripts.iso_audit.types import AuditResult, ISOViolation


class TestGenerateReport:
    """Tests pour generate_report."""

    def test_generate_report_creates_files(self, tmp_path: Path) -> None:
        """Test création fichiers rapport."""
        result = AuditResult()
        result.add_violation("ISO 5055", "HIGH", "test.py", "Too long")

        generate_report(result, tmp_path)

        assert (tmp_path / "iso-audit" / "iso-audit-report.json").exists()
        assert (tmp_path / "iso-audit" / "ISO_AUDIT_REPORT.md").exists()

    def test_json_report_content(self, tmp_path: Path) -> None:
        """Test contenu rapport JSON."""
        result = AuditResult()
        result.add_violation("ISO 5055", "CRITICAL", "a.py", "msg")
        result.compliant = True

        output_path = tmp_path / "iso-audit"
        output_path.mkdir(parents=True)
        _generate_json_report(result, output_path)

        json_file = output_path / "iso-audit-report.json"
        data = json.loads(json_file.read_text())

        assert data["compliant"] is True
        assert len(data["violations"]) == 1

    def test_markdown_report_content(self, tmp_path: Path) -> None:
        """Test contenu rapport Markdown."""
        result = AuditResult()
        result.add_violation("ISO 5055", "HIGH", "test.py", "Too long")
        result.compliant = False

        output_path = tmp_path / "iso-audit"
        output_path.mkdir(parents=True)
        _generate_markdown_report(result, output_path)

        md_file = output_path / "ISO_AUDIT_REPORT.md"
        content = md_file.read_text()

        assert "Rapport Audit ISO" in content
        assert "[FAIL] NON CONFORME" in content
        assert "ISO 5055" in content


class TestBuildMarkdownContent:
    """Tests pour _build_markdown_content."""

    def test_compliant_status(self) -> None:
        """Test statut conforme."""
        result = AuditResult()
        result.compliant = True

        md = _build_markdown_content(result)
        assert "[OK] CONFORME" in md

    def test_non_compliant_status(self) -> None:
        """Test statut non conforme."""
        result = AuditResult()
        result.compliant = False
        result.add_violation("ISO 5055", "HIGH", "a.py", "msg")

        md = _build_markdown_content(result)
        assert "[FAIL] NON CONFORME" in md
        assert "| 1 |" in md


class TestBuildViolationsByNorm:
    """Tests pour _build_violations_by_norm."""

    def test_empty_violations(self) -> None:
        """Test liste vide."""
        content = _build_violations_by_norm([])
        assert content == ""

    def test_grouped_by_norm(self) -> None:
        """Test regroupement par norme."""
        violations = [
            ISOViolation("ISO 5055", "HIGH", "a.py", "msg1"),
            ISOViolation("ISO 5055", "MEDIUM", "b.py", "msg2"),
            ISOViolation("ISO 29119", "LOW", "c.py", "msg3"),
        ]

        content = _build_violations_by_norm(violations)

        assert "### ISO 5055" in content
        assert "### ISO 29119" in content
        assert "a.py" in content
        assert "b.py" in content
        assert "c.py" in content
