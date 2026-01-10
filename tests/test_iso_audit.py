"""Tests pour scripts/iso_audit - ISO 29119.

Ce module teste les fonctionnalités d'audit ISO:
- ISOViolation et AuditResult (types.py)
- Fonctions d'audit (auditors.py)
- Génération de rapports (report.py)

ISO 29119: Test coverage pour le module d'audit ISO.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.iso_audit.auditors import (
    _check_coverage,
    _check_file_size,
    audit_iso_5055_maintainability,
    audit_iso_5259_data_quality,
    audit_iso_15289_documentation,
    audit_iso_25010_quality,
    count_lines,
)
from scripts.iso_audit.report import (
    _build_markdown_content,
    _build_violations_by_norm,
    _generate_json_report,
    _generate_markdown_report,
    generate_report,
)
from scripts.iso_audit.types import AuditResult, ISOViolation

# =============================================================================
# Tests pour types.py
# =============================================================================


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


# =============================================================================
# Tests pour auditors.py
# =============================================================================


class TestCountLines:
    """Tests pour count_lines."""

    def test_count_lines_file(self, tmp_path: Path) -> None:
        """Test comptage lignes fichier existant."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\n")
        assert count_lines(test_file) == 3

    def test_count_lines_empty(self, tmp_path: Path) -> None:
        """Test fichier vide."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")
        assert count_lines(test_file) == 0

    def test_count_lines_nonexistent(self, tmp_path: Path) -> None:
        """Test fichier inexistant."""
        assert count_lines(tmp_path / "missing.py") == 0


class TestCheckFileSize:
    """Tests pour _check_file_size."""

    def test_file_under_limit(self) -> None:
        """Test fichier sous la limite."""
        result = AuditResult()
        _check_file_size(result, Path("small.py"), 100)
        assert len(result.violations) == 0

    def test_file_medium_violation(self) -> None:
        """Test fichier > 300 lignes (MEDIUM)."""
        result = AuditResult()
        _check_file_size(result, Path("medium.py"), 350)
        assert len(result.violations) == 1
        assert result.violations[0].severity == "MEDIUM"

    def test_file_high_violation(self) -> None:
        """Test fichier > 600 lignes (HIGH)."""
        result = AuditResult()
        _check_file_size(result, Path("large.py"), 700)
        assert len(result.violations) == 1
        assert result.violations[0].severity == "HIGH"

    def test_file_critical_violation(self) -> None:
        """Test fichier > 900 lignes (CRITICAL)."""
        result = AuditResult()
        _check_file_size(result, Path("huge.py"), 1000)
        assert len(result.violations) == 1
        assert result.violations[0].severity == "CRITICAL"


class TestCheckCoverage:
    """Tests pour _check_coverage."""

    def test_coverage_good(self) -> None:
        """Test couverture suffisante."""
        result = AuditResult()
        _check_coverage(result, Path("good.py"), 85.0)
        assert len(result.violations) == 0

    def test_coverage_zero(self) -> None:
        """Test couverture 0% (CRITICAL)."""
        result = AuditResult()
        _check_coverage(result, Path("untested.py"), 0.0)
        assert len(result.violations) == 1
        assert result.violations[0].severity == "CRITICAL"

    def test_coverage_low(self) -> None:
        """Test couverture < 50% (HIGH)."""
        result = AuditResult()
        _check_coverage(result, Path("low.py"), 30.0)
        assert len(result.violations) == 1
        assert result.violations[0].severity == "HIGH"

    def test_coverage_medium(self) -> None:
        """Test couverture < 80% (MEDIUM)."""
        result = AuditResult()
        _check_coverage(result, Path("medium.py"), 65.0)
        assert len(result.violations) == 1
        assert result.violations[0].severity == "MEDIUM"


class TestAuditISO5055:
    """Tests pour audit_iso_5055_maintainability."""

    def test_audit_empty_project(self, tmp_path: Path) -> None:
        """Test audit projet vide."""
        result = AuditResult()
        audit_iso_5055_maintainability(result, tmp_path)
        assert "iso_5055" in result.stats
        assert result.stats["iso_5055"]["total_files"] == 0

    def test_audit_with_files(self, tmp_path: Path) -> None:
        """Test audit avec fichiers Python."""
        # Créer structure
        app_dir = tmp_path / "app"
        app_dir.mkdir()

        # Fichier conforme
        (app_dir / "small.py").write_text("# Comment\nprint('hello')\n")

        # Fichier trop long
        (app_dir / "large.py").write_text("x = 1\n" * 350)

        result = AuditResult()
        audit_iso_5055_maintainability(result, tmp_path)

        assert result.stats["iso_5055"]["total_files"] == 2
        assert result.stats["iso_5055"]["compliant_files"] == 1
        assert len(result.violations) == 1

    def test_audit_excludes_tests(self, tmp_path: Path) -> None:
        """Test exclusion fichiers test."""
        (tmp_path / "test_something.py").write_text("x = 1\n" * 500)

        result = AuditResult()
        audit_iso_5055_maintainability(result, tmp_path)

        assert result.stats["iso_5055"]["total_files"] == 0


class TestAuditISO15289:
    """Tests pour audit_iso_15289_documentation."""

    def test_audit_missing_docs(self, tmp_path: Path) -> None:
        """Test audit documents manquants."""
        result = AuditResult()
        audit_iso_15289_documentation(result, tmp_path)

        assert "iso_15289" in result.stats
        assert result.stats["iso_15289"]["found_docs"] == 0
        assert len(result.violations) > 0

    def test_audit_with_readme(self, tmp_path: Path) -> None:
        """Test audit avec README."""
        (tmp_path / "README.md").write_text("# Project\n" + "Content " * 100)

        result = AuditResult()
        audit_iso_15289_documentation(result, tmp_path)

        assert result.stats["iso_15289"]["found_docs"] == 1

    def test_audit_short_doc(self, tmp_path: Path) -> None:
        """Test document trop court."""
        (tmp_path / "README.md").write_text("# Short")

        result = AuditResult()
        audit_iso_15289_documentation(result, tmp_path)

        # Chercher violation LOW pour contenu trop court
        low_violations = [v for v in result.violations if v.severity == "LOW"]
        assert len(low_violations) == 1


class TestAuditISO5259:
    """Tests pour audit_iso_5259_data_quality."""

    def test_audit_no_fe_files(self, tmp_path: Path) -> None:
        """Test audit sans fichiers feature."""
        result = AuditResult()
        audit_iso_5259_data_quality(result, tmp_path)

        assert "iso_5259" in result.stats
        assert result.stats["iso_5259"]["files_checked"] == 0

    def test_audit_fillna_violation(self, tmp_path: Path) -> None:
        """Test détection fillna(0)."""
        fe_dir = tmp_path / "features"
        fe_dir.mkdir()
        (fe_dir / "feature_engineering.py").write_text("df.fillna(0)")

        result = AuditResult()
        audit_iso_5259_data_quality(result, tmp_path)

        assert len(result.violations) == 1
        assert "fillna()" in result.violations[0].message


class TestAuditISO25010:
    """Tests pour audit_iso_25010_quality."""

    def test_audit_empty_project(self, tmp_path: Path) -> None:
        """Test audit projet vide."""
        result = AuditResult()
        audit_iso_25010_quality(result, tmp_path)

        assert "iso_25010" in result.stats
        assert result.stats["iso_25010"]["srp_violations"] == 0

    def test_audit_srp_violation(self, tmp_path: Path) -> None:
        """Test détection violation SRP (>20 fonctions)."""
        # Créer fichier avec beaucoup de fonctions
        funcs = "\n".join([f"def func_{i}():\n    pass\n" for i in range(25)])
        (tmp_path / "big_module.py").write_text(funcs)

        result = AuditResult()
        audit_iso_25010_quality(result, tmp_path)

        assert result.stats["iso_25010"]["srp_violations"] == 1


# =============================================================================
# Tests pour report.py
# =============================================================================


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
        assert "| 1 |" in md  # Violations totales


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
