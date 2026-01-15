"""Tests Auditors ISO Audit - ISO 29119.

Document ID: ALICE-TEST-ISO-AUDIT-AUDITORS
Version: 1.0.0

Tests pour les fonctions d'audit ISO.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

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
from scripts.iso_audit.types import AuditResult


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
        app_dir = tmp_path / "app"
        app_dir.mkdir()

        (app_dir / "small.py").write_text("# Comment\nprint('hello')\n")
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
        funcs = "\n".join([f"def func_{i}():\n    pass\n" for i in range(25)])
        (tmp_path / "big_module.py").write_text(funcs)

        result = AuditResult()
        audit_iso_25010_quality(result, tmp_path)

        assert result.stats["iso_25010"]["srp_violations"] == 1
