"""Tests Retention - ISO 29119.

Document ID: ALICE-TEST-MODEL-RETENTION
Version: 1.0.0
Tests: 1 classes

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

from scripts.model_registry import (
    apply_retention_policy,
    get_retention_status,
    list_model_versions,
)


class TestRetentionPolicy:
    """Tests pour politique de rétention (ISO 27001)."""

    def test_retention_under_limit(self, tmp_path: Path) -> None:
        """Test rétention sous la limite."""
        # Créer 3 versions (sous DEFAULT_MAX_VERSIONS=10)
        for i in range(3):
            v = tmp_path / f"v2024010{i}_120000"
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        deleted = apply_retention_policy(tmp_path)

        assert len(deleted) == 0
        assert len(list_model_versions(tmp_path)) == 3

    def test_retention_over_limit(self, tmp_path: Path) -> None:
        """Test rétention au-dessus de la limite."""
        # Créer 5 versions avec max_versions=3
        for i in range(5):
            v = tmp_path / f"v2024010{i}_120000"
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        deleted = apply_retention_policy(tmp_path, max_versions=3)

        assert len(deleted) == 2  # 5 - 3 = 2
        assert len(list_model_versions(tmp_path)) == 3

    def test_retention_dry_run(self, tmp_path: Path) -> None:
        """Test dry_run ne supprime pas."""
        for i in range(5):
            v = tmp_path / f"v2024010{i}_120000"
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        deleted = apply_retention_policy(tmp_path, max_versions=3, dry_run=True)

        assert len(deleted) == 2
        assert len(list_model_versions(tmp_path)) == 5  # Rien supprimé

    def test_retention_keeps_newest(self, tmp_path: Path) -> None:
        """Test garde les plus récentes."""
        versions = ["v20240101_120000", "v20240102_120000", "v20240103_120000"]
        for v_name in versions:
            v = tmp_path / v_name
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        apply_retention_policy(tmp_path, max_versions=2)

        remaining = list_model_versions(tmp_path)
        assert len(remaining) == 2
        # Les plus récentes gardées
        assert remaining[0].name == "v20240103_120000"
        assert remaining[1].name == "v20240102_120000"

    def test_get_retention_status(self, tmp_path: Path) -> None:
        """Test statut rétention."""
        for i in range(3):
            v = tmp_path / f"v2024010{i}_120000"
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        status = get_retention_status(tmp_path, max_versions=5)

        assert status["current_count"] == 3
        assert status["max_versions"] == 5
        assert status["versions_to_delete"] == 0
        assert status["retention_applied"] is True

    def test_retention_status_over_limit(self, tmp_path: Path) -> None:
        """Test statut quand au-dessus limite."""
        for i in range(5):
            v = tmp_path / f"v2024010{i}_120000"
            v.mkdir()
            (v / "metadata.json").write_text("{}")

        status = get_retention_status(tmp_path, max_versions=3)

        assert status["current_count"] == 5
        assert status["versions_to_delete"] == 2
        assert status["retention_applied"] is False


# ==============================================================================
# P3 TESTS - Chiffrement AES-256 (ISO 27001 - Confidentiality)
# ==============================================================================
