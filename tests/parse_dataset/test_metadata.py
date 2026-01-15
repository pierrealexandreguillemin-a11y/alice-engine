"""Tests Metadata - ISO 29119.

Document ID: ALICE-TEST-PARSE-METADATA
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

from scripts.parse_dataset import (
    extract_metadata_from_path,
)

# ==============================================================================
# TESTS CONSTANTS
# ==============================================================================


class TestExtractMetadataFromPath:
    """Tests pour extract_metadata_from_path."""

    def test_national_competition(self, tmp_path: Path) -> None:
        """Test extraction metadata compétition nationale."""
        data_root = tmp_path / "dataset"
        data_root.mkdir()
        groupe_dir = data_root / "2025" / "Interclubs" / "Nationale_1" / "Groupe_A"
        groupe_dir.mkdir(parents=True)

        meta = extract_metadata_from_path(groupe_dir, data_root)

        assert meta.saison == 2025
        assert "Interclubs" in meta.competition
        assert meta.type_competition == "national"
        assert meta.niveau == 1

    def test_regional_competition(self, tmp_path: Path) -> None:
        """Test extraction metadata compétition régionale."""
        data_root = tmp_path / "dataset"
        data_root.mkdir()
        groupe_dir = data_root / "2025" / "Ligue_de_Ile_de_France" / "R1" / "Groupe_1"
        groupe_dir.mkdir(parents=True)

        meta = extract_metadata_from_path(groupe_dir, data_root)

        assert meta.saison == 2025
        assert meta.type_competition == "regional"

    def test_minimal_path(self, tmp_path: Path) -> None:
        """Test extraction avec chemin minimal."""
        data_root = tmp_path / "dataset"
        data_root.mkdir()
        groupe_dir = data_root / "2025"
        groupe_dir.mkdir(parents=True)

        meta = extract_metadata_from_path(groupe_dir, data_root)

        assert meta.saison == 2025
        assert meta.competition == ""

    def test_invalid_saison(self, tmp_path: Path) -> None:
        """Test extraction avec saison invalide."""
        data_root = tmp_path / "dataset"
        data_root.mkdir()
        groupe_dir = data_root / "invalid" / "Interclubs"
        groupe_dir.mkdir(parents=True)

        meta = extract_metadata_from_path(groupe_dir, data_root)

        assert meta.saison == 0

    def test_coupe_de_france(self, tmp_path: Path) -> None:
        """Test extraction Coupe de France."""
        data_root = tmp_path / "dataset"
        data_root.mkdir()
        groupe_dir = data_root / "2025" / "Coupe_de_France" / "Tour_1" / "Groupe"
        groupe_dir.mkdir(parents=True)

        meta = extract_metadata_from_path(groupe_dir, data_root)

        assert meta.type_competition == "coupe"
