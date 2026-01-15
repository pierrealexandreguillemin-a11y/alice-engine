"""Tests DataLoader Init - ISO 29119.

Document ID: ALICE-TEST-DATA-LOADER-INIT
Version: 1.0.0

Tests pour l'initialisation de DataLoader.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

from services.data_loader import DataLoader


class TestDataLoaderInit:
    """Tests initialisation DataLoader."""

    def test_init_without_config(self) -> None:
        """Test initialisation sans configuration."""
        loader = DataLoader()

        assert loader.mongodb_uri is None
        assert loader.dataset_path is None
        assert loader.db is None

    def test_init_with_mongodb_uri(self) -> None:
        """Test initialisation avec URI MongoDB."""
        loader = DataLoader(mongodb_uri="mongodb://localhost:27017")

        assert loader.mongodb_uri == "mongodb://localhost:27017"
        assert loader.db is None

    def test_init_with_dataset_path(self, tmp_path: Path) -> None:
        """Test initialisation avec chemin dataset."""
        loader = DataLoader(dataset_path=tmp_path)

        assert loader.dataset_path == tmp_path
        assert loader.mongodb_uri is None

    def test_init_with_both_configs(self, tmp_path: Path) -> None:
        """Test initialisation avec les deux configs."""
        loader = DataLoader(
            mongodb_uri="mongodb://localhost:27017",
            dataset_path=tmp_path,
        )

        assert loader.mongodb_uri == "mongodb://localhost:27017"
        assert loader.dataset_path == tmp_path
