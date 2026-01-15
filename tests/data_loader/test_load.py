"""Tests DataLoader Load Training Data - ISO 29119.

Document ID: ALICE-TEST-DATA-LOADER-LOAD
Version: 1.0.0

Tests pour load_training_data et edge cases.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from services.data_loader import DataLoader


class TestLoadTrainingData:
    """Tests pour load_training_data()."""

    def test_load_training_data_no_path(self, loader: DataLoader) -> None:
        """Test sans chemin dataset."""
        result = loader.load_training_data()

        assert result is None

    def test_load_training_data_missing_file(self, tmp_path: Path) -> None:
        """Test fichier parquet manquant."""
        loader = DataLoader(dataset_path=tmp_path)
        result = loader.load_training_data()

        assert result is None

    def test_load_training_data_success(self, sample_parquet_data: Path) -> None:
        """Test chargement réussi."""
        loader = DataLoader(dataset_path=sample_parquet_data)
        result = loader.load_training_data()

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_load_training_data_filter_seasons(self, sample_parquet_data: Path) -> None:
        """Test filtrage par saisons."""
        loader = DataLoader(dataset_path=sample_parquet_data)
        result = loader.load_training_data(seasons=[2024])

        assert result is not None
        assert len(result) == 2
        assert all(result["saison"] == 2024)

    def test_load_training_data_multiple_seasons(self, sample_parquet_data: Path) -> None:
        """Test filtrage plusieurs saisons."""
        loader = DataLoader(dataset_path=sample_parquet_data)
        result = loader.load_training_data(seasons=[2023, 2025])

        assert result is not None
        assert len(result) == 2
        assert set(result["saison"].unique()) == {2023, 2025}

    def test_load_training_data_empty_seasons_filter(self, sample_parquet_data: Path) -> None:
        """Test filtre saison inexistante."""
        loader = DataLoader(dataset_path=sample_parquet_data)
        result = loader.load_training_data(seasons=[2020])

        assert result is not None
        assert len(result) == 0


class TestDataLoaderEdgeCases:
    """Tests edge cases DataLoader."""

    def test_empty_parquet_file(self, tmp_path: Path) -> None:
        """Test fichier parquet vide."""
        df = pd.DataFrame(columns=["saison", "blanc_nom", "noir_nom"])
        df.to_parquet(tmp_path / "echiquiers.parquet")

        loader = DataLoader(dataset_path=tmp_path)
        result = loader.load_training_data()

        assert result is not None
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_opponent_history_empty_club_id(self) -> None:
        """Test avec club_id vide."""
        loader = DataLoader(mongodb_uri="mongodb://localhost")

        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=[])

        mock_collection = MagicMock()
        mock_collection.find.return_value = mock_cursor

        loader.db = MagicMock()
        loader.db.compositions = mock_collection

        result = await loader.get_opponent_history("")

        assert result == []
        call_args = mock_collection.find.call_args
        assert call_args[0][0]["clubId"] == ""

    def test_load_training_data_with_corrupted_parquet(self, tmp_path: Path) -> None:
        """Test avec fichier parquet corrompu."""
        corrupted_file = tmp_path / "echiquiers.parquet"
        corrupted_file.write_text("This is not a parquet file")

        loader = DataLoader(dataset_path=tmp_path)
        result = loader.load_training_data()

        assert result is None
