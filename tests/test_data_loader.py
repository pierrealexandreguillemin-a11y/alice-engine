"""Module: test_data_loader.py - Tests DataLoader Service.

Tests unitaires pour le service d'acces aux donnees.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (unit tests, edge cases)
- ISO/IEC 5259:2024 - Data Quality for ML (data loading testing)
- ISO/IEC 25010:2023 - System Quality (testability)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from services.data_loader import DataLoader

# ==============================================================================
# FIXTURES (ISO 29119-3: Test Design)
# ==============================================================================


@pytest.fixture
def loader() -> DataLoader:
    """Fixture loader sans configuration."""
    return DataLoader()


@pytest.fixture
def loader_with_mongodb() -> DataLoader:
    """Fixture loader avec URI MongoDB."""
    return DataLoader(mongodb_uri="mongodb://localhost:27017/alice")


@pytest.fixture
def loader_with_dataset(tmp_path: Path) -> DataLoader:
    """Fixture loader avec chemin dataset."""
    return DataLoader(dataset_path=tmp_path)


@pytest.fixture
def sample_parquet_data(tmp_path: Path) -> Path:
    """Fixture creant un fichier parquet de test."""
    df = pd.DataFrame(
        [
            {"saison": 2023, "blanc_nom": "A", "noir_nom": "B", "resultat_blanc": 1.0},
            {"saison": 2024, "blanc_nom": "C", "noir_nom": "D", "resultat_blanc": 0.5},
            {"saison": 2024, "blanc_nom": "E", "noir_nom": "F", "resultat_blanc": 0.0},
            {"saison": 2025, "blanc_nom": "G", "noir_nom": "H", "resultat_blanc": 1.0},
        ]
    )
    parquet_path = tmp_path / "echiquiers.parquet"
    df.to_parquet(parquet_path)
    return tmp_path


# ==============================================================================
# TESTS: DataLoader Init
# ==============================================================================


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


# ==============================================================================
# TESTS: connect (Async MongoDB)
# ==============================================================================


class TestDataLoaderConnect:
    """Tests connexion MongoDB."""

    @pytest.mark.asyncio
    async def test_connect_no_uri_returns_false(self, loader: DataLoader) -> None:
        """Test connect sans URI retourne False."""
        result = await loader.connect()

        assert result is False
        assert loader.db is None

    @pytest.mark.asyncio
    async def test_connect_success_with_mock(self) -> None:
        """Test connexion avec MongoDB mocke."""
        import sys

        # Create mock motor module
        mock_motor = MagicMock()
        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        mock_client.get_default_database.return_value = MagicMock()
        mock_motor.motor_asyncio.AsyncIOMotorClient.return_value = mock_client

        with patch.dict(
            sys.modules, {"motor": mock_motor, "motor.motor_asyncio": mock_motor.motor_asyncio}
        ):
            loader = DataLoader(mongodb_uri="mongodb://localhost:27017/alice")
            result = await loader.connect()

            assert result is True
            assert loader.db is not None

    @pytest.mark.asyncio
    async def test_connect_failure_returns_false(self) -> None:
        """Test connexion echouee retourne False."""
        import sys

        # Create mock motor module that raises exception
        mock_motor = MagicMock()
        mock_motor.motor_asyncio.AsyncIOMotorClient.side_effect = Exception("Connection refused")

        with patch.dict(
            sys.modules, {"motor": mock_motor, "motor.motor_asyncio": mock_motor.motor_asyncio}
        ):
            loader = DataLoader(mongodb_uri="mongodb://localhost:27017/alice")
            result = await loader.connect()

            assert result is False
            assert loader.db is None


# ==============================================================================
# TESTS: get_opponent_history (Async)
# ==============================================================================


class TestGetOpponentHistory:
    """Tests pour get_opponent_history()."""

    @pytest.mark.asyncio
    async def test_get_opponent_history_no_db(self, loader: DataLoader) -> None:
        """Test sans connexion DB retourne liste vide."""
        result = await loader.get_opponent_history("CLUB123")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_opponent_history_with_mock_db(self) -> None:
        """Test avec DB mockee."""
        loader = DataLoader(mongodb_uri="mongodb://localhost")

        # Mock cursor
        mock_data = [
            {"roundNumber": 1, "players": ["A", "B"], "date": "2025-01-01"},
            {"roundNumber": 2, "players": ["C", "D"], "date": "2025-01-15"},
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=mock_data)

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.find.return_value = mock_cursor

        # Mock db
        loader.db = MagicMock()
        loader.db.compositions = mock_collection

        result = await loader.get_opponent_history("CLUB123", limit=50)

        assert len(result) == 2
        assert result[0]["roundNumber"] == 1
        mock_collection.find.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_opponent_history_with_exception(self) -> None:
        """Test gestion exception."""
        loader = DataLoader(mongodb_uri="mongodb://localhost")
        loader.db = MagicMock()
        loader.db.compositions.find.side_effect = Exception("Query error")

        result = await loader.get_opponent_history("CLUB123")

        assert result == []


# ==============================================================================
# TESTS: get_club_players (Async)
# ==============================================================================


class TestGetClubPlayers:
    """Tests pour get_club_players()."""

    @pytest.mark.asyncio
    async def test_get_club_players_no_db(self, loader: DataLoader) -> None:
        """Test sans connexion DB retourne liste vide."""
        result = await loader.get_club_players("CLUB123")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_club_players_with_mock_db(self) -> None:
        """Test avec DB mockee."""
        loader = DataLoader(mongodb_uri="mongodb://localhost")

        # Mock data
        mock_data = [
            {"ffeId": "A123", "firstName": "Magnus", "lastName": "Carlsen", "elo": 2850},
            {"ffeId": "B456", "firstName": "Fabiano", "lastName": "Caruana", "elo": 2800},
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=mock_data)

        mock_collection = MagicMock()
        mock_collection.find.return_value = mock_cursor

        loader.db = MagicMock()
        loader.db.players = mock_collection

        result = await loader.get_club_players("CLUB123", active_only=True)

        assert len(result) == 2
        assert result[0]["elo"] == 2850

    @pytest.mark.asyncio
    async def test_get_club_players_active_filter(self) -> None:
        """Test filtre joueurs actifs."""
        loader = DataLoader(mongodb_uri="mongodb://localhost")

        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=[])

        mock_collection = MagicMock()
        mock_collection.find.return_value = mock_cursor

        loader.db = MagicMock()
        loader.db.players = mock_collection

        await loader.get_club_players("CLUB123", active_only=True)

        # Verifie que isActive=True est dans la query
        call_args = mock_collection.find.call_args
        query = call_args[0][0]
        assert query["isActive"] is True

    @pytest.mark.asyncio
    async def test_get_club_players_all_players(self) -> None:
        """Test sans filtre actif."""
        loader = DataLoader(mongodb_uri="mongodb://localhost")

        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=[])

        mock_collection = MagicMock()
        mock_collection.find.return_value = mock_cursor

        loader.db = MagicMock()
        loader.db.players = mock_collection

        await loader.get_club_players("CLUB123", active_only=False)

        # isActive ne doit pas etre dans la query
        call_args = mock_collection.find.call_args
        query = call_args[0][0]
        assert "isActive" not in query


# ==============================================================================
# TESTS: load_training_data (Sync)
# ==============================================================================


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
        """Test chargement reussi."""
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


# ==============================================================================
# TESTS: Edge Cases (ISO 29119-4: Boundary Value Analysis)
# ==============================================================================


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
        # Verifie que la query contient le clubId vide
        call_args = mock_collection.find.call_args
        assert call_args[0][0]["clubId"] == ""

    def test_load_training_data_with_corrupted_parquet(self, tmp_path: Path) -> None:
        """Test avec fichier parquet corrompu."""
        # Creer un fichier qui n'est pas un parquet valide
        corrupted_file = tmp_path / "echiquiers.parquet"
        corrupted_file.write_text("This is not a parquet file")

        loader = DataLoader(dataset_path=tmp_path)
        result = loader.load_training_data()

        assert result is None
