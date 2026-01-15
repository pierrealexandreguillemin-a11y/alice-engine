"""Tests DataLoader Queries - ISO 29119.

Document ID: ALICE-TEST-DATA-LOADER-QUERIES
Version: 1.0.0

Tests pour get_opponent_history et get_club_players.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.data_loader import DataLoader


class TestGetOpponentHistory:
    """Tests pour get_opponent_history()."""

    @pytest.mark.asyncio
    async def test_get_opponent_history_no_db(self, loader: DataLoader) -> None:
        """Test sans connexion DB retourne liste vide."""
        result = await loader.get_opponent_history("CLUB123")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_opponent_history_with_mock_db(self) -> None:
        """Test avec DB mockée."""
        loader = DataLoader(mongodb_uri="mongodb://localhost")

        mock_data = [
            {"roundNumber": 1, "players": ["A", "B"], "date": "2025-01-01"},
            {"roundNumber": 2, "players": ["C", "D"], "date": "2025-01-15"},
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=mock_data)

        mock_collection = MagicMock()
        mock_collection.find.return_value = mock_cursor

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


class TestGetClubPlayers:
    """Tests pour get_club_players()."""

    @pytest.mark.asyncio
    async def test_get_club_players_no_db(self, loader: DataLoader) -> None:
        """Test sans connexion DB retourne liste vide."""
        result = await loader.get_club_players("CLUB123")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_club_players_with_mock_db(self) -> None:
        """Test avec DB mockée."""
        loader = DataLoader(mongodb_uri="mongodb://localhost")

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

        call_args = mock_collection.find.call_args
        query = call_args[0][0]
        assert "isActive" not in query
