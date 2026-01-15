"""Tests DataLoader Connect - ISO 29119.

Document ID: ALICE-TEST-DATA-LOADER-CONNECT
Version: 1.0.0

Tests pour la connexion MongoDB.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.data_loader import DataLoader


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
        """Test connexion avec MongoDB mocké."""
        import sys

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
        """Test connexion échouée retourne False."""
        import sys

        mock_motor = MagicMock()
        mock_motor.motor_asyncio.AsyncIOMotorClient.side_effect = Exception("Connection refused")

        with patch.dict(
            sys.modules, {"motor": mock_motor, "motor.motor_asyncio": mock_motor.motor_asyncio}
        ):
            loader = DataLoader(mongodb_uri="mongodb://localhost:27017/alice")
            result = await loader.connect()

            assert result is False
            assert loader.db is None
