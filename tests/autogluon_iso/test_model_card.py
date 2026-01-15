"""Tests Model Card ISO 42001 - ISO 29119.

Document ID: ALICE-TEST-AUTOGLUON-ISO-CARD
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from scripts.autogluon.iso_compliance import (
    ISO42001ModelCard,
    generate_model_card,
)


class TestISO42001ModelCard:
    """Tests pour ISO42001ModelCard."""

    def test_dataclass_creation(self) -> None:
        """Test creation du dataclass."""
        card = ISO42001ModelCard(
            model_id="test_001",
            model_name="TestModel",
            version="1.0.0",
            created_at="2026-01-10",
            training_data_hash="abc123",
            hyperparameters={"lr": 0.01},
            metrics={"accuracy": 0.85},
        )

        assert card.model_id == "test_001"
        assert card.model_name == "TestModel"
        assert card.version == "1.0.0"

    def test_default_values(self) -> None:
        """Test valeurs par defaut."""
        card = ISO42001ModelCard(
            model_id="test",
            model_name="Test",
            version="1.0",
            created_at="2026-01-10",
            training_data_hash="abc",
            hyperparameters={},
            metrics={},
        )

        assert "echecs" in card.intended_use.lower()
        assert "FFE" in card.limitations

    def test_feature_importance_default(self) -> None:
        """Test importance features par defaut."""
        card = ISO42001ModelCard(
            model_id="test",
            model_name="Test",
            version="1.0",
            created_at="2026-01-10",
            training_data_hash="abc",
            hyperparameters={},
            metrics={},
        )

        assert card.feature_importance == {}


class TestGenerateModelCard:
    """Tests pour generate_model_card."""

    def test_generates_valid_card(
        self,
        mock_training_result: MagicMock,
    ) -> None:
        """Test generation d'une carte valide."""
        card = generate_model_card(mock_training_result)

        assert isinstance(card, ISO42001ModelCard)
        assert card.model_name == "AutoGluon_best_quality"
        assert card.best_model == "CatBoost"
        assert card.training_data_hash == mock_training_result.data_hash

    def test_saves_to_file(
        self,
        mock_training_result: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test sauvegarde dans fichier."""
        output_path = tmp_path / "model_card.json"
        generate_model_card(mock_training_result, output_path=output_path)

        assert output_path.exists()

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "model_id" in data
        assert "hyperparameters" in data

    def test_includes_feature_importance(
        self,
        mock_training_result: MagicMock,
    ) -> None:
        """Test inclusion de l'importance des features."""
        card = generate_model_card(mock_training_result)

        assert len(card.feature_importance) > 0
        assert "feature_1" in card.feature_importance

    def test_handles_feature_importance_error(
        self,
        mock_training_result: MagicMock,
    ) -> None:
        """Test gestion erreur importance features."""
        mock_training_result.predictor.feature_importance.side_effect = Exception("Error")

        card = generate_model_card(mock_training_result)

        assert card.feature_importance == {}
