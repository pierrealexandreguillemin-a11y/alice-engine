"""Tests Integrity - ISO 29119.

Document ID: ALICE-TEST-MODEL-INTEGRITY
Version: 1.0.0
Tests: 2 classes

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from scripts.model_registry import (
    ModelArtifact,
    compute_file_checksum,
    extract_feature_importance,
    validate_model_integrity,
)


class TestValidateModelIntegrity:
    """Tests pour validate_model_integrity (ISO 27001)."""

    def test_valid_integrity(self, tmp_path: Path) -> None:
        """Test validation réussie avec checksum correct."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        checksum = compute_file_checksum(model_file)

        artifact = ModelArtifact(
            name="CatBoost",
            path=model_file,
            format=".cbm",
            checksum=checksum,
            size_bytes=model_file.stat().st_size,
        )

        result = validate_model_integrity(artifact)

        assert result is True

    def test_invalid_checksum(self, tmp_path: Path) -> None:
        """Test validation échouée avec checksum incorrect."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")

        artifact = ModelArtifact(
            name="CatBoost",
            path=model_file,
            format=".cbm",
            checksum="invalid_checksum_" + "0" * 48,  # Wrong checksum
            size_bytes=model_file.stat().st_size,
        )

        result = validate_model_integrity(artifact)

        assert result is False

    def test_missing_file(self, tmp_path: Path) -> None:
        """Test validation échouée avec fichier manquant."""
        artifact = ModelArtifact(
            name="CatBoost",
            path=tmp_path / "nonexistent.cbm",
            format=".cbm",
            checksum="a" * 64,
            size_bytes=100,
        )

        result = validate_model_integrity(artifact)

        assert result is False


class TestExtractFeatureImportance:
    """Tests pour extract_feature_importance (ISO 42001)."""

    def test_catboost_feature_importance(self) -> None:
        """Test extraction importance CatBoost."""
        mock_model = MagicMock()
        mock_model.get_feature_importance.return_value = np.array([0.3, 0.5, 0.2])

        feature_names = ["elo", "niveau", "ronde"]
        importance = extract_feature_importance(mock_model, "CatBoost", feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == 3
        # Normalisé et trié par importance décroissante
        assert list(importance.keys())[0] == "niveau"  # 0.5 is highest
        assert abs(sum(importance.values()) - 1.0) < 0.001  # Sum to 1

    def test_xgboost_feature_importance(self) -> None:
        """Test extraction importance XGBoost."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.4, 0.4, 0.2])

        feature_names = ["elo", "niveau", "ronde"]
        importance = extract_feature_importance(mock_model, "XGBoost", feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert abs(sum(importance.values()) - 1.0) < 0.001

    def test_lightgbm_feature_importance(self) -> None:
        """Test extraction importance LightGBM."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.1, 0.6, 0.3])

        feature_names = ["elo", "niveau", "ronde"]
        importance = extract_feature_importance(mock_model, "LightGBM", feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert list(importance.keys())[0] == "niveau"  # 0.6 is highest

    def test_unknown_model_returns_empty(self) -> None:
        """Test modèle inconnu retourne dict vide."""
        mock_model = MagicMock(spec=[])  # No feature importance methods

        feature_names = ["elo", "niveau"]
        importance = extract_feature_importance(mock_model, "UnknownModel", feature_names)

        assert importance == {}

    def test_empty_features_list(self) -> None:
        """Test liste features vide."""
        mock_model = MagicMock()
        mock_model.get_feature_importance.return_value = np.array([])

        importance = extract_feature_importance(mock_model, "CatBoost", [])

        assert importance == {}


# ==============================================================================
# P2 TESTS - Signature HMAC, Schema validation, Retention policy
# ==============================================================================
