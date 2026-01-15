"""Tests Artifacts - ISO 42001/27001.

Document ID: ALICE-TEST-ARTIFACTS
Version: 1.0.0
Tests: 12

Classes:
- TestExtractFeatureImportance: Tests extraction (3 tests)
- TestNormalizeImportance: Tests normalisation (2 tests)
- TestSaveModelArtifact: Tests sauvegarde (3 tests)
- TestValidateModelIntegrity: Tests intégrité (2 tests)
- TestLoadModelWithValidation: Tests chargement (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management (Traçabilité)
- ISO/IEC 27001:2022 - Information Security (Integrity)
- ISO/IEC 5055:2021 - Code Quality (<120 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.model_registry.artifacts import (
    _build_importance_dict,
    _get_raw_importances,
    _normalize_importance,
    extract_feature_importance,
    validate_model_integrity,
)
from scripts.model_registry.dataclasses import ModelArtifact


class TestExtractFeatureImportance:
    """Tests pour extract_feature_importance."""

    def test_catboost_importance(self):
        """Extrait importance CatBoost."""
        mock_model = MagicMock()
        mock_model.get_feature_importance.return_value = [0.5, 0.3, 0.2]

        result = extract_feature_importance(mock_model, "CatBoost", ["a", "b", "c"])

        assert len(result) == 3
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_sklearn_importance(self):
        """Extrait importance via feature_importances_."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = [0.4, 0.4, 0.2]
        del mock_model.get_feature_importance

        result = extract_feature_importance(mock_model, "XGBoost", ["x", "y", "z"])

        assert len(result) == 3

    def test_handles_exception(self):
        """Retourne dict vide en cas d'erreur."""
        mock_model = MagicMock()
        mock_model.get_feature_importance.side_effect = Exception("Error")

        result = extract_feature_importance(mock_model, "CatBoost", ["a"])

        assert result == {}


class TestGetRawImportances:
    """Tests pour _get_raw_importances."""

    def test_catboost_method(self):
        """Utilise get_feature_importance pour CatBoost."""
        mock_model = MagicMock()
        mock_model.get_feature_importance.return_value = [0.5, 0.5]

        result = _get_raw_importances(mock_model, "CatBoost")

        assert result == [0.5, 0.5]

    def test_sklearn_attribute(self):
        """Utilise feature_importances_ pour sklearn."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = [0.3, 0.7]
        del mock_model.get_feature_importance

        result = _get_raw_importances(mock_model, "XGBoost")

        assert result == [0.3, 0.7]


class TestNormalizeImportance:
    """Tests pour _normalize_importance."""

    def test_normalization(self):
        """Normalise pour sommer à 1."""
        importance = {"a": 2.0, "b": 3.0}

        result = _normalize_importance(importance)

        assert abs(result["a"] - 0.4) < 1e-6
        assert abs(result["b"] - 0.6) < 1e-6

    def test_empty_dict(self):
        """Retourne dict vide si vide."""
        result = _normalize_importance({})
        assert result == {}


class TestBuildImportanceDict:
    """Tests pour _build_importance_dict."""

    def test_builds_dict(self):
        """Construit dictionnaire correct."""
        result = _build_importance_dict([0.5, 0.3], ["a", "b"])

        assert result == {"a": 0.5, "b": 0.3}

    def test_handles_mismatch(self):
        """Gère décalage longueurs."""
        result = _build_importance_dict([0.5], ["a", "b", "c"])

        assert len(result) == 1
        assert "a" in result


class TestValidateModelIntegrity:
    """Tests pour validate_model_integrity."""

    def test_valid_checksum(self, tmp_path):
        """Valide si checksum correspond."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"test model content")

        with patch("scripts.model_registry.artifacts.compute_file_checksum") as mock_checksum:
            mock_checksum.return_value = "abc123"

            artifact = ModelArtifact(
                name="TestModel",
                path=model_file,
                format=".cbm",
                checksum="abc123",
                size_bytes=100,
            )

            result = validate_model_integrity(artifact)

            assert result is True

    def test_invalid_checksum(self, tmp_path):
        """Échoue si checksum différent."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"test model content")

        with patch("scripts.model_registry.artifacts.compute_file_checksum") as mock_checksum:
            mock_checksum.return_value = "different"

            artifact = ModelArtifact(
                name="TestModel",
                path=model_file,
                format=".cbm",
                checksum="abc123",
                size_bytes=100,
            )

            result = validate_model_integrity(artifact)

            assert result is False

    def test_missing_file(self):
        """Échoue si fichier manquant."""
        artifact = ModelArtifact(
            name="TestModel",
            path=Path("/nonexistent/model.cbm"),
            format=".cbm",
            checksum="abc123",
            size_bytes=100,
        )

        result = validate_model_integrity(artifact)

        assert result is False


class TestSaveModelArtifact:
    """Tests pour save_model_artifact."""

    def test_saves_catboost(self, tmp_path):
        """Sauvegarde modèle CatBoost."""
        from scripts.model_registry.artifacts import save_model_artifact

        mock_model = MagicMock()
        mock_model.save_model = MagicMock()

        with patch("scripts.model_registry.artifacts.compute_file_checksum") as mock_checksum:
            mock_checksum.return_value = "checksum123"

            # Create fake file after save
            def create_file(path):
                Path(path).write_bytes(b"model data")

            mock_model.save_model.side_effect = create_file

            result = save_model_artifact(mock_model, "CatBoost", tmp_path, ["a", "b"])

            assert result is not None
            assert result.name == "CatBoost"

    def test_handles_save_error(self, tmp_path):
        """Retourne None en cas d'erreur."""
        from scripts.model_registry.artifacts import save_model_artifact

        mock_model = MagicMock()
        mock_model.save_model.side_effect = Exception("Save error")

        result = save_model_artifact(mock_model, "CatBoost", tmp_path, [])

        assert result is None


class TestLoadModelWithValidation:
    """Tests pour load_model_with_validation."""

    def test_load_catboost(self, tmp_path):
        """Charge modèle CatBoost."""
        from scripts.model_registry.artifacts import load_model_with_validation

        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model data")

        artifact = ModelArtifact(
            name="CatBoost",
            path=model_file,
            format=".cbm",
            checksum="abc123",
            size_bytes=100,
        )

        mock_catboost = MagicMock()
        mock_model = MagicMock()
        mock_catboost.CatBoostClassifier.return_value = mock_model

        with (
            patch("scripts.model_registry.artifacts.validate_model_integrity") as mock_valid,
            patch.dict(sys.modules, {"catboost": mock_catboost}),
        ):
            mock_valid.return_value = True

            result = load_model_with_validation(artifact)

            assert result is not None

    def test_returns_none_if_invalid(self, tmp_path):
        """Retourne None si intégrité échoue."""
        from scripts.model_registry.artifacts import load_model_with_validation

        artifact = ModelArtifact(
            name="TestModel",
            path=tmp_path / "model.cbm",
            format=".cbm",
            checksum="abc123",
            size_bytes=100,
        )

        with patch("scripts.model_registry.artifacts.validate_model_integrity") as mock_valid:
            mock_valid.return_value = False

            result = load_model_with_validation(artifact)

            assert result is None
