"""Tests Integration - ISO 29119.

Document ID: ALICE-TEST-MODEL-INTEGRATION
Version: 1.0.0
Tests: 1 classes

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.model_registry import (
    compute_file_checksum,
    save_production_models,
)


class TestSaveProductionModelsIntegration:
    """Tests d'intégration pour save_production_models."""

    @pytest.fixture
    def mock_catboost_model(self) -> MagicMock:
        """Crée un mock CatBoost model."""
        model = MagicMock()
        model.get_feature_importance.return_value = np.array([0.3, 0.5, 0.2])

        def save_model_side_effect(path: str) -> None:
            Path(path).write_bytes(b"CatBoost model binary data")

        model.save_model.side_effect = save_model_side_effect
        return model

    @pytest.fixture
    def sample_dataframes(
        self, tmp_path: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
        """Crée des DataFrames de test."""
        # Features
        train = pd.DataFrame(
            {
                "blanc_elo": [1500, 1600, 1700, 1400, 1550] * 20,
                "noir_elo": [1450, 1550, 1650, 1500, 1500] * 20,
                "diff_elo": [50, 50, 50, -100, 50] * 20,
                "resultat_blanc": [1.0, 0.0, 1.0, 0.0, 1.0] * 20,
            }
        )
        valid = train.head(20).copy()
        test = train.head(10).copy()

        # Sauvegarder en parquet
        train_path = tmp_path / "train.parquet"
        valid_path = tmp_path / "valid.parquet"
        test_path = tmp_path / "test.parquet"

        train.to_parquet(train_path)
        valid.to_parquet(valid_path)
        test.to_parquet(test_path)

        return train, valid, test, train_path, valid_path, test_path

    def test_save_production_models_creates_artifacts(
        self,
        tmp_path: Path,
        mock_catboost_model: MagicMock,
        sample_dataframes: tuple,
    ) -> None:
        """Test création complète des artefacts production."""
        train, valid, test, train_path, valid_path, test_path = sample_dataframes
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        models = {"CatBoost": mock_catboost_model}
        metrics = {"CatBoost": {"auc_roc": 0.85, "accuracy": 0.80}}
        feature_names = ["blanc_elo", "noir_elo", "diff_elo"]
        hyperparameters = {"catboost": {"iterations": 1000}}
        # Utiliser un dict simple (MagicMock ne peut pas être sérialisé par joblib)
        label_encoders = {"club": {"classes": ["A", "B", "C"]}}

        version_dir = save_production_models(
            models=models,
            metrics=metrics,
            models_dir=models_dir,
            train_df=train,
            valid_df=valid,
            test_df=test,
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path,
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            label_encoders=label_encoders,
        )

        # Vérifier structure
        assert version_dir.exists()
        assert version_dir.name.startswith("v")
        assert (version_dir / "metadata.json").exists()
        assert (version_dir / "catboost.cbm").exists()
        assert (version_dir / "label_encoders.joblib").exists()

        # Vérifier symlink "current"
        current_link = models_dir / "current"
        assert current_link.exists()

    def test_save_production_models_metadata_content(
        self,
        tmp_path: Path,
        mock_catboost_model: MagicMock,
        sample_dataframes: tuple,
    ) -> None:
        """Test contenu du metadata.json."""
        import json

        train, valid, test, train_path, valid_path, test_path = sample_dataframes
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        models = {"CatBoost": mock_catboost_model}
        metrics = {"CatBoost": {"auc_roc": 0.85}}

        version_dir = save_production_models(
            models=models,
            metrics=metrics,
            models_dir=models_dir,
            train_df=train,
            valid_df=valid,
            test_df=test,
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path,
            feature_names=["blanc_elo", "noir_elo", "diff_elo"],
            hyperparameters={},
            label_encoders={},
        )

        # Lire metadata
        metadata_path = version_dir / "metadata.json"
        with metadata_path.open() as f:
            metadata = json.load(f)

        # Vérifier structure ISO 42001
        assert "version" in metadata
        assert "environment" in metadata
        assert "data_lineage" in metadata
        assert "artifacts" in metadata
        assert "metrics" in metadata
        assert "best_model" in metadata
        assert "conformance" in metadata

        # Vérifier data lineage
        assert metadata["data_lineage"]["train"]["samples"] == 100
        assert metadata["data_lineage"]["valid"]["samples"] == 20
        assert metadata["data_lineage"]["test"]["samples"] == 10

        # Vérifier conformance ISO
        assert "iso_42001" in metadata["conformance"]
        assert "iso_27001" in metadata["conformance"]

    def test_save_production_models_checksums_valid(
        self,
        tmp_path: Path,
        mock_catboost_model: MagicMock,
        sample_dataframes: tuple,
    ) -> None:
        """Test que les checksums sont valides."""
        import json

        train, valid, test, train_path, valid_path, test_path = sample_dataframes
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        models = {"CatBoost": mock_catboost_model}
        metrics = {"CatBoost": {"auc_roc": 0.85}}

        version_dir = save_production_models(
            models=models,
            metrics=metrics,
            models_dir=models_dir,
            train_df=train,
            valid_df=valid,
            test_df=test,
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path,
            feature_names=["blanc_elo"],
            hyperparameters={},
            label_encoders={},
        )

        # Lire metadata
        with (version_dir / "metadata.json").open() as f:
            metadata = json.load(f)

        # Vérifier checksum du modèle
        for artifact in metadata["artifacts"]:
            if artifact["name"] == "CatBoost":
                model_path = version_dir / Path(artifact["path"]).name
                computed_checksum = compute_file_checksum(model_path)
                assert computed_checksum == artifact["checksum"]
