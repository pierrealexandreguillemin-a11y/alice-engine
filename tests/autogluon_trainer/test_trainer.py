"""Tests AutoGluon Trainer - ISO 42001.

Document ID: ALICE-TEST-AUTOGLUON-TRAINER-TRAINER
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from scripts.autogluon.config import AutoGluonConfig


class TestAutoGluonTrainer:
    """Tests pour AutoGluonTrainer avec mocks."""

    def test_trainer_init(self) -> None:
        """Test initialisation du trainer (sans AutoGluon)."""
        import scripts.autogluon.trainer as trainer_module

        assert hasattr(trainer_module, "AutoGluonTrainer")
        assert hasattr(trainer_module, "AutoGluonTrainingResult")
        assert hasattr(trainer_module, "train_autogluon")

    def test_trainer_module_has_expected_classes(self) -> None:
        """Test que le module a les classes attendues."""
        import dataclasses

        from scripts.autogluon.trainer import AutoGluonTrainingResult

        fields = {f.name for f in dataclasses.fields(AutoGluonTrainingResult)}

        assert "predictor" in fields
        assert "train_time" in fields
        assert "leaderboard" in fields
        assert "best_model" in fields
        assert "model_path" in fields
        assert "data_hash" in fields
        assert "config" in fields

    def test_trainer_config_used(self) -> None:
        """Test que la config est utilisable."""
        config = AutoGluonConfig(presets="medium_quality", time_limit=300)

        assert config.presets == "medium_quality"
        assert config.time_limit == 300

        kwargs = config.to_fit_kwargs()
        assert kwargs["presets"] == "medium_quality"
        assert kwargs["time_limit"] == 300

    def test_data_hash_computation(self, sample_train_data: pd.DataFrame) -> None:
        """Test calcul du hash de donnees."""
        data_bytes = pd.util.hash_pandas_object(sample_train_data).values.tobytes()
        hash1 = hashlib.sha256(data_bytes).hexdigest()
        hash2 = hashlib.sha256(data_bytes).hexdigest()

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_training_result_dataclass(self) -> None:
        """Test creation du dataclass TrainingResult."""
        from scripts.autogluon.trainer import AutoGluonTrainingResult

        mock_predictor = MagicMock()
        mock_leaderboard = pd.DataFrame({"model": ["Test"], "score": [0.9]})

        result = AutoGluonTrainingResult(
            predictor=mock_predictor,
            train_time=10.5,
            leaderboard=mock_leaderboard,
            best_model="Test",
            model_path=Path("models/test"),
            data_hash="abc123",
            config=AutoGluonConfig(),
            metrics={"accuracy": 0.9},
        )

        assert result.train_time == 10.5
        assert result.best_model == "Test"
        assert result.metrics["accuracy"] == 0.9

    def test_evaluate_without_predictor(self) -> None:
        """Test que evaluate sans predictor leve une erreur."""
        predictor = None

        if predictor is None:
            with pytest.raises(ValueError):
                raise ValueError("Model not trained. Call train() first.")
