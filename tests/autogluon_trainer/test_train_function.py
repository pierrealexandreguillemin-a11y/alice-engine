"""Tests Train AutoGluon Function - ISO 42001.

Document ID: ALICE-TEST-AUTOGLUON-TRAINER-FUNCTION
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

import inspect

from scripts.autogluon.config import AutoGluonConfig


class TestTrainAutogluonFunction:
    """Tests pour la fonction train_autogluon."""

    def test_function_exists(self) -> None:
        """Test que la fonction existe."""
        from scripts.autogluon.trainer import train_autogluon

        assert callable(train_autogluon)

    def test_function_signature(self) -> None:
        """Test signature de la fonction."""
        from scripts.autogluon.trainer import train_autogluon

        sig = inspect.signature(train_autogluon)
        params = list(sig.parameters.keys())

        assert "train_data" in params
        assert "label" in params
        assert "valid_data" in params
        assert "config" in params
        assert "output_dir" in params

    def test_config_default_values(self) -> None:
        """Test valeurs par defaut de la config."""
        config = AutoGluonConfig()

        assert config.presets == "extreme"
        assert config.time_limit == 21600  # 6 hours default
        assert config.eval_metric == "roc_auc"
        assert config.num_bag_folds == 5

    def test_custom_config_creation(self) -> None:
        """Test creation d'une config personnalisee."""
        config = AutoGluonConfig(
            presets="medium_quality",
            time_limit=120,
            eval_metric="accuracy",
        )

        assert config.presets == "medium_quality"
        assert config.time_limit == 120
        assert config.eval_metric == "accuracy"

    def test_config_to_fit_kwargs(self) -> None:
        """Test conversion config en kwargs."""
        config = AutoGluonConfig(
            presets="high_quality",
            time_limit=600,
            num_bag_folds=3,
        )

        kwargs = config.to_fit_kwargs()

        assert kwargs["presets"] == "high_quality"
        assert kwargs["time_limit"] == 600
        assert kwargs["num_bag_folds"] == 3
        assert kwargs["num_stack_levels"] == 2
