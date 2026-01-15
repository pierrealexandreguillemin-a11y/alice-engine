"""Tests MLflow Tracking - ISO 42001.

Document ID: ALICE-TEST-MLFLOW
Version: 1.0.0
Tests: 5

Classes:
- TestSetupMlflow: Tests configuration (2 tests)
- TestLogToMlflow: Tests logging (2 tests)
- TestLogGlobalConfig: Tests fonctions internes (1 test)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management (Traçabilité)
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import sys
from unittest.mock import MagicMock, patch


class TestSetupMlflow:
    """Tests pour setup_mlflow."""

    def test_setup_with_mlflow_installed(self):
        """Setup réussit avec MLflow installé."""
        from scripts.training.mlflow_tracking import setup_mlflow

        config = {
            "mlflow": {
                "experiment_name": "test-experiment",
                "tracking_uri": "./test-mlruns",
            }
        }

        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = setup_mlflow(config)

        assert result is True

    def test_setup_default_values(self):
        """Utilise valeurs par défaut si config vide."""
        from scripts.training.mlflow_tracking import setup_mlflow

        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = setup_mlflow({})

        assert result is True


class TestLogToMlflow:
    """Tests pour log_to_mlflow."""

    def test_log_with_results(self, mock_training_result):
        """Log les résultats dans MLflow."""
        from scripts.training.mlflow_tracking import log_to_mlflow

        results = {"CatBoost": mock_training_result}
        config = {"catboost": {"iterations": 100}}

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            log_to_mlflow(results, config)

        # Should not raise

    def test_log_handles_exception(self, mock_training_result):
        """Gère les exceptions gracieusement."""
        from scripts.training.mlflow_tracking import log_to_mlflow

        results = {"CatBoost": mock_training_result}

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.side_effect = Exception("MLflow error")

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            # Should not raise, just log warning
            log_to_mlflow(results, {})


class TestLogGlobalConfig:
    """Tests pour _log_global_config."""

    def test_logs_params(self):
        """Log les paramètres globaux."""
        from scripts.training.mlflow_tracking import _log_global_config

        config = {"global": {"random_seed": 123}}

        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            _log_global_config(config)

        mock_mlflow.log_param.assert_called()
