"""Tests Execution (Save, Run, Main) - ISO 29119.

Document ID: ALICE-TEST-TRAIN-EXEC
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from tests.train_models.conftest import MockMetricsTPP, MockModelResultTPP


class TestSaveModels:
    """Tests pour _save_models."""

    @patch("scripts.train_models_parallel.save_production_models")
    def test_save_models(self, mock_save: MagicMock, tmp_path: Path) -> None:
        """Test sauvegarde modeles."""
        from scripts.train_models_parallel import _save_models

        mock_save.return_value = tmp_path / "v1"

        mock_model = MagicMock()
        results = {
            "CatBoost": MockModelResultTPP(model=mock_model, metrics=MockMetricsTPP(test_auc=0.85)),
        }

        train_df = pd.DataFrame({"a": [1], "target": [0]})
        valid_df = pd.DataFrame({"a": [2], "target": [1]})
        test_df = pd.DataFrame({"a": [3], "target": [0]})
        X_train = pd.DataFrame({"a": [1]})

        version_dir = _save_models(
            results=results,
            config={"param": 1},
            models_dir=tmp_path,
            train=train_df,
            valid=valid_df,
            test=test_df,
            data_dir=tmp_path,
            X_train=X_train,
            encoders={"encoder": MagicMock()},
        )

        mock_save.assert_called_once()
        assert version_dir == tmp_path / "v1"

    @patch("scripts.train_models_parallel.save_production_models")
    def test_save_models_filters_none(self, mock_save: MagicMock, tmp_path: Path) -> None:
        """Test sauvegarde filtre modeles None."""
        from scripts.train_models_parallel import _save_models

        mock_save.return_value = tmp_path / "v1"

        results = {
            "OK": MockModelResultTPP(model=MagicMock()),
            "Failed": MockModelResultTPP(model=None),
        }

        _save_models(
            results=results,
            config={},
            models_dir=tmp_path,
            train=pd.DataFrame(),
            valid=pd.DataFrame(),
            test=pd.DataFrame(),
            data_dir=tmp_path,
            X_train=pd.DataFrame(),
            encoders={},
        )

        call_kwargs = mock_save.call_args[1]
        assert len(call_kwargs["models"]) == 1
        assert "OK" in call_kwargs["models"]


class TestRunTraining:
    """Tests pour run_training."""

    @patch("scripts.train_models_parallel._load_datasets")
    @patch("scripts.train_models_parallel.load_hyperparameters")
    @patch("scripts.train_models_parallel.prepare_features")
    @patch("scripts.train_models_parallel.train_all_models_parallel")
    @patch("scripts.train_models_parallel._evaluate_on_test")
    @patch("scripts.train_models_parallel._save_models")
    @patch("scripts.train_models_parallel._build_summary")
    def test_run_training_no_mlflow(
        self,
        mock_summary: MagicMock,
        mock_save: MagicMock,
        mock_eval: MagicMock,
        mock_train: MagicMock,
        mock_prepare: MagicMock,
        mock_load_hp: MagicMock,
        mock_load_data: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test pipeline sans MLflow."""
        from scripts.train_models_parallel import run_training

        train_df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        valid_df = pd.DataFrame({"a": [3], "target": [0]})
        test_df = pd.DataFrame({"a": [4], "target": [1]})

        mock_load_data.return_value = (train_df, valid_df, test_df)
        mock_load_hp.return_value = {}

        X_train = pd.DataFrame({"a": [1, 2]})
        y_train = pd.Series([0, 1])
        mock_prepare.return_value = (X_train, y_train, {})

        mock_model = MagicMock()
        mock_train.return_value = {
            "CatBoost": MockModelResultTPP(model=mock_model),
        }

        mock_save.return_value = tmp_path / "v1"
        mock_summary.return_value = {"best_model": "CatBoost"}

        result = run_training(
            data_dir=tmp_path,
            config_path=tmp_path / "config.yaml",
            models_dir=tmp_path / "models",
            use_mlflow=False,
        )

        assert result["best_model"] == "CatBoost"
        mock_train.assert_called_once()


class TestMain:
    """Tests pour main."""

    @patch("scripts.train_models_parallel.run_training")
    def test_main_default_args(self, mock_run: MagicMock) -> None:
        """Test main avec arguments par defaut."""
        from scripts.train_models_parallel import main

        with patch("sys.argv", ["script"]):
            main()

        mock_run.assert_called_once()

    @patch("scripts.train_models_parallel.run_training")
    def test_main_no_mlflow(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test main avec --no-mlflow."""
        from scripts.train_models_parallel import main

        with patch(
            "sys.argv",
            ["script", "--data-dir", str(tmp_path), "--no-mlflow"],
        ):
            main()

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["use_mlflow"] is False

    @patch("scripts.train_models_parallel.run_training")
    def test_main_custom_paths(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test main avec chemins personnalises."""
        from scripts.train_models_parallel import main

        config_path = tmp_path / "custom.yaml"
        models_path = tmp_path / "custom_models"

        with patch(
            "sys.argv",
            [
                "script",
                "--data-dir",
                str(tmp_path),
                "--config",
                str(config_path),
                "--models-dir",
                str(models_path),
            ],
        ):
            main()

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["data_dir"] == tmp_path
        assert call_kwargs["config_path"] == config_path
        assert call_kwargs["models_dir"] == models_path
