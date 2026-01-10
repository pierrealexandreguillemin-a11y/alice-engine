"""Module: test_train_models_parallel.py - Tests Training Pipeline.

Tests unitaires du pipeline d'entrainement parallele.

ISO Compliance:
- ISO/IEC 29119 - Software Testing (unit tests, coverage)
- ISO/IEC 42001:2023 - AI Management (training validation)
- ISO/IEC 23894:2023 - AI Risk Management (training risks)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.ml_types import ModelMetrics
from scripts.training import (
    compute_all_metrics,
    get_default_hyperparameters,
    load_hyperparameters,
    prepare_features,
)


class TestLoadHyperparameters:
    """Tests pour load_hyperparameters."""

    def test_load_existing_config(self, tmp_path: Path) -> None:
        """Test chargement config existante."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "global": {"random_seed": 42},
            "catboost": {"iterations": 100},
        }
        with config_path.open("w") as f:
            yaml.dump(config_data, f)

        result = load_hyperparameters(config_path)

        assert result["global"] == {"random_seed": 42}
        assert result["catboost"] == {"iterations": 100}

    def test_load_nonexistent_config_returns_defaults(self, tmp_path: Path) -> None:
        """Test config inexistante retourne defaults."""
        config_path = tmp_path / "nonexistent.yaml"

        result = load_hyperparameters(config_path)

        assert "global" in result
        assert "catboost" in result
        assert "xgboost" in result
        assert "lightgbm" in result


class TestGetDefaultHyperparameters:
    """Tests pour get_default_hyperparameters."""

    def test_returns_all_model_configs(self) -> None:
        """Test retourne config pour tous les modeles."""
        defaults = get_default_hyperparameters()

        assert "global" in defaults
        assert "catboost" in defaults
        assert "xgboost" in defaults
        assert "lightgbm" in defaults

    def test_global_has_random_seed(self) -> None:
        """Test global contient random_seed."""
        defaults = get_default_hyperparameters()
        global_config = defaults["global"]

        assert isinstance(global_config, dict)
        assert "random_seed" in global_config
        assert global_config["random_seed"] == 42


class TestPrepareFeatures:
    """Tests pour prepare_features."""

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """DataFrame de test."""
        return pd.DataFrame(
            {
                "blanc_elo": [1500, 1600, 1700, 1800],
                "noir_elo": [1550, 1650, 1750, 1850],
                "diff_elo": [-50, -50, -50, -50],
                "echiquier": [1, 2, 3, 4],
                "niveau": [1, 1, 2, 2],
                "ronde": [1, 1, 1, 1],
                "type_competition": ["N1", "N1", "N2", "N2"],
                "division": ["A", "A", "B", "B"],
                "ligue_code": ["IDF", "IDF", "ARA", "ARA"],
                "blanc_titre": ["", "FM", "", "IM"],
                "noir_titre": ["", "", "FM", ""],
                "jour_semaine": ["samedi", "samedi", "dimanche", "dimanche"],
                "resultat_blanc": [1.0, 0.5, 0.0, 1.0],
            }
        )

    def test_prepare_features_fit_encoders(self, sample_dataframe: pd.DataFrame) -> None:
        """Test preparation avec fit_encoders=True."""
        X, y, encoders = prepare_features(sample_dataframe, fit_encoders=True)

        assert len(X) == 4
        assert len(y) == 4
        assert len(encoders) > 0
        assert "type_competition" in encoders

    def test_prepare_features_reuse_encoders(self, sample_dataframe: pd.DataFrame) -> None:
        """Test preparation avec encodeurs existants."""
        _, _, encoders = prepare_features(sample_dataframe, fit_encoders=True)

        # Nouveau DataFrame avec meme structure
        new_df = sample_dataframe.copy()
        X, y, _ = prepare_features(new_df, label_encoders=encoders)

        assert len(X) == 4
        assert len(y) == 4

    def test_prepare_features_target_binary(self, sample_dataframe: pd.DataFrame) -> None:
        """Test target est binaire (victoire=1, autre=0)."""
        _, y, _ = prepare_features(sample_dataframe, fit_encoders=True)

        assert set(y.unique()).issubset({0, 1})
        assert y.sum() == 2  # 2 victoires sur 4

    def test_prepare_features_unknown_category(self, sample_dataframe: pd.DataFrame) -> None:
        """Test gestion categorie inconnue."""
        _, _, encoders = prepare_features(sample_dataframe, fit_encoders=True)

        # DataFrame avec categorie inconnue
        new_df = sample_dataframe.copy()
        new_df.loc[0, "type_competition"] = "UNKNOWN_CATEGORY"

        X, _, _ = prepare_features(new_df, label_encoders=encoders)

        # Doit fonctionner sans erreur (remplace par UNKNOWN)
        assert len(X) == 4


class TestComputeAllMetrics:
    """Tests pour compute_all_metrics."""

    def test_perfect_predictions(self) -> None:
        """Test metriques predictions parfaites."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.auc_roc == 1.0

    def test_all_wrong_predictions(self) -> None:
        """Test metriques predictions inversees."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.2, 0.1])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert metrics.accuracy == 0.0
        assert metrics.recall == 0.0

    def test_metrics_return_type(self) -> None:
        """Test type retour ModelMetrics."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_proba = np.array([0.3, 0.7, 0.6, 0.4])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert isinstance(metrics, ModelMetrics)
        assert isinstance(metrics.auc_roc, float)
        assert isinstance(metrics.accuracy, float)

    def test_confusion_matrix_values(self) -> None:
        """Test valeurs matrice confusion."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.3, 0.6, 0.4, 0.7])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        assert metrics.true_negatives == 1
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.true_positives == 1

    def test_metrics_to_dict(self) -> None:
        """Test conversion metriques en dict."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.9])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)
        metrics_dict = metrics.to_dict()

        assert "auc_roc" in metrics_dict
        assert "accuracy" in metrics_dict
        assert "f1_score" in metrics_dict
        assert isinstance(metrics_dict["auc_roc"], float)


# =============================================================================
# Tests pour scripts/train_models_parallel.py functions
# =============================================================================


@dataclass
class MockMetricsTPP:
    """Mock pour ModelMetrics (train_models_parallel tests)."""

    auc_roc: float = 0.85
    accuracy: float = 0.80
    precision: float = 0.78
    recall: float = 0.82
    f1_score: float = 0.80
    log_loss: float = 0.45
    train_time_s: float = 10.0
    test_auc: float | None = None
    test_accuracy: float | None = None
    test_f1: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "auc_roc": self.auc_roc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "log_loss": self.log_loss,
            "train_time_s": self.train_time_s,
            "test_auc": self.test_auc,
            "test_accuracy": self.test_accuracy,
            "test_f1": self.test_f1,
        }


@dataclass
class MockModelResultTPP:
    """Mock pour SingleModelResult."""

    model: Any = None
    metrics: MockMetricsTPP = field(default_factory=MockMetricsTPP)


class TestLoadDatasets:
    """Tests pour _load_datasets."""

    def test_load_datasets(self, tmp_path: Path) -> None:
        """Test chargement des datasets."""
        from scripts.train_models_parallel import _load_datasets

        # Créer des parquet de test
        train_df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        valid_df = pd.DataFrame({"a": [4, 5], "target": [1, 0]})
        test_df = pd.DataFrame({"a": [6, 7], "target": [0, 1]})

        train_df.to_parquet(tmp_path / "train.parquet")
        valid_df.to_parquet(tmp_path / "valid.parquet")
        test_df.to_parquet(tmp_path / "test.parquet")

        train, valid, test = _load_datasets(tmp_path)

        assert len(train) == 3
        assert len(valid) == 2
        assert len(test) == 2


class TestEvaluateOnTest:
    """Tests pour _evaluate_on_test."""

    def test_evaluate_models(self) -> None:
        """Test évaluation sur test set."""
        from scripts.train_models_parallel import _evaluate_on_test

        # Mock model avec predict_proba
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])

        results = {
            "CatBoost": MockModelResultTPP(model=mock_model),
        }

        X_test = pd.DataFrame({"a": [1, 2]})
        y_test = pd.Series([1, 0])

        with patch("scripts.train_models_parallel.compute_all_metrics") as mock_metrics:
            mock_metrics.return_value = MagicMock(auc_roc=0.90, accuracy=0.85, f1_score=0.82)
            _evaluate_on_test(results, X_test, y_test)

        assert results["CatBoost"].metrics.test_auc == 0.90
        assert results["CatBoost"].metrics.test_accuracy == 0.85

    def test_skip_none_model(self) -> None:
        """Test skip modèle None."""
        from scripts.train_models_parallel import _evaluate_on_test

        results = {"Failed": MockModelResultTPP(model=None)}
        X_test = pd.DataFrame({"a": [1]})
        y_test = pd.Series([0])

        # Ne devrait pas lever d'exception
        _evaluate_on_test(results, X_test, y_test)


class TestBuildSummary:
    """Tests pour _build_summary."""

    def test_build_summary(self, tmp_path: Path) -> None:
        """Test construction résumé."""
        from scripts.train_models_parallel import _build_summary

        mock_model = MagicMock()
        results = {
            "CatBoost": MockModelResultTPP(
                model=mock_model,
                metrics=MockMetricsTPP(auc_roc=0.85, test_auc=0.82),
            ),
            "XGBoost": MockModelResultTPP(
                model=mock_model,
                metrics=MockMetricsTPP(auc_roc=0.83, test_auc=0.80),
            ),
        }

        version_dir = tmp_path / "v1"
        summary = _build_summary(results, version_dir)

        assert summary["best_model"] == "CatBoost"
        assert summary["best_auc"] == 0.82
        assert "version_dir" in summary

    def test_build_summary_with_none_models(self, tmp_path: Path) -> None:
        """Test résumé avec modèles None."""
        from scripts.train_models_parallel import _build_summary

        results = {
            "Failed": MockModelResultTPP(model=None),
            "Success": MockModelResultTPP(
                model=MagicMock(),
                metrics=MockMetricsTPP(test_auc=0.75),
            ),
        }

        summary = _build_summary(results, tmp_path / "v1")

        assert summary["best_model"] == "Success"
        assert summary["best_auc"] == 0.75

    def test_build_summary_fallback_auc(self, tmp_path: Path) -> None:
        """Test résumé utilise auc_roc si test_auc None."""
        from scripts.train_models_parallel import _build_summary

        results = {
            "Model": MockModelResultTPP(
                model=MagicMock(),
                metrics=MockMetricsTPP(auc_roc=0.88, test_auc=None),
            ),
        }

        summary = _build_summary(results, tmp_path / "v1")

        assert summary["best_auc"] == 0.88


class TestSaveModels:
    """Tests pour _save_models."""

    @patch("scripts.train_models_parallel.save_production_models")
    def test_save_models(self, mock_save: MagicMock, tmp_path: Path) -> None:
        """Test sauvegarde modèles."""
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
        """Test sauvegarde filtre modèles None."""
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

        # Vérifier que seul le modèle OK est passé
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

        # Setup mocks
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

    @patch("scripts.train_models_parallel._load_datasets")
    @patch("scripts.train_models_parallel.load_hyperparameters")
    @patch("scripts.train_models_parallel.prepare_features")
    @patch("scripts.train_models_parallel.train_all_models_parallel")
    @patch("scripts.train_models_parallel._evaluate_on_test")
    @patch("scripts.train_models_parallel._save_models")
    @patch("scripts.train_models_parallel._build_summary")
    @patch("scripts.train_models_parallel.setup_mlflow")
    @patch("scripts.train_models_parallel.log_to_mlflow")
    def test_run_training_with_mlflow(
        self,
        mock_log_mlflow: MagicMock,
        mock_setup_mlflow: MagicMock,
        mock_summary: MagicMock,
        mock_save: MagicMock,
        mock_eval: MagicMock,
        mock_train: MagicMock,
        mock_prepare: MagicMock,
        mock_load_hp: MagicMock,
        mock_load_data: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test pipeline avec MLflow."""
        from scripts.train_models_parallel import run_training

        # Setup mocks
        train_df = pd.DataFrame({"a": [1], "target": [0]})
        mock_load_data.return_value = (train_df, train_df, train_df)
        mock_load_hp.return_value = {}
        mock_prepare.return_value = (pd.DataFrame({"a": [1]}), pd.Series([0]), {})
        mock_train.return_value = {"Model": MockModelResultTPP(model=MagicMock())}
        mock_save.return_value = tmp_path / "v1"
        mock_summary.return_value = {}

        run_training(
            data_dir=tmp_path,
            config_path=tmp_path / "config.yaml",
            models_dir=tmp_path,
            use_mlflow=True,
        )

        mock_setup_mlflow.assert_called_once()
        mock_log_mlflow.assert_called_once()


class TestMain:
    """Tests pour main."""

    @patch("scripts.train_models_parallel.run_training")
    def test_main_default_args(self, mock_run: MagicMock) -> None:
        """Test main avec arguments par défaut."""
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
        """Test main avec chemins personnalisés."""
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
