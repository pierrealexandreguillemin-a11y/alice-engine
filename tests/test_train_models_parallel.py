"""Module: test_train_models_parallel.py - Tests Training Pipeline.

Tests unitaires du pipeline d'entrainement parallele.

ISO Compliance:
- ISO/IEC 29119 - Software Testing (unit tests, coverage)
- ISO/IEC 42001:2023 - AI Management (training validation)
- ISO/IEC 23894:2023 - AI Risk Management (training risks)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from pathlib import Path

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
