# tests/test_ensemble_stacking.py
"""Tests pour ensemble_stacking.py - ISO 29119.

Tests unitaires du stacking ensemble avec soft voting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.ensemble_stacking import (
    MODEL_NAMES,
    StackingMetrics,
    compute_soft_voting,
    load_hyperparameters,
    prepare_features,
)


class TestComputeSoftVoting:
    """Tests pour compute_soft_voting."""

    def test_simple_average(self) -> None:
        """Test moyenne simple sans poids."""
        # 3 modeles, 4 samples
        test_matrix = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.2, 0.4, 0.6],
            ]
        )

        result = compute_soft_voting(test_matrix)

        # Moyenne par ligne
        expected = np.array([0.2, 0.5, 0.8, 0.4])
        np.testing.assert_array_almost_equal(result, expected)

    def test_weighted_average(self) -> None:
        """Test moyenne ponderee avec poids."""
        test_matrix = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )
        weights = {"CatBoost": 0.5, "XGBoost": 0.3, "LightGBM": 0.2}

        result = compute_soft_voting(test_matrix, weights)

        # Poids normalises: 0.5, 0.3, 0.2
        # Row 0: 0.1*0.5 + 0.2*0.3 + 0.3*0.2 = 0.05 + 0.06 + 0.06 = 0.17
        # Row 1: 0.4*0.5 + 0.5*0.3 + 0.6*0.2 = 0.20 + 0.15 + 0.12 = 0.47
        expected = np.array([0.17, 0.47])
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_weights_uses_average(self) -> None:
        """Test poids None utilise moyenne simple."""
        test_matrix = np.array(
            [
                [0.3, 0.3, 0.3],
                [0.6, 0.6, 0.6],
            ]
        )

        result = compute_soft_voting(test_matrix, weights=None)

        expected = np.array([0.3, 0.6])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_model(self) -> None:
        """Test avec un seul modele."""
        test_matrix = np.array([[0.5], [0.7], [0.3]])

        result = compute_soft_voting(test_matrix)

        expected = np.array([0.5, 0.7, 0.3])
        np.testing.assert_array_almost_equal(result, expected)


class TestLoadHyperparameters:
    """Tests pour load_hyperparameters."""

    def test_load_existing_config(self, tmp_path: Path) -> None:
        """Test chargement config existante."""
        config_path = tmp_path / "stacking_config.yaml"
        config_data = {
            "stacking": {
                "meta_learner": "logistic_regression",
                "logistic_regression": {"C": 1.0},
            },
            "catboost": {"iterations": 500},
        }
        with config_path.open("w") as f:
            yaml.dump(config_data, f)

        result = load_hyperparameters(config_path)

        assert "stacking" in result
        assert result["stacking"]["meta_learner"] == "logistic_regression"

    def test_nonexistent_config_returns_empty(self, tmp_path: Path) -> None:
        """Test config inexistante retourne dict vide."""
        config_path = tmp_path / "nonexistent.yaml"

        result = load_hyperparameters(config_path)

        assert result == {}


class TestPrepareFeatures:
    """Tests pour prepare_features du stacking."""

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """DataFrame de test."""
        return pd.DataFrame(
            {
                "blanc_elo": [1500, 1600, 1700],
                "noir_elo": [1550, 1650, 1750],
                "diff_elo": [-50, -50, -50],
                "echiquier": [1, 2, 3],
                "niveau": [1, 1, 2],
                "ronde": [1, 1, 1],
                "type_competition": ["N1", "N1", "N2"],
                "division": ["A", "A", "B"],
                "ligue_code": ["IDF", "IDF", "ARA"],
                "blanc_titre": ["", "FM", ""],
                "noir_titre": ["", "", "FM"],
                "jour_semaine": ["samedi", "samedi", "dimanche"],
                "resultat_blanc": [1.0, 0.5, 0.0],
            }
        )

    def test_prepare_features_shape(self, sample_dataframe: pd.DataFrame) -> None:
        """Test forme des features."""
        X, y, encoders = prepare_features(sample_dataframe, fit_encoders=True)

        assert len(X) == 3
        assert len(y) == 3
        assert X.shape[1] > 0

    def test_prepare_features_target(self, sample_dataframe: pd.DataFrame) -> None:
        """Test target binaire."""
        _, y, _ = prepare_features(sample_dataframe, fit_encoders=True)

        # 1 victoire (1.0) sur 3
        assert y.sum() == 1


class TestStackingMetrics:
    """Tests pour StackingMetrics dataclass."""

    def test_stacking_metrics_creation(self) -> None:
        """Test creation StackingMetrics."""
        metrics = StackingMetrics(
            single_models={
                "CatBoost": {"oof_auc": 0.85, "test_auc": 0.84},
                "XGBoost": {"oof_auc": 0.83, "test_auc": 0.82},
                "LightGBM": {"oof_auc": 0.84, "test_auc": 0.83},
            },
            stacking_train_auc=0.87,
            stacking_test_auc=0.86,
            soft_voting_test_auc=0.845,
            gain_vs_best_single=0.02,
            gain_vs_soft_voting=0.015,
            best_single_name="CatBoost",
            best_single_auc=0.84,
        )

        assert metrics.stacking_test_auc == 0.86
        assert metrics.soft_voting_test_auc == 0.845
        assert metrics.best_single_name == "CatBoost"

    def test_gain_calculations(self) -> None:
        """Test calculs de gain."""
        metrics = StackingMetrics(
            single_models={},
            stacking_train_auc=0.90,
            stacking_test_auc=0.88,
            soft_voting_test_auc=0.85,
            gain_vs_best_single=0.03,
            gain_vs_soft_voting=0.03,
            best_single_name="XGBoost",
            best_single_auc=0.85,
        )

        # gain_vs_best = stacking_test - best_single = 0.88 - 0.85 = 0.03
        assert metrics.gain_vs_best_single == 0.03
        # gain_vs_soft = stacking_test - soft_voting = 0.88 - 0.85 = 0.03
        assert metrics.gain_vs_soft_voting == 0.03


class TestModelNames:
    """Tests pour constantes MODEL_NAMES."""

    def test_model_names_tuple(self) -> None:
        """Test MODEL_NAMES est un tuple."""
        assert isinstance(MODEL_NAMES, tuple)

    def test_model_names_content(self) -> None:
        """Test contenu MODEL_NAMES."""
        assert "CatBoost" in MODEL_NAMES
        assert "XGBoost" in MODEL_NAMES
        assert "LightGBM" in MODEL_NAMES
        assert len(MODEL_NAMES) == 3
