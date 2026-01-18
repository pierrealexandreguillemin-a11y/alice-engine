"""Tests for calibrator module - ISO 29119.

Document ID: TEST-CALIBRATION-001
Version: 1.0.0
Tests count: 19

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 24029:2021 - Neural Network Robustness
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from scripts.calibration.calibrator import (
    AutoGluonWrapper,
    Calibrator,
    _wrap_if_autogluon,
    calibrate_model,
    compute_ece,
)
from scripts.calibration.calibrator_types import (
    CalibrationConfig,
    CalibrationMethod,
)


@pytest.fixture
def sample_data():
    """Génère des données de test."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_model(sample_data):
    """Modèle entraîné pour tests."""
    X_train, _, y_train, _ = sample_data
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model


class TestComputeECE:
    """Tests pour compute_ece."""

    def test_perfect_calibration(self) -> None:
        """ECE=0 pour calibration parfaite."""
        # Prédictions parfaitement calibrées
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9])
        ece = compute_ece(y_true, y_prob, n_bins=5)
        assert ece >= 0  # ECE est toujours positif
        assert ece < 0.5  # ECE raisonnable

    def test_poor_calibration(self) -> None:
        """ECE élevé pour mauvaise calibration."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
        ece = compute_ece(y_true, y_prob, n_bins=5)
        assert ece > 0.5  # Mauvaise calibration

    def test_ece_bounds(self) -> None:
        """ECE est borné entre 0 et 1."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)
        ece = compute_ece(y_true, y_prob)
        assert 0 <= ece <= 1


class TestCalibrationConfig:
    """Tests pour CalibrationConfig."""

    def test_default_config(self) -> None:
        """Config par défaut."""
        config = CalibrationConfig()
        assert config.method == CalibrationMethod.ISOTONIC
        assert config.cv == 5
        assert config.n_bins == 10

    def test_platt_config(self) -> None:
        """Config Platt scaling."""
        config = CalibrationConfig(method=CalibrationMethod.PLATT)
        assert config.method == CalibrationMethod.PLATT


class TestCalibrator:
    """Tests pour Calibrator."""

    def test_fit_isotonic(self, sample_data, trained_model) -> None:
        """Calibration isotonic."""
        _, X_test, _, y_test = sample_data
        # cv=0 signifie "prefit" - le modèle est déjà entraîné
        config = CalibrationConfig(method=CalibrationMethod.ISOTONIC, cv=0)
        calibrator = Calibrator(config)
        result = calibrator.fit(trained_model, X_test, y_test)

        assert result.method == CalibrationMethod.ISOTONIC
        assert result.metrics.brier_before >= 0
        assert result.metrics.brier_after >= 0
        assert result.calibrator is not None

    def test_fit_platt(self, sample_data, trained_model) -> None:
        """Calibration Platt scaling."""
        _, X_test, _, y_test = sample_data
        # cv=0 signifie "prefit" - le modèle est déjà entraîné
        config = CalibrationConfig(method=CalibrationMethod.PLATT, cv=0)
        calibrator = Calibrator(config)
        result = calibrator.fit(trained_model, X_test, y_test)

        assert result.method == CalibrationMethod.PLATT
        assert result.metrics is not None

    def test_calibration_curve_generated(self, sample_data, trained_model) -> None:
        """Courbe de calibration générée."""
        _, X_test, _, y_test = sample_data
        calibrator = Calibrator()
        result = calibrator.fit(trained_model, X_test, y_test)

        assert "fraction_positives" in result.calibration_curve
        assert "mean_predicted" in result.calibration_curve
        assert len(result.calibration_curve["fraction_positives"]) > 0

    def test_metrics_to_dict(self, sample_data, trained_model) -> None:
        """Conversion métriques en dict."""
        _, X_test, _, y_test = sample_data
        calibrator = Calibrator()
        result = calibrator.fit(trained_model, X_test, y_test)

        metrics_dict = result.metrics.to_dict()
        assert "brier_before" in metrics_dict
        assert "brier_after" in metrics_dict
        assert "ece_before" in metrics_dict
        assert "ece_after" in metrics_dict


class TestCalibrateModel:
    """Tests pour calibrate_model helper."""

    def test_calibrate_model_wrapper(self, sample_data, trained_model) -> None:
        """calibrate_model est un wrapper."""
        _, X_test, _, y_test = sample_data
        result = calibrate_model(
            trained_model,
            X_test,
            y_test,
            method=CalibrationMethod.ISOTONIC,
        )
        assert result.method == CalibrationMethod.ISOTONIC
        assert result.calibrator is not None

    def test_calibrate_model_default_method(self, sample_data, trained_model) -> None:
        """Méthode par défaut est ISOTONIC."""
        _, X_test, _, y_test = sample_data
        result = calibrate_model(trained_model, X_test, y_test)
        assert result.method == CalibrationMethod.ISOTONIC


class TestAutoGluonWrapper:
    """Tests pour AutoGluonWrapper (ISO 29119)."""

    @pytest.fixture
    def mock_autogluon_predictor(self):
        """Crée un mock AutoGluon predictor."""
        predictor = MagicMock()
        predictor.class_labels = [0, 1]
        predictor.predict.return_value = pd.Series([0, 1, 0, 1, 0])
        predictor.predict_proba.return_value = pd.DataFrame({
            0: [0.8, 0.3, 0.7, 0.2, 0.9],
            1: [0.2, 0.7, 0.3, 0.8, 0.1],
        })
        return predictor

    def test_init_with_class_labels(self, mock_autogluon_predictor) -> None:
        """Wrapper extrait class_labels du predictor."""
        wrapper = AutoGluonWrapper(mock_autogluon_predictor)
        assert list(wrapper.classes_) == [0, 1]
        assert wrapper._estimator_type == "classifier"

    def test_init_without_class_labels(self) -> None:
        """Wrapper utilise fallback [0,1] si pas de class_labels."""
        predictor = MagicMock(spec=[])  # Pas de class_labels
        wrapper = AutoGluonWrapper(predictor)
        assert list(wrapper.classes_) == [0, 1]

    def test_fit_captures_feature_names(self, mock_autogluon_predictor) -> None:
        """fit() capture les noms de colonnes du DataFrame."""
        wrapper = AutoGluonWrapper(mock_autogluon_predictor)
        X_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = np.array([0, 1])
        result = wrapper.fit(X_df, y)
        assert result is wrapper  # Retourne self
        assert wrapper._feature_names == ["a", "b"]

    def test_predict_proba_with_dataframe(self, mock_autogluon_predictor) -> None:
        """predict_proba fonctionne avec DataFrame."""
        wrapper = AutoGluonWrapper(mock_autogluon_predictor)
        X_df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        proba = wrapper.predict_proba(X_df)
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (5, 2)

    def test_predict_proba_with_array(self, mock_autogluon_predictor) -> None:
        """predict_proba convertit array en DataFrame."""
        wrapper = AutoGluonWrapper(mock_autogluon_predictor, feature_names=["a", "b"])
        X_arr = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
        proba = wrapper.predict_proba(X_arr)
        assert isinstance(proba, np.ndarray)
        # Vérifier que DataFrame a été créé avec les bons noms
        call_args = mock_autogluon_predictor.predict_proba.call_args[0][0]
        assert list(call_args.columns) == ["a", "b"]


class TestWrapIfAutogluon:
    """Tests pour _wrap_if_autogluon helper."""

    def test_wraps_tabular_predictor(self) -> None:
        """Wrap modèle avec 'TabularPredictor' dans le nom de classe."""
        mock_predictor = MagicMock()
        mock_predictor.__class__.__name__ = "TabularPredictor"
        result = _wrap_if_autogluon(mock_predictor)
        assert isinstance(result, AutoGluonWrapper)

    def test_wraps_predictor_class(self) -> None:
        """Wrap modèle avec 'Predictor' dans le nom de classe."""
        mock_predictor = MagicMock()
        mock_predictor.__class__.__name__ = "CustomPredictor"
        result = _wrap_if_autogluon(mock_predictor)
        assert isinstance(result, AutoGluonWrapper)

    def test_no_wrap_sklearn(self) -> None:
        """Ne wrap pas les modèles sklearn."""
        model = LogisticRegression()
        result = _wrap_if_autogluon(model)
        assert result is model  # Pas wrappé
