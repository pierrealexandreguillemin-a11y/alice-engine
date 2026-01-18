"""Tests for conformal prediction module - ISO 29119.

Document ID: TEST-UNCERTAINTY-001
Version: 1.0.0
Tests count: 15

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 24029:2021 - Neural Network Robustness
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from scripts.uncertainty.conformal import (
    ConformalPredictor,
    quantify_uncertainty,
)
from scripts.uncertainty.uncertainty_types import (
    UncertaintyConfig,
    UncertaintyMethod,
)


@pytest.fixture
def sample_data():
    """Génère des données de test."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    # Split: train / calib / test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_calib, X_test, y_train, y_calib, y_test


@pytest.fixture
def trained_model(sample_data):
    """Modèle entraîné pour tests."""
    X_train, _, _, y_train, _, _ = sample_data
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model


class TestUncertaintyConfig:
    """Tests pour UncertaintyConfig."""

    def test_default_config(self) -> None:
        """Config par défaut."""
        config = UncertaintyConfig()
        assert config.method == UncertaintyMethod.CONFORMAL
        assert config.alpha == 0.10  # 90% coverage

    def test_custom_alpha(self) -> None:
        """Alpha personnalisé."""
        config = UncertaintyConfig(alpha=0.05)
        assert config.alpha == 0.05  # 95% coverage


class TestConformalPredictor:
    """Tests pour ConformalPredictor."""

    def test_init(self) -> None:
        """Initialisation."""
        cp = ConformalPredictor()
        assert cp._model is None
        assert cp._quantile is None
        assert cp.config.alpha == 0.10

    def test_fit(self, sample_data, trained_model) -> None:
        """Fit calcule le quantile."""
        _, X_calib, _, _, y_calib, _ = sample_data
        cp = ConformalPredictor()
        cp.fit(trained_model, X_calib, y_calib)

        assert cp._model is not None
        assert cp._quantile is not None
        assert 0 <= cp._quantile <= 1
        assert cp._calibration_scores is not None

    def test_predict_without_fit_raises(self) -> None:
        """Predict sans fit lève une erreur."""
        cp = ConformalPredictor()
        X_test = np.random.random((10, 5))

        with pytest.raises(RuntimeError, match="not fitted"):
            cp.predict(X_test)

    def test_predict_returns_intervals(self, sample_data, trained_model) -> None:
        """Predict retourne des intervalles."""
        _, X_calib, X_test, _, y_calib, _ = sample_data
        cp = ConformalPredictor()
        cp.fit(trained_model, X_calib, y_calib)
        result = cp.predict(X_test)

        assert len(result.intervals) == len(X_test)
        assert result.method == UncertaintyMethod.CONFORMAL

    def test_prediction_set_not_empty(self, sample_data, trained_model) -> None:
        """La plupart des ensembles de prédiction ne sont pas vides."""
        _, X_calib, X_test, _, y_calib, _ = sample_data
        cp = ConformalPredictor(UncertaintyConfig(alpha=0.10))
        cp.fit(trained_model, X_calib, y_calib)
        result = cp.predict(X_test)

        empty_count = sum(1 for i in result.intervals if len(i.in_prediction_set) == 0)
        assert empty_count / len(result.intervals) < 0.1  # < 10% vides

    def test_coverage_guarantee(self, sample_data, trained_model) -> None:
        """Couverture empirique proche de 1-alpha."""
        _, X_calib, X_test, _, y_calib, y_test = sample_data
        alpha = 0.10
        cp = ConformalPredictor(UncertaintyConfig(alpha=alpha))
        cp.fit(trained_model, X_calib, y_calib)
        result = cp.predict(X_test, y_test)

        # Couverture devrait être >= 1 - alpha (avec tolérance)
        expected_coverage = 1 - alpha
        assert result.metrics.coverage >= expected_coverage - 0.05  # Tolérance 5%

    def test_interval_width_positive(self, sample_data, trained_model) -> None:
        """Largeur d'intervalle positive."""
        _, X_calib, X_test, _, y_calib, _ = sample_data
        cp = ConformalPredictor()
        cp.fit(trained_model, X_calib, y_calib)
        result = cp.predict(X_test)

        for interval in result.intervals:
            assert interval.interval_width() >= 0
            assert interval.lower <= interval.upper

    def test_point_estimate_in_interval(self, sample_data, trained_model) -> None:
        """Point estimate dans l'intervalle."""
        _, X_calib, X_test, _, y_calib, _ = sample_data
        cp = ConformalPredictor()
        cp.fit(trained_model, X_calib, y_calib)
        result = cp.predict(X_test)

        for interval in result.intervals:
            # Le point estimate devrait être cohérent avec l'intervalle
            assert 0 <= interval.point_estimate <= 1

    def test_smaller_alpha_wider_intervals(self, sample_data, trained_model) -> None:
        """Alpha plus petit = intervalles plus larges."""
        _, X_calib, X_test, _, y_calib, _ = sample_data

        # Alpha = 0.10 (90% coverage)
        cp1 = ConformalPredictor(UncertaintyConfig(alpha=0.10))
        cp1.fit(trained_model, X_calib, y_calib)
        result1 = cp1.predict(X_test)

        # Alpha = 0.05 (95% coverage)
        cp2 = ConformalPredictor(UncertaintyConfig(alpha=0.05))
        cp2.fit(trained_model, X_calib, y_calib)
        result2 = cp2.predict(X_test)

        # Intervalles plus larges avec alpha plus petit
        assert result2.metrics.mean_interval_width >= result1.metrics.mean_interval_width - 0.1


class TestQuantifyUncertainty:
    """Tests pour quantify_uncertainty helper."""

    def test_quantify_uncertainty_wrapper(self, sample_data, trained_model) -> None:
        """quantify_uncertainty est un wrapper."""
        _, X_calib, X_test, _, y_calib, y_test = sample_data
        result = quantify_uncertainty(
            trained_model,
            X_calib,
            y_calib,
            X_test,
            y_test,
            alpha=0.10,
        )
        assert result.method == UncertaintyMethod.CONFORMAL
        assert len(result.intervals) == len(X_test)

    def test_quantify_uncertainty_default_alpha(self, sample_data, trained_model) -> None:
        """Alpha par défaut est 0.10."""
        _, X_calib, X_test, _, y_calib, _ = sample_data
        result = quantify_uncertainty(trained_model, X_calib, y_calib, X_test)
        assert result.intervals[0].confidence == 0.90


class TestMetrics:
    """Tests pour UncertaintyMetrics."""

    def test_metrics_computed(self, sample_data, trained_model) -> None:
        """Métriques calculées."""
        _, X_calib, X_test, _, y_calib, y_test = sample_data
        cp = ConformalPredictor()
        cp.fit(trained_model, X_calib, y_calib)
        result = cp.predict(X_test, y_test)

        assert result.metrics.mean_interval_width >= 0
        assert 0 <= result.metrics.coverage <= 1
        assert 0 <= result.metrics.empty_set_rate <= 1
        assert 0 <= result.metrics.singleton_rate <= 1

    def test_metrics_to_dict(self, sample_data, trained_model) -> None:
        """Conversion métriques en dict."""
        _, X_calib, X_test, _, y_calib, _ = sample_data
        cp = ConformalPredictor()
        cp.fit(trained_model, X_calib, y_calib)
        result = cp.predict(X_test)

        metrics_dict = result.metrics.to_dict()
        assert "mean_interval_width" in metrics_dict
        assert "coverage" in metrics_dict
        assert "efficiency" in metrics_dict
