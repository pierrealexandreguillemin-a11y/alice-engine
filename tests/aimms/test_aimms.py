"""Tests for AIMMS module - ISO 29119.

Document ID: TEST-AIMMS-001
Version: 1.0.0
Tests count: 13

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management System
"""

from __future__ import annotations

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from scripts.aimms.aimms_types import (
    AIMSConfig,
    AIMSResult,
    AlertingSummary,
    CalibrationSummary,
    LifecyclePhase,
    UncertaintySummary,
)
from scripts.aimms.postprocessor import AIMSPostprocessor, run_postprocessing


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


class TestAIMSConfig:
    """Tests pour AIMSConfig."""

    def test_default_config(self) -> None:
        """Config par défaut."""
        config = AIMSConfig()
        assert config.enable_calibration is True
        assert config.enable_uncertainty is True
        assert config.enable_alerting is True
        assert config.calibration_cv == 0
        assert config.uncertainty_alpha == 0.10

    def test_custom_config(self) -> None:
        """Config personnalisée."""
        config = AIMSConfig(
            enable_calibration=False,
            uncertainty_alpha=0.05,
        )
        assert config.enable_calibration is False
        assert config.uncertainty_alpha == 0.05


class TestLifecyclePhase:
    """Tests pour LifecyclePhase enum."""

    def test_phases_exist(self) -> None:
        """Toutes les phases ISO 42001 existent."""
        assert LifecyclePhase.DEVELOPMENT.value == "development"
        assert LifecyclePhase.VALIDATION.value == "validation"
        assert LifecyclePhase.DEPLOYMENT.value == "deployment"
        assert LifecyclePhase.MONITORING.value == "monitoring"
        assert LifecyclePhase.RETIREMENT.value == "retirement"


class TestAIMSResult:
    """Tests pour AIMSResult."""

    def test_create_factory(self) -> None:
        """Factory method crée résultat vide."""
        result = AIMSResult.create("v1.0.0")
        assert result.model_version == "v1.0.0"
        assert result.phase == LifecyclePhase.VALIDATION
        assert result.calibration is None
        assert result.uncertainty is None

    def test_to_dict(self) -> None:
        """Conversion en dictionnaire."""
        result = AIMSResult.create("v1.0.0")
        result.recommendations = ["Test recommendation"]
        d = result.to_dict()
        assert d["iso_standard"] == "ISO/IEC 42001:2023"
        assert d["model_version"] == "v1.0.0"
        assert "Test recommendation" in d["recommendations"]


class TestCalibrationSummary:
    """Tests pour CalibrationSummary."""

    def test_to_dict(self) -> None:
        """Conversion en dictionnaire."""
        summary = CalibrationSummary(
            method="isotonic",
            brier_before=0.25,
            brier_after=0.20,
            ece_before=0.15,
            ece_after=0.10,
            improvement_pct=20.0,
        )
        d = summary.to_dict()
        assert d["method"] == "isotonic"
        assert d["improvement_pct"] == 20.0


class TestUncertaintySummary:
    """Tests pour UncertaintySummary."""

    def test_to_dict(self) -> None:
        """Conversion en dictionnaire."""
        summary = UncertaintySummary(
            method="conformal",
            coverage=0.92,
            mean_interval_width=0.45,
            singleton_rate=0.70,
            empty_set_rate=0.02,
        )
        d = summary.to_dict()
        assert d["coverage"] == 0.92
        assert d["method"] == "conformal"


class TestAlertingSummary:
    """Tests pour AlertingSummary."""

    def test_to_dict(self) -> None:
        """Conversion en dictionnaire."""
        summary = AlertingSummary(
            enabled=True,
            slack_configured=False,
            min_severity="warning",
            cooldown_minutes=60,
        )
        d = summary.to_dict()
        assert d["enabled"] is True
        assert d["slack_configured"] is False


class TestAIMSPostprocessor:
    """Tests pour AIMSPostprocessor."""

    def test_init_default(self) -> None:
        """Initialisation avec config par défaut."""
        pp = AIMSPostprocessor()
        assert pp.config.enable_calibration is True

    def test_run_full_pipeline(self, sample_data, trained_model) -> None:
        """Pipeline complet AIMMS."""
        _, X_calib, X_test, _, y_calib, y_test = sample_data
        pp = AIMSPostprocessor(AIMSConfig(calibration_cv=0))
        result = pp.run(trained_model, X_calib, y_calib, X_test, y_test, "test-v1")

        assert result.model_version == "test-v1"
        assert result.calibration is not None
        assert result.uncertainty is not None
        assert result.alerting is not None

    def test_run_calibration_only(self, sample_data, trained_model) -> None:
        """Pipeline calibration uniquement."""
        _, X_calib, X_test, _, y_calib, y_test = sample_data
        config = AIMSConfig(
            enable_calibration=True,
            enable_uncertainty=False,
            enable_alerting=False,
            calibration_cv=0,
        )
        pp = AIMSPostprocessor(config)
        result = pp.run(trained_model, X_calib, y_calib, X_test, y_test)

        assert result.calibration is not None
        assert result.uncertainty is None
        assert result.alerting is None

    def test_recommendations_generated(self, sample_data, trained_model) -> None:
        """Recommandations ISO 42001 générées."""
        _, X_calib, X_test, _, y_calib, y_test = sample_data
        pp = AIMSPostprocessor(AIMSConfig(calibration_cv=0))
        result = pp.run(trained_model, X_calib, y_calib, X_test, y_test)

        assert len(result.recommendations) > 0
        # Au minimum, recommandation pour Slack non configuré
        assert any("ALERTING" in r or "AIMMS" in r for r in result.recommendations)


class TestRunPostprocessing:
    """Tests pour run_postprocessing helper."""

    def test_helper_function(self, sample_data, trained_model) -> None:
        """Helper function fonctionne."""
        _, X_calib, X_test, _, y_calib, y_test = sample_data
        result = run_postprocessing(
            model=trained_model,
            X_calib=X_calib,
            y_calib=y_calib,
            X_test=X_test,
            y_test=y_test,
            model_version="helper-test",
            config=AIMSConfig(calibration_cv=0),
        )
        assert result.model_version == "helper-test"
        assert result.phase == LifecyclePhase.VALIDATION
