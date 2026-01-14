"""Tests Bias Tracker - ISO 24027.

Document ID: ALICE-TEST-BIAS-TRACK
Version: 1.0.0
Tests: 8

Classes:
- TestComputeBiasMetrics: Tests calcul métriques (2 tests)
- TestCheckBiasAlerts: Tests détection alertes (2 tests)
- TestMonitorBias: Tests monitoring complet (3 tests)
- TestConfigIO: Tests persistance (1 test)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<120 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import tempfile
from pathlib import Path

from scripts.monitoring.bias_tracker import (
    compute_bias_metrics,
    load_bias_config,
    monitor_bias,
    save_bias_config,
)
from scripts.monitoring.bias_types import (
    BiasMonitorResult,
    FairnessStatus,
)

# Fixtures (biased_predictions, default_config, fair_predictions, strict_config, y_prob)
# are auto-loaded via pytest_plugins in conftest.py


class TestComputeBiasMetrics:
    """Tests pour compute_bias_metrics."""

    def test_metrics_by_group(self, fair_predictions):
        """Calcule métriques par groupe."""
        y_true, y_pred, protected = fair_predictions
        metrics = compute_bias_metrics(y_true, y_pred, protected, "group")
        assert len(metrics) == 2  # Groups A and B
        assert all(m.n_samples > 0 for m in metrics)

    def test_metrics_with_probs(self, fair_predictions, y_prob):
        """Calcule métriques avec probabilités."""
        y_true, y_pred, protected = fair_predictions
        metrics = compute_bias_metrics(y_true, y_pred, protected, "group", y_prob=y_prob)
        assert all(m.calibration is not None for m in metrics)


class TestCheckBiasAlerts:
    """Tests pour check_bias_alerts."""

    def test_no_alerts_fair(self, fair_predictions, default_config):
        """Pas d'alertes pour modèle équitable."""
        y_true, y_pred, protected = fair_predictions
        result = monitor_bias(y_true, y_pred, protected, "gender", default_config)
        # Fair predictions should have few/no alerts
        assert result.overall_status in (FairnessStatus.FAIR, FairnessStatus.CAUTION)

    def test_alerts_biased(self, biased_predictions, default_config):
        """Alertes pour modèle biaisé."""
        y_true, y_pred, protected = biased_predictions
        result = monitor_bias(y_true, y_pred, protected, "gender", default_config)
        assert len(result.alerts) > 0
        assert result.overall_status in (FairnessStatus.BIASED, FairnessStatus.CRITICAL)


class TestMonitorBias:
    """Tests pour monitor_bias."""

    def test_monitor_structure(self, fair_predictions):
        """Vérifie la structure du résultat."""
        y_true, y_pred, protected = fair_predictions
        result = monitor_bias(y_true, y_pred, protected, "test_attr")

        assert isinstance(result, BiasMonitorResult)
        assert result.protected_attribute == "test_attr"
        assert result.n_total_samples == len(y_true)
        assert 0 <= result.demographic_parity <= 1

    def test_monitor_to_dict(self, fair_predictions, default_config):
        """Vérifie la sérialisation."""
        y_true, y_pred, protected = fair_predictions
        result = monitor_bias(y_true, y_pred, protected, "gender", default_config)
        d = result.to_dict()

        assert "timestamp" in d
        assert "metrics" in d
        assert "groups" in d
        assert "status" in d

    def test_monitor_recommendations(self, biased_predictions, default_config):
        """Génère recommandations pour biais."""
        y_true, y_pred, protected = biased_predictions
        result = monitor_bias(y_true, y_pred, protected, "gender", default_config)
        if result.overall_status in (FairnessStatus.BIASED, FairnessStatus.CRITICAL):
            assert len(result.recommendations) > 0


class TestConfigIO:
    """Tests pour persistance config."""

    def test_save_load_config(self, default_config):
        """Sauvegarde et chargement config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bias_config.json"
            save_bias_config(default_config, path)
            loaded = load_bias_config(path)

            assert loaded is not None
            assert loaded.model_version == default_config.model_version
            assert (
                loaded.demographic_parity_threshold == default_config.demographic_parity_threshold
            )
