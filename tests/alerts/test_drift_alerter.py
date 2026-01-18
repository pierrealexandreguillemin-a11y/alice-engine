"""Tests for drift_alerter module - ISO 29119.

Document ID: TEST-ALERTS-ALERTER-001
Version: 1.0.0
Tests count: 14

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 27034 - Secure Coding (URL validation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from scripts.alerts.alert_types import AlertConfig, AlertSeverity
from scripts.alerts.drift_alerter import (
    SLACK_WEBHOOK_PREFIX,
    DriftAlerter,
    _validate_slack_url,
    send_drift_alert,
)


# Mock DriftMetrics pour tests
@dataclass
class MockDriftMetrics:
    """Mock de DriftMetrics pour tests."""

    round_number: int = 1
    psi_score: float = 0.15
    accuracy: float = 0.80
    elo_mean_shift: float = 25.0
    has_warning: bool = False
    has_critical: bool = False


@dataclass
class MockDriftReport:
    """Mock de DriftReport pour tests."""

    model_version: str = "v1.0.0"
    season: str = "2025-2026"
    rounds: list = field(default_factory=list)

    def get_summary(self) -> dict:
        """Retourne un summary mock."""
        if not self.rounds:
            return {"recommendation": "OK"}
        latest = self.rounds[-1]
        if latest.has_critical:
            return {"recommendation": "RETRAIN_URGENT"}
        if latest.has_warning:
            return {"recommendation": "RETRAIN_RECOMMENDED"}
        return {"recommendation": "OK"}


class TestValidateSlackUrl:
    """Tests pour _validate_slack_url."""

    def test_valid_slack_url(self) -> None:
        """URL Slack valide."""
        url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXX"
        assert _validate_slack_url(url) is True

    def test_empty_url(self) -> None:
        """URL vide retourne False."""
        assert _validate_slack_url("") is False

    def test_http_url_rejected(self) -> None:
        """URL HTTP (non HTTPS) rejetée."""
        url = "http://hooks.slack.com/services/T00000000/B00000000/XXXXXXX"
        assert _validate_slack_url(url) is False

    def test_non_slack_url_rejected(self) -> None:
        """URL non-Slack rejetée."""
        assert _validate_slack_url("https://example.com/webhook") is False
        assert _validate_slack_url("https://google.com") is False

    def test_slack_url_prefix(self) -> None:
        """Vérifie le préfixe attendu."""
        assert SLACK_WEBHOOK_PREFIX == "https://hooks.slack.com/"


class TestDriftAlerter:
    """Tests pour DriftAlerter."""

    def test_init_default_config(self) -> None:
        """Initialisation avec config par défaut."""
        alerter = DriftAlerter()
        assert alerter.config.min_severity == AlertSeverity.WARNING
        assert alerter._last_alert_time is None

    def test_init_custom_config(self) -> None:
        """Initialisation avec config personnalisée."""
        config = AlertConfig(min_severity=AlertSeverity.CRITICAL)
        alerter = DriftAlerter(config)
        assert alerter.config.min_severity == AlertSeverity.CRITICAL

    def test_check_and_alert_empty_report(self) -> None:
        """Rapport vide ne génère pas d'alerte."""
        alerter = DriftAlerter()
        report = MockDriftReport(rounds=[])
        alert = alerter.check_and_alert(report)
        assert alert is None

    def test_check_and_alert_info_below_threshold(self) -> None:
        """Alerte INFO ignorée si min_severity=WARNING."""
        alerter = DriftAlerter(AlertConfig(min_severity=AlertSeverity.WARNING))
        report = MockDriftReport(rounds=[MockDriftMetrics()])  # No warning/critical
        alert = alerter.check_and_alert(report)
        assert alert is None

    def test_check_and_alert_warning_above_threshold(self) -> None:
        """Alerte WARNING envoyée si min_severity=WARNING."""
        config = AlertConfig(min_severity=AlertSeverity.WARNING, enable_slack=False)
        alerter = DriftAlerter(config)
        report = MockDriftReport(rounds=[MockDriftMetrics(has_warning=True)])
        alert = alerter.check_and_alert(report)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING

    def test_check_and_alert_critical(self) -> None:
        """Alerte CRITICAL toujours envoyée."""
        config = AlertConfig(min_severity=AlertSeverity.CRITICAL, enable_slack=False)
        alerter = DriftAlerter(config)
        report = MockDriftReport(rounds=[MockDriftMetrics(has_critical=True)])
        alert = alerter.check_and_alert(report)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL

    def test_cooldown_respected(self) -> None:
        """Cooldown empêche les alertes répétées."""
        config = AlertConfig(cooldown_minutes=60, enable_slack=False)
        alerter = DriftAlerter(config)
        report = MockDriftReport(rounds=[MockDriftMetrics(has_warning=True)])

        # Première alerte
        alert1 = alerter.check_and_alert(report)
        assert alert1 is not None

        # Deuxième alerte immédiate bloquée par cooldown
        alert2 = alerter.check_and_alert(report)
        assert alert2 is None

    def test_cooldown_expired(self) -> None:
        """Alerte envoyée après expiration du cooldown."""
        config = AlertConfig(cooldown_minutes=1, enable_slack=False)
        alerter = DriftAlerter(config)
        report = MockDriftReport(rounds=[MockDriftMetrics(has_warning=True)])

        # Première alerte
        alert1 = alerter.check_and_alert(report)
        assert alert1 is not None

        # Simuler expiration du cooldown
        alerter._last_alert_time = datetime.now() - timedelta(minutes=5)

        # Deuxième alerte acceptée
        alert2 = alerter.check_and_alert(report)
        assert alert2 is not None


class TestSendDriftAlert:
    """Tests pour send_drift_alert helper."""

    def test_send_drift_alert_wrapper(self) -> None:
        """send_drift_alert est un wrapper pour DriftAlerter."""
        config = AlertConfig(enable_slack=False)
        report = MockDriftReport(rounds=[MockDriftMetrics(has_critical=True)])
        alert = send_drift_alert(report, config)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
