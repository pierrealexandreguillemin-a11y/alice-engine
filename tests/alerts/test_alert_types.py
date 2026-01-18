"""Tests for alert_types module - ISO 29119.

Document ID: TEST-ALERTS-TYPES-001
Version: 1.0.0
Tests count: 8

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 23894:2023 - AI Risk Management
"""

from __future__ import annotations

from scripts.alerts.alert_types import (
    AlertConfig,
    AlertSeverity,
    DriftAlert,
)


class TestAlertSeverity:
    """Tests pour AlertSeverity IntEnum."""

    def test_severity_ordering(self) -> None:
        """Vérifie que INFO < WARNING < CRITICAL."""
        assert AlertSeverity.INFO < AlertSeverity.WARNING
        assert AlertSeverity.WARNING < AlertSeverity.CRITICAL
        assert AlertSeverity.INFO < AlertSeverity.CRITICAL

    def test_severity_values(self) -> None:
        """Vérifie les valeurs numériques."""
        assert AlertSeverity.INFO == 1
        assert AlertSeverity.WARNING == 2
        assert AlertSeverity.CRITICAL == 3

    def test_severity_comparison_not_alphabetical(self) -> None:
        """Vérifie que la comparaison n'est PAS alphabétique."""
        # "critical" < "info" alphabétiquement, mais CRITICAL > INFO numériquement
        assert AlertSeverity.CRITICAL > AlertSeverity.INFO

    def test_to_string(self) -> None:
        """Vérifie la conversion en string."""
        assert AlertSeverity.INFO.to_string() == "info"
        assert AlertSeverity.WARNING.to_string() == "warning"
        assert AlertSeverity.CRITICAL.to_string() == "critical"


class TestAlertConfig:
    """Tests pour AlertConfig."""

    def test_default_values(self) -> None:
        """Vérifie les valeurs par défaut."""
        config = AlertConfig()
        assert config.slack_webhook_url == ""
        assert config.enable_slack is True
        assert config.enable_email is False
        assert config.min_severity == AlertSeverity.WARNING
        assert config.cooldown_minutes == 60

    def test_custom_values(self) -> None:
        """Vérifie les valeurs personnalisées."""
        config = AlertConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            min_severity=AlertSeverity.CRITICAL,
            cooldown_minutes=30,
        )
        assert config.slack_webhook_url == "https://hooks.slack.com/test"
        assert config.min_severity == AlertSeverity.CRITICAL
        assert config.cooldown_minutes == 30


class TestDriftAlert:
    """Tests pour DriftAlert."""

    def test_create_alert(self) -> None:
        """Vérifie la création d'une alerte."""
        alert = DriftAlert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
            metrics={"psi": 0.15},
            recommendation="Monitor",
            timestamp="2026-01-17T12:00:00",
        )
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.metrics["psi"] == 0.15

    def test_to_slack_payload(self) -> None:
        """Vérifie la génération du payload Slack."""
        alert = DriftAlert(
            severity=AlertSeverity.CRITICAL,
            title="Critical Alert",
            message="Urgent",
            metrics={"accuracy": 0.75},
        )
        payload = alert.to_slack_payload()

        assert "channel" in payload
        assert "attachments" in payload
        assert len(payload["attachments"]) == 1
        assert payload["attachments"][0]["color"] == "#ff0000"  # Red for critical
