"""Drift Alerter - ISO 23894 (AI Risk Management).

Module d'alerting automatique pour drift ML via Slack webhook.

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management (alerting, monitoring)
- ISO/IEC 5055:2021 - Code Quality (SRP)
- ISO/IEC 27034 - Secure Coding (URL validation)

Usage:
    from scripts.alerts import DriftAlerter, AlertConfig

    alerter = DriftAlerter(AlertConfig(slack_webhook_url="https://..."))
    alerter.check_and_alert(drift_report)

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from scripts.model_registry.drift_dataclasses import DriftMetrics, DriftReport

from scripts.alerts.alert_types import (
    AlertConfig,
    AlertSeverity,
    DriftAlert,
)

logger = logging.getLogger(__name__)

# Variable d'environnement pour Slack webhook
ENV_SLACK_WEBHOOK = "ALICE_SLACK_WEBHOOK"

# Préfixe valide pour webhooks Slack
SLACK_WEBHOOK_PREFIX = "https://hooks.slack.com/"


def _validate_slack_url(url: str) -> bool:
    """Valide qu'une URL est un webhook Slack valide (ISO 27034).

    Args:
        url: URL à valider

    Returns:
        True si URL valide, False sinon
    """
    if not url:
        return False
    if not url.startswith(SLACK_WEBHOOK_PREFIX):
        logger.warning(f"Invalid Slack webhook URL: must start with {SLACK_WEBHOOK_PREFIX}")
        return False
    return True


class DriftAlerter:
    """Gestionnaire d'alertes drift (ISO 23894).

    Surveille les rapports de drift et envoie des alertes
    via Slack lorsque les seuils sont dépassés.

    Example:
        config = AlertConfig(slack_webhook_url=os.environ["ALICE_SLACK_WEBHOOK"])
        alerter = DriftAlerter(config)

        # Après chaque ronde de prédictions
        alert = alerter.check_and_alert(drift_report)
        if alert:
            print(f"Alert sent: {alert.title}")
    """

    def __init__(self, config: AlertConfig | None = None) -> None:
        """Initialise l'alerter avec configuration."""
        self.config = config or AlertConfig()
        self._last_alert_time: datetime | None = None

        # Priorité: config > env var
        if not self.config.slack_webhook_url:
            self.config.slack_webhook_url = os.environ.get(ENV_SLACK_WEBHOOK, "")

    def check_and_alert(self, report: DriftReport) -> DriftAlert | None:
        """Vérifie le drift et envoie une alerte si nécessaire.

        Args:
            report: Rapport de drift à analyser

        Returns:
            DriftAlert si alerte envoyée, None sinon
        """
        if not report.rounds:
            return None

        latest = report.rounds[-1]
        summary = report.get_summary()
        recommendation = summary.get("recommendation", "OK")

        # Déterminer sévérité
        severity = self._determine_severity(latest, recommendation)

        if severity < self.config.min_severity:
            logger.debug(
                f"Drift severity {severity.name} below threshold {self.config.min_severity.name}"
            )
            return None

        # Vérifier cooldown
        if not self._check_cooldown():
            logger.debug("Alert cooldown active, skipping")
            return None

        # Créer et envoyer alerte
        alert = self._create_alert(report, latest, severity, recommendation)

        if self.config.enable_slack and _validate_slack_url(self.config.slack_webhook_url):
            self._send_slack_alert(alert)

        self._last_alert_time = datetime.now()
        return alert

    def _determine_severity(self, metrics: DriftMetrics, recommendation: str) -> AlertSeverity:
        """Détermine la sévérité basée sur métriques et recommandation."""
        if metrics.has_critical or recommendation == "RETRAIN_URGENT":
            return AlertSeverity.CRITICAL
        if metrics.has_warning or recommendation in ("RETRAIN_RECOMMENDED", "MONITOR_CLOSELY"):
            return AlertSeverity.WARNING
        return AlertSeverity.INFO

    def _check_cooldown(self) -> bool:
        """Vérifie si le cooldown est respecté."""
        if self._last_alert_time is None:
            return True
        elapsed = (datetime.now() - self._last_alert_time).total_seconds() / 60
        return elapsed >= self.config.cooldown_minutes

    def _create_alert(
        self,
        report: DriftReport,
        latest: DriftMetrics,
        severity: AlertSeverity,
        recommendation: str,
    ) -> DriftAlert:
        """Crée une alerte drift structurée."""
        titles = {
            AlertSeverity.CRITICAL: "CRITICAL: Model Drift Detected",
            AlertSeverity.WARNING: "WARNING: Model Drift Warning",
            AlertSeverity.INFO: "INFO: Drift Metrics Update",
        }

        recommendations = {
            "RETRAIN_URGENT": "Retraining urgent required. Model performance severely degraded.",
            "RETRAIN_RECOMMENDED": "Consider retraining the model soon.",
            "MONITOR_CLOSELY": "Monitor closely for further degradation.",
            "OK": "No action required.",
        }

        return DriftAlert(
            severity=severity,
            title=titles.get(severity, "Drift Alert"),
            message=f"Model `{report.model_version}` - Season `{report.season}` - Round {latest.round_number}",
            metrics={
                "psi_score": latest.psi_score,
                "accuracy": latest.accuracy,
                "elo_mean_shift": latest.elo_mean_shift,
            },
            recommendation=recommendations.get(recommendation, recommendation),
            timestamp=datetime.now().isoformat(),
        )

    def _send_slack_alert(self, alert: DriftAlert) -> bool:
        """Envoie l'alerte via Slack webhook."""
        payload = alert.to_slack_payload()

        try:
            # URL validée par _validate_slack_url avant appel (HTTPS uniquement)
            req = Request(  # noqa: S310
                self.config.slack_webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=10) as response:  # noqa: S310
                if response.status == 200:
                    logger.info(f"Slack alert sent: {alert.title}")
                    return True
                logger.warning(f"Slack webhook returned {response.status}")
                return False
        except URLError as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


def send_drift_alert(report: DriftReport, config: AlertConfig | None = None) -> DriftAlert | None:
    """Fonction utilitaire pour envoi rapide d'alerte.

    Args:
        report: Rapport de drift
        config: Configuration optionnelle

    Returns:
        DriftAlert si envoyée, None sinon
    """
    alerter = DriftAlerter(config)
    return alerter.check_and_alert(report)
