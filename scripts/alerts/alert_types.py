"""Types pour alerting drift - ISO 23894.

Structures de données pour les alertes de drift ML.

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 5055:2021 - Code Quality (SRP)

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class AlertSeverity(IntEnum):
    """Niveaux de sévérité des alertes (ISO 23894).

    Utilise IntEnum pour comparaison correcte: INFO < WARNING < CRITICAL.
    """

    INFO = 1
    WARNING = 2
    CRITICAL = 3

    def to_string(self) -> str:
        """Retourne le nom en minuscules."""
        return self.name.lower()


class AlertChannel(Enum):
    """Canaux de notification."""

    SLACK = "slack"
    EMAIL = "email"
    LOG = "log"


@dataclass
class AlertConfig:
    """Configuration des alertes drift.

    Attributes:
        slack_webhook_url: URL du webhook Slack (env: ALICE_SLACK_WEBHOOK)
        enable_slack: Activer les notifications Slack
        enable_email: Activer les notifications email (future)
        min_severity: Sévérité minimale pour déclencher alerte
        cooldown_minutes: Temps minimum entre deux alertes identiques
    """

    slack_webhook_url: str = ""
    enable_slack: bool = True
    enable_email: bool = False
    min_severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_minutes: int = 60


@dataclass
class DriftAlert:
    """Alerte de drift détectée.

    Attributes:
        severity: Niveau de sévérité
        title: Titre court de l'alerte
        message: Description détaillée
        metrics: Métriques associées (PSI, accuracy, etc.)
        recommendation: Action recommandée (ISO 23894)
        timestamp: Horodatage ISO 8601
    """

    severity: AlertSeverity
    title: str
    message: str
    metrics: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    timestamp: str = ""

    def to_slack_payload(self, channel: str = "#alice-alerts") -> dict:
        """Convertit en payload Slack Block Kit."""
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.CRITICAL: "#ff0000",
        }.get(self.severity, "#808080")

        emoji = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.CRITICAL: ":rotating_light:",
        }.get(self.severity, ":bell:")

        metrics_text = "\n".join(f"  {k}: {v:.4f}" for k, v in self.metrics.items())

        return {
            "channel": channel,
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {"type": "plain_text", "text": f"{emoji} {self.title}"},
                        },
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": self.message},
                        },
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": f"*Metrics:*\n```{metrics_text}```"},
                        },
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": f"*Recommendation:* {self.recommendation}"},
                        },
                        {
                            "type": "context",
                            "elements": [{"type": "mrkdwn", "text": f"Timestamp: {self.timestamp}"}],
                        },
                    ],
                }
            ],
        }
