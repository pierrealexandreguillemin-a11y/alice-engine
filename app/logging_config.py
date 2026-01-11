"""Module: logging_config.py - Configuration logging centralisee.

Audit logging structure pour tracabilite ISO 27001.
Logs sans donnees sensibles (pas de tokens, passwords, PII).

ISO Compliance:
- ISO/IEC 27001 - Information Security (audit trails)
- ISO/IEC 27034 - Secure Coding (pas de secrets dans logs)
- ISO/IEC 42001 - AI Management (tracabilite decisions AI)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import MutableMapping

# Champs sensibles a masquer (ISO 27001)
SENSITIVE_FIELDS = frozenset(
    {
        "password",
        "api_key",
        "x_api_key",
        "token",
        "secret",
        "authorization",
        "cookie",
        "credit_card",
    }
)


def _mask_sensitive_data(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Masque les donnees sensibles dans les logs (ISO 27001)."""
    for key in list(event_dict.keys()):
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in SENSITIVE_FIELDS):
            event_dict[key] = "[REDACTED]"
    return event_dict


def _add_app_context(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Ajoute le contexte applicatif aux logs."""
    event_dict["service"] = "alice-engine"
    return event_dict


def configure_logging(*, debug: bool = False, json_format: bool = True) -> None:
    """Configure le logging structure pour l'application.

    Args:
    ----
        debug: Active le niveau DEBUG
        json_format: Format JSON (production) ou console (dev)

    """
    log_level = logging.DEBUG if debug else logging.INFO

    # Processeurs structlog
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_app_context,
        _mask_sensitive_data,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_format and not debug:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configuration logging stdlib
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> Any:
    """Retourne un logger structure.

    Args:
    ----
        name: Nom du module (typiquement __name__)

    Returns:
    -------
        Logger structure avec contexte

    """
    return structlog.get_logger(name)


# Logger pour audit securite (ISO 27001)
def get_audit_logger() -> Any:
    """Retourne le logger d'audit securite.

    Utilise pour:
    - Authentification (succes/echec)
    - Autorisation (acces ressources)
    - Operations sensibles (encryption, training)
    - Erreurs securite

    Returns
    -------
        Logger audit avec prefixe "audit"

    """
    return structlog.get_logger("audit.security")
