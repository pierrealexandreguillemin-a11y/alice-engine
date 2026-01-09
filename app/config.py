"""Module: config.py - Configuration centralisee.

Toutes les variables d'environnement sont chargees ici.
Aucun secret hardcode dans le code.

ISO Compliance:
- ISO/IEC 27001 - Information Security (secrets en env vars)
- ISO/IEC 27034 - Secure Coding (CWE-798: pas de hardcoded secrets)
- ISO/IEC 25010 - System Quality (configurabilite)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration de l'application.

    Les valeurs sont chargees depuis les variables d'environnement
    ou le fichier .env (jamais committes).

    @see ISO 27001 - Gestion des secrets
    @see ISO 27034 - CWE-798 (pas de hardcoded secrets)
    """

    # Application
    app_name: str = "ALICE"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # API (0.0.0.0 requis pour Render/Docker)
    api_host: str = "0.0.0.0"  # noqa: S104  # nosec B104 - required for Docker/Render
    api_port: int = 8000

    # MongoDB (lecture seule)
    mongodb_uri: str = ""
    mongodb_database: str = "chess-app"

    # Securite
    api_key: str = ""  # Pour endpoint /train

    # Modeles ML
    model_path: str = "./models"
    default_scenario_count: int = 20

    # Render (production)
    render_external_url: str = ""

    class Config:
        """Pydantic settings configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignorer les vars env non definies dans Settings


@lru_cache
def get_settings() -> Settings:
    """Singleton pour la configuration.

    Utilise lru_cache pour ne charger qu'une fois.
    """
    return Settings()


# Export pour import facile
settings = get_settings()
