"""Signature HMAC-SHA256 - ISO 27001.

Ce module implémente la signature des modèles pour intégrité.

Conformité ISO/IEC 27001 (Authenticity).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import platform
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

# Variable d'environnement pour la clé
ENV_SIGNING_KEY = "ALICE_SIGNING_KEY"


def generate_signing_key() -> str:
    """Génère une clé de signature HMAC-SHA256 (32 bytes hex)."""
    return secrets.token_hex(32)


def compute_model_signature(file_path: Path, secret_key: str) -> str:
    """Calcule la signature HMAC-SHA256 d'un fichier modèle.

    Args:
    ----
        file_path: Chemin vers le fichier modèle
        secret_key: Clé secrète HMAC (hex string)

    Returns:
    -------
        Signature HMAC-SHA256 en hex
    """
    key_bytes = bytes.fromhex(secret_key)
    h = hmac.new(key_bytes, digestmod=hashlib.sha256)

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_model_signature(file_path: Path, signature: str, secret_key: str) -> bool:
    """Vérifie la signature HMAC-SHA256 d'un fichier modèle.

    Args:
    ----
        file_path: Chemin vers le fichier modèle
        signature: Signature attendue (hex)
        secret_key: Clé secrète HMAC (hex string)

    Returns:
    -------
        True si signature valide, False sinon
    """
    if not file_path.exists():
        logger.error(f"File not found for signature verification: {file_path}")
        return False

    computed = compute_model_signature(file_path, secret_key)
    is_valid = hmac.compare_digest(computed, signature)

    if not is_valid:
        logger.error(f"Signature mismatch for {file_path.name}")

    return is_valid


def save_signing_key(key: str, key_path: Path) -> None:
    """Sauvegarde la clé de signature de manière sécurisée."""
    key_path.write_text(key)
    if platform.system() != "Windows":
        os.chmod(key_path, 0o600)
    logger.info(f"  Signing key saved to {key_path.name}")


def load_signing_key(key_path: Path) -> str | None:
    """Charge la clé de signature."""
    if not key_path.exists():
        logger.warning(f"Signing key not found: {key_path}")
        return None
    return key_path.read_text().strip()


def get_signing_key_from_env() -> str | None:
    """Récupère la clé de signature depuis l'environnement."""
    return os.environ.get(ENV_SIGNING_KEY)
