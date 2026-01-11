"""Chiffrement AES-256-GCM - ISO 27001.

Ce module implémente le chiffrement des modèles pour confidentialité.

Conformité ISO/IEC 27001, 27034 (Confidentiality).
"""

from __future__ import annotations

import logging
import os
import platform
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

# Extension pour fichiers chiffrés
ENCRYPTED_EXTENSION = ".enc"

# Variable d'environnement pour la clé
ENV_ENCRYPTION_KEY = "ALICE_ENCRYPTION_KEY"


def get_key_from_env(env_var: str) -> bytes | None:
    """Récupère une clé depuis une variable d'environnement.

    Args:
    ----
        env_var: Nom de la variable d'environnement

    Returns:
    -------
        Clé en bytes ou None si non définie

    Note:
    ----
        ISO 27001 - Les clés ne doivent JAMAIS être stockées avec les données.
    """
    import base64

    key_value = os.environ.get(env_var)
    if not key_value:
        return None

    try:
        return base64.b64decode(key_value)
    except Exception:
        logger.warning(f"Invalid base64 in {env_var}")
        return None


def generate_encryption_key() -> bytes:
    """Génère une clé de chiffrement AES-256 (32 bytes)."""
    return secrets.token_bytes(32)


def save_encryption_key(key: bytes, key_path: Path) -> None:
    """Sauvegarde la clé de chiffrement de manière sécurisée."""
    import base64

    key_b64 = base64.b64encode(key).decode("ascii")
    key_path.write_text(key_b64)

    if platform.system() != "Windows":
        os.chmod(key_path, 0o600)

    logger.info(f"  Encryption key saved to {key_path.name}")


def load_encryption_key(key_path: Path) -> bytes | None:
    """Charge la clé de chiffrement."""
    import base64

    if not key_path.exists():
        logger.warning(f"Encryption key not found: {key_path}")
        return None

    key_b64 = key_path.read_text().strip()
    return base64.b64decode(key_b64)


def encrypt_model_file(
    input_path: Path,
    output_path: Path | None = None,
    encryption_key: bytes | None = None,
) -> tuple[Path, bytes]:
    """Chiffre un fichier modèle avec AES-256-GCM.

    Args:
    ----
        input_path: Chemin du fichier à chiffrer
        output_path: Chemin de sortie (défaut: input_path + .enc)
        encryption_key: Clé AES-256 (génère une nouvelle si None)

    Returns:
    -------
        (chemin_fichier_chiffré, clé_utilisée)
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if encryption_key is None:
        encryption_key = generate_encryption_key()

    if output_path is None:
        output_path = input_path.with_suffix(input_path.suffix + ENCRYPTED_EXTENSION)

    plaintext = input_path.read_bytes()
    nonce = secrets.token_bytes(12)

    aesgcm = AESGCM(encryption_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    encrypted_data = nonce + ciphertext
    output_path.write_bytes(encrypted_data)

    logger.info(
        f"  Encrypted {input_path.name} -> {output_path.name} "
        f"({len(plaintext):,} -> {len(encrypted_data):,} bytes)"
    )

    return output_path, encryption_key


def decrypt_model_file(
    encrypted_path: Path,
    output_path: Path | None = None,
    encryption_key: bytes | None = None,
    key_path: Path | None = None,
) -> Path | None:
    """Déchiffre un fichier modèle chiffré avec AES-256-GCM.

    Args:
    ----
        encrypted_path: Chemin du fichier chiffré
        output_path: Chemin de sortie (défaut: enlève .enc)
        encryption_key: Clé AES-256
        key_path: Chemin de la clé (alternatif à encryption_key)

    Returns:
    -------
        Chemin du fichier déchiffré ou None si échec
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if encryption_key is None and key_path is not None:
        encryption_key = load_encryption_key(key_path)

    if encryption_key is None:
        logger.error("No encryption key provided")
        return None

    if not encrypted_path.exists():
        logger.error(f"Encrypted file not found: {encrypted_path}")
        return None

    if output_path is None:
        if encrypted_path.suffix == ENCRYPTED_EXTENSION:
            output_path = encrypted_path.with_suffix("")
        else:
            output_path = encrypted_path.with_suffix(".decrypted")

    try:
        encrypted_data = encrypted_path.read_bytes()
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]

        aesgcm = AESGCM(encryption_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        output_path.write_bytes(plaintext)

        logger.info(
            f"  Decrypted {encrypted_path.name} -> {output_path.name} "
            f"({len(encrypted_data):,} -> {len(plaintext):,} bytes)"
        )

        return output_path

    except Exception as e:
        logger.error(f"Decryption failed for {encrypted_path.name}: {e}")
        return None


def encrypt_model_directory(
    version_dir: Path,
    encryption_key: bytes | None = None,
    *,
    delete_originals: bool = False,
    save_key_to_file: bool = False,
) -> tuple[list[Path], bytes]:
    """Chiffre tous les fichiers modèle d'un répertoire version.

    Args:
    ----
        version_dir: Répertoire contenant les modèles
        encryption_key: Clé AES-256 (génère une nouvelle si None)
        delete_originals: Supprimer les fichiers originaux après chiffrement
        save_key_to_file: Sauvegarder la clé dans le répertoire (NON RECOMMANDÉ)

    Returns:
    -------
        (liste_fichiers_chiffrés, clé_utilisée)

    Warning:
    -------
        ISO 27001 - La clé NE DOIT PAS être stockée avec les données chiffrées.
    """
    if encryption_key is None:
        encryption_key = get_key_from_env(ENV_ENCRYPTION_KEY)

    if encryption_key is None:
        encryption_key = generate_encryption_key()
        logger.warning(
            "Generated new encryption key. Store it securely in ALICE_ENCRYPTION_KEY env var!"
        )

    encrypted_files: list[Path] = []
    model_extensions = {".cbm", ".ubj", ".txt", ".joblib", ".onnx"}

    for file_path in version_dir.iterdir():
        if file_path.suffix in model_extensions and not file_path.name.endswith(
            ENCRYPTED_EXTENSION
        ):
            enc_path, _ = encrypt_model_file(file_path, encryption_key=encryption_key)
            encrypted_files.append(enc_path)

            if delete_originals:
                file_path.unlink()
                logger.info(f"  Deleted original: {file_path.name}")

    if save_key_to_file:
        # ISO 27001: Interdit en production - clés ne doivent JAMAIS être stockées avec données
        env_debug = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")
        if not env_debug:
            raise RuntimeError(
                "ISO 27001 VIOLATION: save_key_to_file=True is forbidden in production. "
                "Store encryption keys in ALICE_ENCRYPTION_KEY environment variable."
            )
        key_path = version_dir / "encryption.key"
        save_encryption_key(encryption_key, key_path)
        logger.warning(
            f"  Key saved to {key_path.name} - DEBUG MODE ONLY, forbidden in production!"
        )

    logger.info(f"  Encrypted {len(encrypted_files)} model files in {version_dir.name}")

    return encrypted_files, encryption_key


def decrypt_model_directory(
    version_dir: Path,
    encryption_key: bytes | None = None,
    *,
    delete_encrypted: bool = False,
) -> list[Path]:
    """Déchiffre tous les fichiers modèle chiffrés d'un répertoire.

    Args:
    ----
        version_dir: Répertoire contenant les modèles chiffrés
        encryption_key: Clé AES-256 (priorité: param > env var > fichier local)
        delete_encrypted: Supprimer les fichiers chiffrés après déchiffrement

    Returns:
    -------
        Liste des fichiers déchiffrés
    """
    if encryption_key is None:
        encryption_key = get_key_from_env(ENV_ENCRYPTION_KEY)

    if encryption_key is None:
        key_path = version_dir / "encryption.key"
        if key_path.exists():
            encryption_key = load_encryption_key(key_path)
            logger.warning("Loaded key from file - consider migrating to env var")

    if encryption_key is None:
        logger.error("No encryption key available (set ALICE_ENCRYPTION_KEY env var)")
        return []

    decrypted_files: list[Path] = []

    for file_path in version_dir.iterdir():
        if file_path.suffix == ENCRYPTED_EXTENSION:
            dec_path = decrypt_model_file(file_path, encryption_key=encryption_key)
            if dec_path:
                decrypted_files.append(dec_path)

                if delete_encrypted:
                    file_path.unlink()
                    logger.info(f"  Deleted encrypted: {file_path.name}")

    logger.info(f"  Decrypted {len(decrypted_files)} model files in {version_dir.name}")

    return decrypted_files
