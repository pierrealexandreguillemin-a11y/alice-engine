"""Sécurité pour Model Registry - ISO 27001.

Ce module est un wrapper qui réexporte les fonctions de sécurité:
- Signature HMAC-SHA256 (security_signing)
- Chiffrement AES-256-GCM (security_encryption)

Conformité ISO/IEC 27001, 27034.
"""

# Signing
# Encryption
from scripts.model_registry.security_encryption import (
    ENCRYPTED_EXTENSION,
    ENV_ENCRYPTION_KEY,
    decrypt_model_directory,
    decrypt_model_file,
    encrypt_model_directory,
    encrypt_model_file,
    generate_encryption_key,
    get_key_from_env,
    load_encryption_key,
    save_encryption_key,
)
from scripts.model_registry.security_signing import (
    ENV_SIGNING_KEY,
    compute_model_signature,
    generate_signing_key,
    get_signing_key_from_env,
    load_signing_key,
    save_signing_key,
    verify_model_signature,
)

__all__ = [
    # Constants
    "ENCRYPTED_EXTENSION",
    "ENV_ENCRYPTION_KEY",
    "ENV_SIGNING_KEY",
    # Signing
    "generate_signing_key",
    "compute_model_signature",
    "verify_model_signature",
    "save_signing_key",
    "load_signing_key",
    "get_signing_key_from_env",
    # Encryption
    "get_key_from_env",
    "generate_encryption_key",
    "save_encryption_key",
    "load_encryption_key",
    "encrypt_model_file",
    "decrypt_model_file",
    "encrypt_model_directory",
    "decrypt_model_directory",
]
