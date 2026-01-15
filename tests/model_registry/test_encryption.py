"""Tests Encryption - ISO 29119.

Document ID: ALICE-TEST-MODEL-ENCRYPTION
Version: 1.0.0
Tests: 1 classes

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

import pytest

from scripts.model_registry import (
    ENCRYPTED_EXTENSION,
    ENV_ENCRYPTION_KEY,
    decrypt_model_directory,
    decrypt_model_file,
    encrypt_model_directory,
    encrypt_model_file,
    generate_encryption_key,
    load_encryption_key,
    save_encryption_key,
)


class TestAES256Encryption:
    """Tests pour chiffrement AES-256-GCM."""

    def test_generate_encryption_key(self) -> None:
        """Test génération clé AES-256."""
        key = generate_encryption_key()

        assert len(key) == 32  # 256 bits
        assert isinstance(key, bytes)

    def test_generate_unique_keys(self) -> None:
        """Test clés générées sont uniques."""
        key1 = generate_encryption_key()
        key2 = generate_encryption_key()

        assert key1 != key2

    def test_save_load_encryption_key(self, tmp_path: Path) -> None:
        """Test sauvegarde et chargement clé."""
        key = generate_encryption_key()
        key_path = tmp_path / "test.key"

        save_encryption_key(key, key_path)
        loaded_key = load_encryption_key(key_path)

        assert loaded_key == key

    def test_load_missing_key(self, tmp_path: Path) -> None:
        """Test chargement clé inexistante."""
        key_path = tmp_path / "nonexistent.key"

        loaded_key = load_encryption_key(key_path)

        assert loaded_key is None

    def test_encrypt_decrypt_file(self, tmp_path: Path) -> None:
        """Test chiffrement/déchiffrement fichier."""
        # Créer fichier original
        original_file = tmp_path / "model.cbm"
        original_data = b"Model binary data for testing AES-256 encryption"
        original_file.write_bytes(original_data)

        # Chiffrer
        encrypted_path, key = encrypt_model_file(original_file)

        assert encrypted_path.exists()
        assert encrypted_path.suffix == ENCRYPTED_EXTENSION
        assert encrypted_path.read_bytes() != original_data  # Chiffré

        # Déchiffrer
        decrypted_path = decrypt_model_file(encrypted_path, encryption_key=key)

        assert decrypted_path is not None
        assert decrypted_path.read_bytes() == original_data

    def test_encrypt_with_provided_key(self, tmp_path: Path) -> None:
        """Test chiffrement avec clé fournie."""
        original_file = tmp_path / "model.cbm"
        original_file.write_bytes(b"Test data")
        key = generate_encryption_key()

        encrypted_path, returned_key = encrypt_model_file(original_file, encryption_key=key)

        assert returned_key == key
        assert encrypted_path.exists()

    def test_decrypt_with_wrong_key(self, tmp_path: Path) -> None:
        """Test déchiffrement avec mauvaise clé échoue."""
        original_file = tmp_path / "model.cbm"
        original_file.write_bytes(b"Test data")

        encrypted_path, _ = encrypt_model_file(original_file)
        wrong_key = generate_encryption_key()

        decrypted_path = decrypt_model_file(encrypted_path, encryption_key=wrong_key)

        assert decrypted_path is None  # Échec attendu

    def test_decrypt_missing_file(self, tmp_path: Path) -> None:
        """Test déchiffrement fichier inexistant."""
        key = generate_encryption_key()

        result = decrypt_model_file(tmp_path / "nonexistent.enc", encryption_key=key)

        assert result is None

    def test_encrypt_directory(self, tmp_path: Path) -> None:
        """Test chiffrement répertoire complet."""
        version_dir = tmp_path / "v20240101"
        version_dir.mkdir()

        # Créer plusieurs fichiers modèle
        (version_dir / "catboost.cbm").write_bytes(b"CatBoost model")
        (version_dir / "xgboost.ubj").write_bytes(b"XGBoost model")
        (version_dir / "encoders.joblib").write_bytes(b"Label encoders")
        (version_dir / "metadata.json").write_text("{}")  # Ne doit pas être chiffré

        encrypted_files, key = encrypt_model_directory(version_dir)

        assert len(encrypted_files) == 3  # .cbm, .ubj, .joblib
        # Clé NON sauvegardée par défaut (ISO 27001)
        assert not (version_dir / "encryption.key").exists()
        # Originaux toujours présents (delete_originals=False par défaut)
        assert (version_dir / "catboost.cbm").exists()

    def test_encrypt_directory_delete_originals(self, tmp_path: Path) -> None:
        """Test chiffrement avec suppression originaux."""
        version_dir = tmp_path / "v20240101"
        version_dir.mkdir()

        (version_dir / "model.cbm").write_bytes(b"Model data")
        key = generate_encryption_key()

        encrypt_model_directory(version_dir, encryption_key=key, delete_originals=True)

        assert not (version_dir / "model.cbm").exists()  # Supprimé
        assert (version_dir / "model.cbm.enc").exists()  # Chiffré présent

    def test_decrypt_directory(self, tmp_path: Path) -> None:
        """Test déchiffrement répertoire complet."""
        version_dir = tmp_path / "v20240101"
        version_dir.mkdir()

        # Créer et chiffrer avec clé explicite
        original_data = b"Original model data"
        (version_dir / "model.cbm").write_bytes(original_data)
        key = generate_encryption_key()
        encrypt_model_directory(version_dir, encryption_key=key, delete_originals=True)

        # Déchiffrer avec la même clé
        decrypted_files = decrypt_model_directory(version_dir, encryption_key=key)

        assert len(decrypted_files) == 1
        assert (version_dir / "model.cbm").exists()
        assert (version_dir / "model.cbm").read_bytes() == original_data

    def test_roundtrip_encryption(self, tmp_path: Path) -> None:
        """Test cycle complet chiffrement/déchiffrement."""
        version_dir = tmp_path / "v20240101"
        version_dir.mkdir()

        # Données originales
        models = {
            "catboost.cbm": b"CatBoost binary model data " * 100,
            "xgboost.ubj": b"XGBoost universal binary " * 100,
            "lightgbm.txt": b"LightGBM text model " * 100,
        }

        for name, data in models.items():
            (version_dir / name).write_bytes(data)

        # Chiffrer et supprimer originaux (avec clé explicite)
        key = generate_encryption_key()
        encrypt_model_directory(version_dir, encryption_key=key, delete_originals=True)

        # Vérifier originaux supprimés
        for name in models:
            assert not (version_dir / name).exists()

        # Déchiffrer avec la même clé
        decrypt_model_directory(version_dir, encryption_key=key, delete_encrypted=True)

        # Vérifier données restaurées
        for name, original_data in models.items():
            assert (version_dir / name).exists()
            assert (version_dir / name).read_bytes() == original_data

    def test_encrypt_directory_with_env_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test chiffrement avec clé depuis variable d'environnement."""
        import base64

        version_dir = tmp_path / "v20240101"
        version_dir.mkdir()
        (version_dir / "model.cbm").write_bytes(b"Model data")

        # Définir la clé en env var
        key = generate_encryption_key()
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, base64.b64encode(key).decode())

        # Chiffrer sans fournir de clé (doit utiliser env var)
        encrypted_files, returned_key = encrypt_model_directory(version_dir)

        assert len(encrypted_files) == 1
        assert returned_key == key  # Doit avoir utilisé la clé de l'env var


# ==============================================================================
# P3 TESTS - Drift Monitoring (ISO 5259 / ISO 42001)
# ==============================================================================
