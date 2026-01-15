"""Tests Signature - ISO 29119.

Document ID: ALICE-TEST-MODEL-SIGNATURE
Version: 1.0.0
Tests: 1 classes

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

from scripts.model_registry import (
    compute_model_signature,
    generate_signing_key,
    verify_model_signature,
)


class TestModelSignature:
    """Tests pour signature HMAC-SHA256 (ISO 27001)."""

    def test_generate_signing_key(self) -> None:
        """Test génération clé de signature."""
        key = generate_signing_key()

        assert len(key) == 64  # 32 bytes hex
        assert all(c in "0123456789abcdef" for c in key)

    def test_compute_signature(self, tmp_path: Path) -> None:
        """Test calcul signature."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        key = generate_signing_key()

        signature = compute_model_signature(model_file, key)

        assert len(signature) == 64  # SHA-256 hex

    def test_signature_deterministic(self, tmp_path: Path) -> None:
        """Test signature déterministe pour même fichier et clé."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        key = generate_signing_key()

        sig1 = compute_model_signature(model_file, key)
        sig2 = compute_model_signature(model_file, key)

        assert sig1 == sig2

    def test_different_key_different_signature(self, tmp_path: Path) -> None:
        """Test clés différentes produisent signatures différentes."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        key1 = generate_signing_key()
        key2 = generate_signing_key()

        sig1 = compute_model_signature(model_file, key1)
        sig2 = compute_model_signature(model_file, key2)

        assert sig1 != sig2

    def test_verify_valid_signature(self, tmp_path: Path) -> None:
        """Test vérification signature valide."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        key = generate_signing_key()
        signature = compute_model_signature(model_file, key)

        result = verify_model_signature(model_file, signature, key)

        assert result is True

    def test_verify_invalid_signature(self, tmp_path: Path) -> None:
        """Test vérification signature invalide."""
        model_file = tmp_path / "model.cbm"
        model_file.write_bytes(b"model binary data")
        key = generate_signing_key()

        result = verify_model_signature(model_file, "invalid" + "0" * 56, key)

        assert result is False

    def test_verify_missing_file(self, tmp_path: Path) -> None:
        """Test vérification fichier manquant."""
        key = generate_signing_key()

        result = verify_model_signature(tmp_path / "nonexistent.cbm", "a" * 64, key)

        assert result is False
