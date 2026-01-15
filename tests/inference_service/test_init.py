"""Tests Inference Service Init and Load - ISO 29119.

Document ID: ALICE-TEST-INFERENCE-SERVICE-INIT
Version: 1.0.0

Tests pour l'initialisation et le chargement du modèle.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from services.inference import InferenceService


class TestInferenceServiceInit:
    """Tests initialisation InferenceService."""

    def test_init_without_model_path(self) -> None:
        """Test initialisation sans chemin modèle."""
        service = InferenceService()

        assert service.model is None
        assert service.model_path is None
        assert not service.is_loaded

    def test_init_with_model_path(self) -> None:
        """Test initialisation avec chemin modèle."""
        service = InferenceService(model_path="/path/to/model.cbm")

        assert service.model is None
        assert service.model_path == "/path/to/model.cbm"
        assert not service.is_loaded


class TestLoadModel:
    """Tests pour load_model()."""

    def test_load_model_no_path_returns_false(self, service: InferenceService) -> None:
        """Test load_model sans chemin retourne False."""
        result = service.load_model()

        assert result is False
        assert not service.is_loaded

    def test_load_model_stub_sets_is_loaded(self, service_with_path: InferenceService) -> None:
        """Test load_model stub (implementation TODO).

        ATTENTION: L'implémentation actuelle est un stub qui:
        - Ne charge pas vraiment le modèle
        - Ne vérifie pas l'existence du fichier
        - Retourne toujours True si model_path est défini

        Ce test documente ce comportement temporaire.
        """
        result = service_with_path.load_model()

        assert result is True
        assert service_with_path.is_loaded
        assert service_with_path.model is None
