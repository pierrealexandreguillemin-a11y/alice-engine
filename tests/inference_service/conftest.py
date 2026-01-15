"""Fixtures Inference Service - ISO 29119.

Document ID: ALICE-TEST-INFERENCE-SERVICE-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pytest

from services.inference import InferenceService


@pytest.fixture
def service() -> InferenceService:
    """Fixture service sans modèle."""
    return InferenceService()


@pytest.fixture
def service_with_path() -> InferenceService:
    """Fixture service avec chemin modèle (fictif)."""
    return InferenceService(model_path="/fake/model/path.cbm")


@pytest.fixture
def sample_players() -> list[dict]:
    """Fixture liste joueurs pour tests."""
    return [
        {"ffe_id": "A12345", "name": "Carlsen Magnus", "elo": 2850},
        {"ffe_id": "B23456", "name": "Caruana Fabiano", "elo": 2800},
        {"ffe_id": "C34567", "name": "Ding Liren", "elo": 2780},
        {"ffe_id": "D45678", "name": "Nepomniachtchi Ian", "elo": 2760},
        {"ffe_id": "E56789", "name": "Firouzja Alireza", "elo": 2750},
        {"ffe_id": "F67890", "name": "Nakamura Hikaru", "elo": 2740},
        {"ffe_id": "G78901", "name": "Aronian Levon", "elo": 2720},
        {"ffe_id": "H89012", "name": "Giri Anish", "elo": 2700},
        {"ffe_id": "I90123", "name": "So Wesley", "elo": 2680},
        {"ffe_id": "J01234", "name": "Mamedyarov Shakhriyar", "elo": 2660},
    ]
