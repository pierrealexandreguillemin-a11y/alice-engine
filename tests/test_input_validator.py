"""Tests Input Validator - ISO 24029.

Document ID: ALICE-TEST-INPUT-VAL
Version: 1.0.0
Tests: 8

Classes:
- TestComputeFeatureBounds: Tests création bounds (2 tests)
- TestValidateInput: Tests validation single (3 tests)
- TestValidateBatch: Tests validation batch (2 tests)
- TestBoundsConfigIO: Tests persistance (1 test)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<120 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import tempfile
from pathlib import Path

import pandas as pd

from scripts.model_registry.input_types import OODAction
from scripts.model_registry.input_validator import (
    compute_feature_bounds,
    create_bounds_config,
    load_bounds_config,
    save_bounds_config,
    validate_batch,
    validate_input,
)

# Fixtures (bounds_config, ood_input, training_data, valid_input) are auto-loaded via pytest_plugins


class TestComputeFeatureBounds:
    """Tests pour compute_feature_bounds."""

    def test_numeric_bounds(self, training_data):
        """Calcule bornes feature numérique."""
        bounds = compute_feature_bounds(training_data, "age")
        assert bounds.feature_name == "age"
        assert bounds.min_value >= 18
        assert bounds.max_value <= 80
        assert bounds.is_categorical is False

    def test_categorical_bounds(self, training_data):
        """Calcule bornes feature catégorielle."""
        bounds = compute_feature_bounds(training_data, "category", is_categorical=True)
        assert bounds.is_categorical is True
        assert bounds.categories is not None
        assert set(bounds.categories) == {"A", "B", "C"}


class TestValidateInput:
    """Tests pour validate_input."""

    def test_valid_input(self, bounds_config, valid_input):
        """Input valide accepté."""
        result = validate_input(valid_input, bounds_config)
        assert result.is_valid is True
        assert result.action == OODAction.ACCEPT

    def test_ood_input(self, bounds_config, ood_input):
        """Input OOD détecté."""
        result = validate_input(ood_input, bounds_config)
        assert len(result.ood_features) > 0
        assert result.action in (OODAction.WARN, OODAction.REJECT)

    def test_partial_input(self, bounds_config):
        """Input avec features manquantes."""
        partial = {"income": 50000}  # category missing
        result = validate_input(partial, bounds_config)
        assert result.is_valid is True  # Only validated features count


class TestValidateBatch:
    """Tests pour validate_batch."""

    def test_batch_validation(self, bounds_config):
        """Valide batch d'inputs."""
        batch = pd.DataFrame(
            [
                {"income": 50000, "category": "A"},
                {"income": 200000, "category": "B"},  # OOD
                {"income": 45000, "category": "C"},
            ]
        )
        results = validate_batch(batch, bounds_config)
        assert len(results) == 3
        assert results[0].is_valid is True
        assert results[2].is_valid is True

    def test_batch_empty(self, bounds_config):
        """Batch vide retourne liste vide."""
        batch = pd.DataFrame([])
        results = validate_batch(batch, bounds_config)
        assert len(results) == 0


class TestBoundsConfigIO:
    """Tests pour persistance config."""

    def test_save_load_config(self, training_data):
        """Sauvegarde et chargement config."""
        config = create_bounds_config(
            training_data,
            model_version="v1.0",
            categorical_features=["category"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bounds.json"
            save_bounds_config(config, path)
            loaded = load_bounds_config(path)

            assert loaded is not None
            assert loaded.model_version == "v1.0"
            assert len(loaded.features) == len(config.features)
