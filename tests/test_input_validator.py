"""Tests Input Validator - ISO 24029.

Document ID: ALICE-TEST-INPUT-VAL
Version: 1.1.0
Tests: 17

Classes:
- TestCheckCategorical: Tests helper FeatureBounds._check_categorical (2 tests)
- TestCheckNumeric: Tests helper FeatureBounds._check_numeric (3 tests)
- TestRecordSeverity: Tests helper _record_severity (2 tests)
- TestDetermineAction: Tests helper _determine_action (2 tests)
- TestComputeFeatureBounds: Tests création bounds (2 tests)
- TestValidateInput: Tests validation single (3 tests)
- TestValidateBatch: Tests validation batch (2 tests)
- TestBoundsConfigIO: Tests persistance (1 test)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<180 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import tempfile
from pathlib import Path

import pandas as pd

from scripts.model_registry.input_types import (
    FeatureBounds,
    InputValidationResult,
    OODAction,
    OODSeverity,
)
from scripts.model_registry.input_validator import (
    _determine_action,
    _record_severity,
    compute_feature_bounds,
    create_bounds_config,
    load_bounds_config,
    save_bounds_config,
    validate_batch,
    validate_input,
)

# Fixtures (bounds_config, ood_input, training_data, valid_input) are auto-loaded via pytest_plugins


class TestCheckCategorical:
    """Tests pour FeatureBounds._check_categorical (ISO 29119 - Unit tests)."""

    def test_valid_category(self):
        """Catégorie valide retourne VALID."""
        bounds = FeatureBounds(
            feature_name="cat",
            min_value=0,
            max_value=0,
            mean=0,
            std=0,
            p01=0,
            p99=0,
            n_samples=100,
            is_categorical=True,
            categories=["A", "B"],
        )
        severity, msg = bounds._check_categorical("A")
        assert severity == OODSeverity.VALID
        assert msg == ""

    def test_unknown_category(self):
        """Catégorie inconnue retourne OUT_OF_BOUNDS."""
        bounds = FeatureBounds(
            feature_name="cat",
            min_value=0,
            max_value=0,
            mean=0,
            std=0,
            p01=0,
            p99=0,
            n_samples=100,
            is_categorical=True,
            categories=["A", "B"],
        )
        severity, msg = bounds._check_categorical("X")
        assert severity == OODSeverity.OUT_OF_BOUNDS
        assert "Unknown category" in msg


class TestCheckNumeric:
    """Tests pour FeatureBounds._check_numeric (ISO 29119 - Unit tests)."""

    def test_value_in_bounds(self):
        """Valeur dans bornes retourne VALID."""
        bounds = FeatureBounds(
            feature_name="num",
            min_value=0,
            max_value=100,
            mean=50,
            std=10,
            p01=5,
            p99=95,
            n_samples=100,
            is_categorical=False,
        )
        severity, msg = bounds._check_numeric(50.0, std_tolerance=4.0)
        assert severity == OODSeverity.VALID

    def test_value_outside_percentiles(self):
        """Valeur hors p01-p99 retourne WARNING."""
        bounds = FeatureBounds(
            feature_name="num",
            min_value=0,
            max_value=100,
            mean=50,
            std=10,
            p01=10,
            p99=90,
            n_samples=100,
            is_categorical=False,
        )
        severity, msg = bounds._check_numeric(95.0, std_tolerance=4.0)
        assert severity == OODSeverity.WARNING

    def test_value_outside_bounds(self):
        """Valeur hors min/max retourne OUT_OF_BOUNDS ou EXTREME."""
        bounds = FeatureBounds(
            feature_name="num",
            min_value=0,
            max_value=100,
            mean=50,
            std=10,
            p01=5,
            p99=95,
            n_samples=100,
            is_categorical=False,
        )
        severity, msg = bounds._check_numeric(150.0, std_tolerance=4.0)
        assert severity in (OODSeverity.OUT_OF_BOUNDS, OODSeverity.EXTREME)


class TestRecordSeverity:
    """Tests pour _record_severity (ISO 29119 - Unit tests)."""

    def test_warning_appended(self):
        """WARNING ajouté à warnings."""
        result = InputValidationResult(is_valid=True, action=OODAction.ACCEPT)
        _record_severity(result, "feature_a", OODSeverity.WARNING, "test msg")
        assert len(result.warnings) == 1
        assert "feature_a" in result.warnings[0]

    def test_ood_appended(self):
        """OUT_OF_BOUNDS ajouté à ood_features et errors."""
        result = InputValidationResult(is_valid=True, action=OODAction.ACCEPT)
        _record_severity(result, "feature_b", OODSeverity.OUT_OF_BOUNDS, "out msg")
        assert "feature_b" in result.ood_features
        assert len(result.errors) == 1


class TestDetermineAction:
    """Tests pour _determine_action (ISO 29119 - Unit tests)."""

    def test_reject_when_ood_ratio_high(self):
        """REJECT si ood_ratio >= threshold."""
        result = InputValidationResult(is_valid=True, action=OODAction.ACCEPT)
        result.ood_ratio = 0.5
        _determine_action(result, rejection_threshold=0.3)
        assert result.action == OODAction.REJECT
        assert result.is_valid is False

    def test_warn_when_ood_features(self):
        """WARN si ood_features présents mais ratio OK."""
        result = InputValidationResult(is_valid=True, action=OODAction.ACCEPT)
        result.ood_ratio = 0.1
        result.ood_features = ["feature_x"]
        _determine_action(result, rejection_threshold=0.3)
        assert result.action == OODAction.WARN


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
