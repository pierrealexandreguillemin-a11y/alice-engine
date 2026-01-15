"""Tests Explainability - ISO 42001.

Document ID: ALICE-TEST-EXPLAIN
Version: 1.0.0
Tests: 8

Classes:
- TestBuildNormalizedImportance: Tests normalisation (3 tests)
- TestGetTopFeatures: Tests top K (2 tests)
- TestComputePermutationImportance: Tests permutation (2 tests)
- TestExplainPrediction: Tests prediction (1 test)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management (Explicabilité)
- ISO/IEC 5055:2021 - Code Quality (<120 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from unittest.mock import MagicMock, patch

import numpy as np

from scripts.model_registry.explainability import (
    _build_normalized_importance,
    get_top_features,
)

# Fixtures loaded via pytest_plugins in conftest.py


class TestBuildNormalizedImportance:
    """Tests pour _build_normalized_importance."""

    def test_normalization(self):
        """Vérifie que les valeurs somment à 1."""
        values = np.array([0.3, 0.5, 0.2])
        features = ["a", "b", "c"]
        result = _build_normalized_importance(values, features)

        assert abs(sum(result.values()) - 1.0) < 1e-6
        assert len(result) == 3

    def test_sorted_descending(self):
        """Vérifie tri décroissant."""
        values = np.array([0.1, 0.5, 0.3])
        features = ["low", "high", "mid"]
        result = _build_normalized_importance(values, features)

        keys = list(result.keys())
        assert keys[0] == "high"
        assert keys[-1] == "low"

    def test_feature_mismatch(self):
        """Gère le décalage features/values."""
        values = np.array([0.5, 0.5])
        features = ["a", "b", "c"]  # Plus de features que values
        result = _build_normalized_importance(values, features)

        # Should create generic names
        assert len(result) == 2


class TestGetTopFeatures:
    """Tests pour get_top_features."""

    def test_top_k(self):
        """Retourne top K features."""
        importance = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
        result = get_top_features(importance, top_k=2)

        assert len(result) == 2
        assert "a" in result
        assert "b" in result
        assert "d" not in result

    def test_top_k_exceeds(self):
        """Top K supérieur au nombre de features."""
        importance = {"a": 0.6, "b": 0.4}
        result = get_top_features(importance, top_k=10)

        assert len(result) == 2


class TestComputePermutationImportance:
    """Tests pour compute_permutation_importance."""

    def test_permutation_with_mock(self, sample_X_train, sample_y_train):  # noqa: N803
        """Test permutation importance avec mock sklearn."""
        from scripts.model_registry.explainability import compute_permutation_importance

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=sample_y_train.values)

        with patch("sklearn.inspection.permutation_importance") as mock_perm:
            mock_perm.return_value = MagicMock(importances_mean=np.array([0.3, 0.5]))
            result = compute_permutation_importance(
                mock_model,
                sample_X_train[["feature_a", "feature_b"]],
                sample_y_train.values,
                ["feature_a", "feature_b"],
            )

        assert isinstance(result, dict)

    def test_permutation_failure_returns_empty(self, sample_X_train, sample_y_train):  # noqa: N803
        """Retourne dict vide en cas d'erreur."""
        from scripts.model_registry.explainability import compute_permutation_importance

        mock_model = MagicMock()
        mock_model.predict = MagicMock(side_effect=Exception("Test error"))

        result = compute_permutation_importance(
            mock_model,
            sample_X_train,
            sample_y_train.values,
            ["a", "b"],
        )
        assert result == {}


class TestExplainPrediction:
    """Tests pour explain_prediction."""

    def test_explain_returns_dict(self):
        """Retourne dict feature -> contribution."""
        from scripts.model_registry.explainability import explain_prediction

        mock_model = MagicMock()
        X_instance = np.array([1.0, 2.0, 3.0])

        # Without SHAP installed, should return empty or handle gracefully
        result = explain_prediction(mock_model, X_instance, ["a", "b", "c"])
        assert isinstance(result, dict)
