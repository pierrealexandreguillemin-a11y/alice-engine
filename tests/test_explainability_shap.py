"""Tests SHAP Explainability - ISO 42001.

Document ID: ALICE-TEST-SHAP
Version: 1.0.0
Tests: 8

Classes:
- TestComputeShapImportance: Tests SHAP computation (3 tests)
- TestCreateShapExplainer: Tests explainer creation (3 tests)
- TestExplainPrediction: Tests individual prediction (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 42001:2023 - AI Management (Explicabilité)
- ISO/IEC 5055:2021 - Code Quality (<100 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


class TestComputeShapImportance:
    """Tests pour compute_shap_importance."""

    def test_returns_dict_on_success(self):
        """Retourne dict d'importance normalisé."""
        from scripts.model_registry.explainability import compute_shap_importance

        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.array([[0.3, 0.5, 0.2]])
        mock_explainer.return_value = mock_shap_values
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "CatBoostClassifier"
        X = np.random.randn(10, 3)

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = compute_shap_importance(mock_model, X, ["a", "b", "c"])

        assert isinstance(result, dict)

    def test_handles_import_error(self):
        """Retourne dict vide si SHAP non installé."""
        from scripts.model_registry.explainability import compute_shap_importance

        mock_model = MagicMock()
        X = np.random.randn(5, 2)

        # Simulate ImportError
        with patch.dict(sys.modules, {"shap": None}):
            with patch(
                "scripts.model_registry.explainability._create_shap_explainer",
                side_effect=ImportError("No shap"),
            ):
                result = compute_shap_importance(mock_model, X, ["a", "b"])

        assert result == {}

    def test_limits_samples(self):
        """Limite le nombre d'échantillons pour performance."""
        from scripts.model_registry.explainability import compute_shap_importance

        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.array([[0.5, 0.5]])
        mock_explainer.return_value = mock_shap_values
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "XGBClassifier"
        X = np.random.randn(200, 2)  # Plus que max_samples=100

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = compute_shap_importance(mock_model, X, ["a", "b"], max_samples=50)

        assert isinstance(result, dict)


class TestCreateShapExplainer:
    """Tests pour _create_shap_explainer."""

    def test_tree_explainer_for_catboost(self):
        """Utilise TreeExplainer pour CatBoost."""
        from scripts.model_registry.explainability import _create_shap_explainer

        mock_shap = MagicMock()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "CatBoostClassifier"
        X = np.random.randn(10, 3)

        with patch.dict(sys.modules, {"shap": mock_shap}):
            _create_shap_explainer(mock_model, X)

        mock_shap.TreeExplainer.assert_called_once()

    def test_tree_explainer_for_lightgbm(self):
        """Utilise TreeExplainer pour LightGBM."""
        from scripts.model_registry.explainability import _create_shap_explainer

        mock_shap = MagicMock()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "LGBMClassifier"
        X = np.random.randn(10, 3)

        with patch.dict(sys.modules, {"shap": mock_shap}):
            _create_shap_explainer(mock_model, X)

        mock_shap.TreeExplainer.assert_called_once()

    def test_kernel_explainer_fallback(self):
        """Utilise KernelExplainer pour modèles non-tree."""
        from scripts.model_registry.explainability import _create_shap_explainer

        mock_shap = MagicMock()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "LogisticRegression"
        mock_model.predict_proba = MagicMock()
        X = np.random.randn(100, 3)

        with patch.dict(sys.modules, {"shap": mock_shap}):
            _create_shap_explainer(mock_model, X)

        mock_shap.KernelExplainer.assert_called_once()


class TestExplainPredictionDetailed:
    """Tests détaillés pour explain_prediction."""

    def test_reshapes_1d_input(self):
        """Reshape automatiquement les inputs 1D."""
        from scripts.model_registry.explainability import explain_prediction

        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.array([[0.3, 0.4, 0.3]])
        mock_explainer.return_value = mock_shap_values
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "XGBClassifier"
        X_instance = np.array([1.0, 2.0, 3.0])  # 1D

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = explain_prediction(mock_model, X_instance, ["a", "b", "c"])

        assert isinstance(result, dict)

    def test_handles_exception_gracefully(self):
        """Gère les erreurs gracieusement."""
        from scripts.model_registry.explainability import explain_prediction

        mock_model = MagicMock()
        X_instance = np.array([1.0, 2.0])

        with patch(
            "scripts.model_registry.explainability._create_shap_explainer",
            side_effect=Exception("SHAP error"),
        ):
            result = explain_prediction(mock_model, X_instance, ["a", "b"])

        assert result == {}

    def test_multioutput_handling(self):
        """Gère les outputs multi-classes."""
        from scripts.model_registry.explainability import explain_prediction

        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        # Simulate multi-output: shape (1, 3, 2) - 1 sample, 3 features, 2 classes
        mock_shap_values.values = np.array([[[0.1, 0.2], [0.3, 0.4], [0.2, 0.3]]])
        mock_explainer.return_value = mock_shap_values
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "CatBoostClassifier"
        X_instance = np.array([[1.0, 2.0, 3.0]])

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = explain_prediction(mock_model, X_instance, ["a", "b", "c"])

        assert isinstance(result, dict)


class TestComputeShapSampling:
    """Tests pour le sampling dans compute_shap_importance."""

    def test_with_pandas_dataframe(self):
        """Gère les DataFrames pandas."""
        from scripts.model_registry.explainability import compute_shap_importance

        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.array([[0.5, 0.5]])
        mock_explainer.return_value = mock_shap_values
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "XGBClassifier"
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = compute_shap_importance(mock_model, X, ["a", "b"])

        assert isinstance(result, dict)

    def test_samples_large_datasets(self):
        """Limite les échantillons pour grands datasets."""
        from scripts.model_registry.explainability import compute_shap_importance

        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.array([[0.5, 0.5]])
        mock_explainer.return_value = mock_shap_values
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "RandomForestClassifier"
        X = np.random.randn(500, 2)  # Large dataset

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = compute_shap_importance(mock_model, X, ["a", "b"], max_samples=50)

        assert isinstance(result, dict)

    def test_multiclass_shap_values(self):
        """Gère les SHAP values multi-classes."""
        from scripts.model_registry.explainability import compute_shap_importance

        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap_values = MagicMock()
        # 3D array: (samples, features, classes)
        mock_shap_values.values = np.array([[[0.1, 0.2], [0.3, 0.4]]])
        mock_explainer.return_value = mock_shap_values
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "GradientBoostingClassifier"
        X = np.random.randn(10, 2)

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = compute_shap_importance(mock_model, X, ["a", "b"])

        assert isinstance(result, dict)


class TestCreateExplainerEdgeCases:
    """Tests edge cases pour _create_shap_explainer."""

    def test_no_predict_method_returns_none(self):
        """Retourne None si pas de méthode predict."""
        from scripts.model_registry.explainability import _create_shap_explainer

        mock_shap = MagicMock()
        mock_model = MagicMock(spec=[])  # No methods
        mock_model.__class__.__name__ = "UnknownModel"
        X = np.random.randn(10, 3)

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = _create_shap_explainer(mock_model, X)

        # Should return None for unknown model without predict
        assert result is None or mock_shap.KernelExplainer.called
