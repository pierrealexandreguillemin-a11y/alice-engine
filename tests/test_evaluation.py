"""Tests Evaluation Module - ISO 29119/25059.

Document ID: ALICE-TEST-EVAL
Version: 1.0.0
Tests: 8

Classes:
- TestConstants: Tests constantes configuration (3 tests)
- TestEvaluateModel: Tests fonction évaluation (3 tests)
- TestEvaluateModelsMain: Tests point d'entrée (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 25059:2023 - AI Quality Model (Métriques)
- ISO/IEC 5055:2021 - Code Quality (<100 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd


class TestConstants:
    """Tests pour scripts/evaluation/constants.py."""

    def test_project_dir_exists(self):
        """PROJECT_DIR pointe vers racine projet."""
        from scripts.evaluation.constants import PROJECT_DIR

        assert PROJECT_DIR.exists()
        assert (PROJECT_DIR / "scripts").exists()

    def test_numeric_features_defined(self):
        """Features numériques définies."""
        from scripts.evaluation.constants import NUMERIC_FEATURES

        assert isinstance(NUMERIC_FEATURES, list)
        assert "blanc_elo" in NUMERIC_FEATURES
        assert "diff_elo" in NUMERIC_FEATURES

    def test_categorical_features_defined(self):
        """Features catégorielles définies."""
        from scripts.evaluation.constants import CATEGORICAL_FEATURES

        assert isinstance(CATEGORICAL_FEATURES, list)
        assert "type_competition" in CATEGORICAL_FEATURES


class TestEvaluateModel:
    """Tests pour scripts/evaluation/metrics.py."""

    def test_evaluate_returns_metrics(self):
        """Retourne dictionnaire de métriques."""
        from scripts.evaluation.metrics import evaluate_model

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array(
            [[0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]]
        )

        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        y = pd.Series([1, 0, 1, 0])

        result = evaluate_model(mock_model, X, y, "TestModel")

        assert "auc_roc" in result
        assert "accuracy" in result
        assert result["model"] == "TestModel"

    def test_evaluate_computes_inference_time(self):
        """Calcule temps d'inférence."""
        from scripts.evaluation.metrics import evaluate_model

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])

        X = pd.DataFrame({"a": [1]})
        y = pd.Series([1])

        result = evaluate_model(mock_model, X, y, "Test")

        assert "inference_time_ms" in result
        assert result["inference_time_ms"] >= 0

    def test_evaluate_counts_samples(self):
        """Compte le nombre d'échantillons."""
        from scripts.evaluation.metrics import evaluate_model

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1, 0, 1])

        result = evaluate_model(mock_model, X, y, "Test")

        assert result["samples"] == 3


class TestEvaluateModelsMain:
    """Tests pour scripts/evaluate_models.py."""

    def test_main_function_exists(self):
        """Fonction main existe."""
        from scripts.evaluate_models import main

        assert callable(main)

    def test_argparse_defaults(self):
        """Arguments par défaut configurés."""
        from scripts.evaluation.constants import DEFAULT_DATA_DIR

        assert DEFAULT_DATA_DIR is not None
        assert isinstance(DEFAULT_DATA_DIR, Path)
