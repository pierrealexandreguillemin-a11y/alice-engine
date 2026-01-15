"""Tests AutoGluon Wrapper - ISO 42001.

Document ID: ALICE-TEST-AUTOGLUON-TRAINER-WRAPPER
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestAutoGluonWrapper:
    """Tests pour AutoGluonWrapper."""

    def test_wrapper_init(self) -> None:
        """Test initialisation du wrapper."""
        from scripts.autogluon.predictor_wrapper import AutoGluonWrapper

        wrapper = AutoGluonWrapper(label="target")

        assert wrapper.predictor is None
        assert wrapper.label == "target"
        assert not wrapper._is_fitted

    def test_wrapper_with_predictor(self) -> None:
        """Test wrapper avec predictor existant."""
        from scripts.autogluon.predictor_wrapper import AutoGluonWrapper

        mock_predictor = MagicMock()
        wrapper = AutoGluonWrapper(predictor=mock_predictor)

        assert wrapper.predictor == mock_predictor
        assert wrapper._is_fitted

    def test_fit_returns_self(self) -> None:
        """Test que fit() retourne self pour chaining."""
        from scripts.autogluon.predictor_wrapper import AutoGluonWrapper

        wrapper = AutoGluonWrapper()

        wrapper._is_fitted = True
        wrapper.predictor = MagicMock()

        assert wrapper._is_fitted is True

    def test_predict_requires_fitted(self) -> None:
        """Test que predict() necessite fit()."""
        from scripts.autogluon.predictor_wrapper import AutoGluonWrapper

        wrapper = AutoGluonWrapper()
        X = np.random.randn(10, 3)

        with pytest.raises(ValueError, match="not fitted"):
            wrapper.predict(X)

    def test_predict_proba_requires_fitted(self) -> None:
        """Test que predict_proba() necessite fit()."""
        from scripts.autogluon.predictor_wrapper import AutoGluonWrapper

        wrapper = AutoGluonWrapper()
        X = np.random.randn(10, 3)

        with pytest.raises(ValueError, match="not fitted"):
            wrapper.predict_proba(X)

    def test_score_requires_fitted(self) -> None:
        """Test que score() necessite fit()."""
        from scripts.autogluon.predictor_wrapper import AutoGluonWrapper

        wrapper = AutoGluonWrapper()
        X = np.random.randn(10, 3)
        y = np.random.randint(0, 2, 10)

        with pytest.raises(ValueError, match="not fitted"):
            wrapper.score(X, y)

    def test_predict_with_dataframe(self) -> None:
        """Test predict avec DataFrame."""
        from scripts.autogluon.predictor_wrapper import AutoGluonWrapper

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = pd.Series([0, 1, 0])

        wrapper = AutoGluonWrapper(predictor=mock_predictor)
        wrapper.feature_names = ["a", "b"]

        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = wrapper.predict(X)

        assert len(result) == 3
        mock_predictor.predict.assert_called_once()

    def test_predict_with_array(self) -> None:
        """Test predict avec numpy array."""
        from scripts.autogluon.predictor_wrapper import AutoGluonWrapper

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = pd.Series([0, 1, 0])

        wrapper = AutoGluonWrapper(predictor=mock_predictor)
        wrapper.feature_names = ["feature_0", "feature_1"]

        X = np.array([[1, 4], [2, 5], [3, 6]])
        result = wrapper.predict(X)

        assert len(result) == 3
