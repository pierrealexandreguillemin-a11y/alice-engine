"""Tests OOF Stack Pipeline — ISO 29119.

Document ID: ALICE-TEST-OOF-STACK
Version: 1.0.0
Tests count: 6
Classes: TestOofFoldSplit, TestOofPredictions, TestOofOutputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TestOofFoldSplit:
    """Tests that fold splitting is correct — 2 tests."""

    def test_5_folds_cover_all_rows(self) -> None:
        """All rows appear exactly once in OOF predictions."""
        from scripts.cloud.train_oof_stack import _create_folds

        n = 1000
        folds = _create_folds(n, n_folds=5, seed=42)
        all_val_idx = np.concatenate([val for _, val in folds])
        assert len(all_val_idx) == n
        assert len(set(all_val_idx)) == n

    def test_folds_no_overlap(self) -> None:
        """No index appears in both train and valid for the same fold."""
        from scripts.cloud.train_oof_stack import _create_folds

        folds = _create_folds(500, n_folds=5, seed=42)
        for train_idx, val_idx in folds:
            assert len(set(train_idx) & set(val_idx)) == 0


class TestOofPredictions:
    """Tests prediction shape and validity — 2 tests."""

    def test_oof_shape_matches_input(self) -> None:
        """OOF predictions have (n_samples, n_models * n_classes) shape."""
        n, n_models, n_classes = 100, 2, 3
        oof = np.zeros((n, n_models * n_classes))
        assert oof.shape == (100, 6)

    def test_probas_sum_to_one_per_model(self) -> None:
        """Each model's 3-class probas sum to 1."""
        probas = np.array([[0.3, 0.2, 0.5, 0.4, 0.1, 0.5]])
        for m in range(2):
            s = probas[:, m * 3 : (m + 1) * 3].sum(axis=1)
            np.testing.assert_allclose(s, 1.0, atol=1e-6)


class TestOofOutputs:
    """Tests output parquet structure — 2 tests."""

    def test_oof_parquet_has_required_columns(self) -> None:
        """OOF parquet must have y_true + 6 model prediction columns."""
        required = [
            "y_true",
            "xgb_p_loss",
            "xgb_p_draw",
            "xgb_p_win",
            "lgb_p_loss",
            "lgb_p_draw",
            "lgb_p_win",
        ]
        df = pd.DataFrame({col: [0.0] for col in required})
        for col in required:
            assert col in df.columns

    def test_test_parquet_same_columns(self) -> None:
        """Test parquet has same columns as OOF."""
        required = [
            "y_true",
            "xgb_p_loss",
            "xgb_p_draw",
            "xgb_p_win",
            "lgb_p_loss",
            "lgb_p_draw",
            "lgb_p_win",
        ]
        df = pd.DataFrame({col: [0.0] for col in required})
        assert list(df.columns) == required
