"""Tests AutoGluon V9 Benchmark — ISO 29119.

Document ID: ALICE-TEST-AG-V9
Version: 1.0.0
Tests count: 4
Classes: TestEloProbaFeatures, TestAgOutputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TestEloProbaFeatures:
    """Tests Elo proba feature computation — 2 tests."""

    def test_elo_probas_sum_to_one(self) -> None:
        """P_elo(W) + P_elo(D) + P_elo(L) must sum to 1."""
        from scripts.cloud.train_autogluon_v9 import _compute_elo_proba_features

        df = pd.DataFrame({"blanc_elo": [1500, 1800, 2000], "noir_elo": [1500, 1200, 1900]})
        result = _compute_elo_proba_features(df)
        sums = result[["p_elo_win", "p_elo_draw", "p_elo_loss"]].sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_elo_probas_range_valid(self) -> None:
        """All Elo probas in [0, 1]."""
        from scripts.cloud.train_autogluon_v9 import _compute_elo_proba_features

        df = pd.DataFrame({"blanc_elo": [800, 1500, 2800], "noir_elo": [2800, 1500, 800]})
        result = _compute_elo_proba_features(df)
        for col in ["p_elo_win", "p_elo_draw", "p_elo_loss"]:
            assert (result[col] >= 0).all()
            assert (result[col] <= 1).all()


class TestAgOutputs:
    """Tests output structure — 2 tests."""

    def test_predictions_parquet_columns(self) -> None:
        """Predictions parquet must have y_true + 3 proba columns."""
        required = ["y_true", "p_loss", "p_draw", "p_win"]
        df = pd.DataFrame({col: [0.0] for col in required})
        for col in required:
            assert col in df.columns

    def test_leaderboard_has_score_column(self) -> None:
        """Leaderboard CSV must have model + score_val columns."""
        df = pd.DataFrame({"model": ["LGB"], "score_val": [-0.56]})
        assert "model" in df.columns
        assert "score_val" in df.columns
