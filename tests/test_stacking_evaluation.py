"""Tests for stacking evaluation — ISO 29119.

Document ID: ALICE-TEST-STACKING
Version: 1.0.0
Tests count: 8
"""

import numpy as np
import pandas as pd
import pytest


def _make_pred_df(n: int = 100, seed: int = 42, y_seed: int = 0) -> pd.DataFrame:
    """Create a synthetic prediction parquet with same schema as V8 outputs."""
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet([1, 1, 1], size=n).astype(np.float32)
    y_rng = np.random.RandomState(y_seed)
    y_true = y_rng.choice([0, 1, 2], size=n, p=[0.42, 0.13, 0.45])
    return pd.DataFrame(
        {
            "y_true": y_true.astype(np.int64),
            "y_proba_loss": raw[:, 0],
            "y_proba_draw": raw[:, 1],
            "y_proba_win": raw[:, 2],
            "y_pred": raw.argmax(axis=1).astype(np.int64),
            "y_proba_cal_loss": raw[:, 0],
            "y_proba_cal_draw": raw[:, 1],
            "y_proba_cal_win": raw[:, 2],
            "y_pred_calibrated": raw.argmax(axis=1).astype(np.int64),
        }
    )


class TestAssembleMetaFeatures:
    """Tests for assembling 9-column meta-feature matrix."""

    def test_output_shape_9_columns(self) -> None:
        from scripts.evaluate_stacking import assemble_meta_features

        dfs = [_make_pred_df(50, seed=i) for i in range(3)]
        X, y = assemble_meta_features(dfs, calibrated=True)
        assert X.shape == (50, 9)
        assert y.shape == (50,)

    def test_uses_raw_probas_when_calibrated_false(self) -> None:
        from scripts.evaluate_stacking import assemble_meta_features

        dfs = [_make_pred_df(30, seed=i) for i in range(3)]
        X_raw, _ = assemble_meta_features(dfs, calibrated=False)
        X_cal, _ = assemble_meta_features(dfs, calibrated=True)
        assert X_raw.shape == X_cal.shape == (30, 9)

    def test_y_true_consistent_across_models(self) -> None:
        from scripts.evaluate_stacking import assemble_meta_features

        df1 = _make_pred_df(20, seed=0)
        df2 = _make_pred_df(20, seed=0)
        df3 = _make_pred_df(20, seed=0)
        _, y = assemble_meta_features([df1, df2, df3], calibrated=True)
        assert np.array_equal(y, df1["y_true"].values)

    def test_raises_on_mismatched_y_true(self) -> None:
        from scripts.evaluate_stacking import assemble_meta_features

        df1 = _make_pred_df(20, seed=0)
        df2 = _make_pred_df(20, seed=99, y_seed=99)
        df3 = _make_pred_df(20, seed=0)
        with pytest.raises(ValueError, match="y_true mismatch"):
            assemble_meta_features([df1, df2, df3], calibrated=True)


class TestComputeAllMetrics:
    """Tests for metrics computation wrapper."""

    def test_returns_required_keys(self) -> None:
        from scripts.evaluate_stacking import compute_all_metrics

        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1, 2], size=200, p=[0.42, 0.13, 0.45])
        y_proba = rng.dirichlet([2, 1, 2], size=200)
        metrics = compute_all_metrics(y_true, y_proba)
        required = {
            "log_loss",
            "rps",
            "es_mae",
            "brier",
            "ece_loss",
            "ece_draw",
            "ece_win",
            "draw_calibration_bias",
            "accuracy",
            "f1_macro",
        }
        assert required.issubset(set(metrics.keys()))

    def test_perfect_predictions_low_loss(self) -> None:
        from scripts.evaluate_stacking import compute_all_metrics

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.eye(3)[y_true] * 0.98 + 0.01 / 3
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        metrics = compute_all_metrics(y_true, y_proba)
        assert metrics["log_loss"] < 0.1
        assert metrics["accuracy"] == 1.0


class TestStackingEvaluation:
    """Tests for the full stacking pipeline on synthetic data."""

    def test_lr_meta_learner_produces_valid_probas(self) -> None:
        from scripts.evaluate_stacking import fit_meta_learner

        rng = np.random.RandomState(42)
        X_train = rng.dirichlet([2, 1, 2], size=(300, 3)).reshape(300, 9)
        y_train = rng.choice([0, 1, 2], size=300)
        X_test = rng.dirichlet([2, 1, 2], size=(100, 3)).reshape(100, 9)

        meta, probas = fit_meta_learner(X_train, y_train, X_test, kind="lr")
        assert probas.shape == (100, 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
        assert (probas >= 0).all()

    def test_mlp_meta_learner_produces_valid_probas(self) -> None:
        from scripts.evaluate_stacking import fit_meta_learner

        rng = np.random.RandomState(42)
        X_train = rng.dirichlet([2, 1, 2], size=(300, 3)).reshape(300, 9)
        y_train = rng.choice([0, 1, 2], size=300)
        X_test = rng.dirichlet([2, 1, 2], size=(100, 3)).reshape(100, 9)

        meta, probas = fit_meta_learner(X_train, y_train, X_test, kind="mlp")
        assert probas.shape == (100, 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
