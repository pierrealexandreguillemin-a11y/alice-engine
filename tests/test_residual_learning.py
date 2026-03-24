"""Tests for residual learning — Elo init scores (ISO 24029).

Document ID: ALICE-TEST-RESIDUAL
Version: 2.0.0
Tests count: 9
"""

import numpy as np


def test_elo_init_scores_shape_and_values():
    """Init scores must be (n, 3) log-odds, finite, matching Elo probas."""
    from scripts.baselines import compute_elo_baseline, compute_elo_init_scores

    n = 100
    blanc_elo = np.full(n, 1600)
    noir_elo = np.full(n, 1400)

    import pandas as pd

    lookup = pd.DataFrame(
        {
            "elo_band": ["1400-1600"],
            "diff_band": ["200-400"],
            "draw_rate_prior": [0.15],
        }
    )
    elo_proba = compute_elo_baseline(blanc_elo, noir_elo, lookup)
    init_scores = compute_elo_init_scores(elo_proba)

    assert init_scores.shape == (n, 3)
    assert np.all(np.isfinite(init_scores))
    assert init_scores[0, 2] > 0  # P(win) > 0.5 → positive log-odds
    # Centered log-odds: rows must sum to ~0 (zero-sum property)
    np.testing.assert_allclose(init_scores.sum(axis=1), 0, atol=1e-10)


def test_elo_init_scores_clipping():
    """Probas near 0 or 1 must not produce inf log-odds."""
    from scripts.baselines import compute_elo_init_scores

    extreme = np.array([[0.001, 0.001, 0.998]])
    scores = compute_elo_init_scores(extreme)
    assert np.all(np.isfinite(scores))
    # With clip_min=1e-4 and 3 classes, theoretical max ≈ 6.14.
    # Bound at 10 to catch regressions if clip defaults change.
    assert np.abs(scores).max() < 10


def test_elo_init_scores_roundtrip():
    """softmax(init_scores) should approximately recover original probas."""
    from scripts.baselines import compute_elo_init_scores

    proba = np.array([[0.3, 0.15, 0.55], [0.45, 0.10, 0.45]])
    scores = compute_elo_init_scores(proba)
    exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
    recovered = exp_s / exp_s.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(recovered, proba, atol=0.01)


def test_train_all_sequential_accepts_init_scores():
    """train_all_sequential must accept optional init_scores parameter."""
    import inspect

    from scripts.kaggle_trainers import train_all_sequential

    sig = inspect.signature(train_all_sequential)
    assert "init_scores_train" in sig.parameters
    assert "init_scores_valid" in sig.parameters


class TestPredictWithInit:
    """Tests predict_with_init dispatch for residual learning — 4 tests."""

    def test_none_init_scores_uses_predict_proba(self):
        """Without init_scores, falls back to model.predict_proba."""
        from unittest.mock import MagicMock

        from scripts.kaggle_metrics import predict_with_init

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.2, 0.5]])
        X = MagicMock()
        result = predict_with_init(model, X, init_scores=None)
        model.predict_proba.assert_called_once()
        assert result.shape == (1, 3)

    def test_catboost_uses_raw_formula(self):
        """CatBoost with init_scores uses RawFormulaVal + softmax."""
        from unittest.mock import MagicMock

        from scripts.kaggle_metrics import predict_with_init

        model = MagicMock()
        type(model).__name__ = "CatBoostClassifier"
        # Neutral raw logits: after adding init_scores the softmax must still sum to 1
        model.predict.return_value = np.array([[0.0, 0.0, 0.0]])
        init = np.array([[0.1, -0.5, 0.4]])
        result = predict_with_init(model, MagicMock(), init_scores=init)
        model.predict.assert_called_once()
        # Output must be valid probability distribution
        np.testing.assert_allclose(result.sum(axis=1), [1.0], atol=1e-6)

    def test_lgbm_uses_raw_score(self):
        """LightGBM with init_scores uses raw_score=True + softmax."""
        from unittest.mock import MagicMock

        from scripts.kaggle_metrics import predict_with_init

        model = MagicMock()
        type(model).__name__ = "LGBMClassifier"
        model.predict.return_value = np.array([[0.1, -0.3, 0.2]])
        init = np.array([[0.0, 0.0, 0.0]])
        result = predict_with_init(model, MagicMock(), init_scores=init)
        model.predict.assert_called_once()
        # Output must be valid probability distribution
        np.testing.assert_allclose(result.sum(axis=1), [1.0], atol=1e-6)
        assert result.shape == (1, 3)

    def test_output_sums_to_one(self):
        """All predict_with_init paths must produce probabilities that sum to 1."""
        from unittest.mock import MagicMock

        from scripts.kaggle_metrics import predict_with_init

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.15, 0.55], [0.4, 0.1, 0.5]])
        result = predict_with_init(model, MagicMock(), init_scores=None)
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], atol=1e-6)


def test_mean_p_draw_below_threshold_fails():
    """Quality gate must fail when mean P(draw) < 0.01 (model ignores draws)."""
    from unittest.mock import MagicMock

    from scripts.kaggle_metrics import check_quality_gates

    results = {
        "CatBoost": {
            "model": MagicMock(),
            "metrics": {
                "test_log_loss": 0.85,
                "test_accuracy": 0.7,
                "test_f1_macro": 0.6,
                "test_rps": 0.15,
                "test_brier": 0.30,
                "test_es_mae": 0.25,
                "ece_class_loss": 0.01,
                "ece_class_draw": 0.01,
                "ece_class_win": 0.01,
                "draw_calibration_bias": 0.005,
                "recall_draw": 0.05,
                "mean_p_draw": 0.005,  # Below 0.01 threshold
            },
            "importance": {},
        }
    }
    baselines = {
        "naive": {"log_loss": 1.10, "rps": 0.22, "brier": 0.40},
        "elo": {"log_loss": 1.05, "rps": 0.20, "es_mae": 0.35},
    }
    gate = check_quality_gates(results, baseline_metrics=baselines)
    assert gate["passed"] is False
    assert "mean_p_draw" in gate["reason"]
