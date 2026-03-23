"""Tests for residual learning — Elo init scores (ISO 24029).

Document ID: ALICE-TEST-RESIDUAL
Version: 1.1.0
Tests count: 4
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
