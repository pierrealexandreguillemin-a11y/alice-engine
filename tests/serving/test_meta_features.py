"""Tests for meta-feature computation (ISO 29119)."""

import numpy as np

from scripts.serving.meta_features import build_meta_features


class TestBuildMetaFeatures:
    """Test meta-feature computation from 3 model probabilities."""

    def test_output_shape(self):
        """9 probas in, 18 features out (9 base + 9 engineered)."""
        p_xgb = np.array([[0.3, 0.2, 0.5]])
        p_lgb = np.array([[0.25, 0.25, 0.5]])
        p_cb = np.array([[0.35, 0.15, 0.5]])
        result = build_meta_features(p_xgb, p_lgb, p_cb)
        assert result.shape == (1, 18)

    def test_first_9_are_probas(self):
        """First 9 columns = concatenated base probabilities."""
        p_xgb = np.array([[0.3, 0.2, 0.5]])
        p_lgb = np.array([[0.25, 0.25, 0.5]])
        p_cb = np.array([[0.35, 0.15, 0.5]])
        result = build_meta_features(p_xgb, p_lgb, p_cb)
        np.testing.assert_array_almost_equal(
            result[0, :9], [0.3, 0.2, 0.5, 0.25, 0.25, 0.5, 0.35, 0.15, 0.5]
        )

    def test_std_features(self):
        """Columns 9-11 = per-class std across 3 models."""
        p_xgb = np.array([[0.3, 0.2, 0.5]])
        p_lgb = np.array([[0.3, 0.2, 0.5]])  # identical
        p_cb = np.array([[0.3, 0.2, 0.5]])
        result = build_meta_features(p_xgb, p_lgb, p_cb)
        np.testing.assert_array_almost_equal(result[0, 9:12], [0.0, 0.0, 0.0])

    def test_batch(self):
        """Works with multiple samples."""
        rng = np.random.RandomState(42)
        n = 100
        p_xgb = rng.dirichlet([1, 1, 1], size=n)
        p_lgb = rng.dirichlet([1, 1, 1], size=n)
        p_cb = rng.dirichlet([1, 1, 1], size=n)
        result = build_meta_features(p_xgb, p_lgb, p_cb)
        assert result.shape == (100, 18)
        assert np.all(np.isfinite(result))
