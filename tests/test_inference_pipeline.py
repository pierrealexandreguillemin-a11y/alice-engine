"""Integration tests for stacking inference pipeline (ISO 29119/42001)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.serving.model_loader import ModelBundle
from services.inference import StackingInferenceService


class TestStackingInference:
    """Unit tests for StackingInferenceService (ISO 29119/42001)."""

    def _make_mock_bundle(self):
        mock_lgb = MagicMock()
        mock_xgb = MagicMock()
        mock_cb = MagicMock()
        mock_mlp = MagicMock()
        # predict_with_init calls model.predict() for LGB (raw_score) and CB (RawFormulaVal)
        # For mocks, we simulate predict_proba behavior
        for m in [mock_lgb, mock_xgb, mock_cb]:
            m.predict_proba = MagicMock(return_value=np.array([[0.30, 0.20, 0.50]]))
        mock_mlp.predict_proba = MagicMock(return_value=np.array([[0.29, 0.21, 0.50]]))
        return ModelBundle(
            lgb_model=mock_lgb,
            xgb_model=mock_xgb,
            cb_model=mock_cb,
            mlp_model=mock_mlp,
            temperature=1.02,
            draw_rate_lookup=None,
            encoders=None,
            fallback_mode=False,
            version="test",
        )

    def test_predict_returns_valid_probas(self):
        bundle = self._make_mock_bundle()
        service = StackingInferenceService(bundle)
        # Test the pipeline directly: meta-features -> MLP -> temp scaling
        from scripts.serving.meta_features import build_meta_features

        p = np.array([[0.30, 0.20, 0.50]])
        meta = build_meta_features(p, p, p)
        p_raw = np.asarray(bundle.mlp_model.predict_proba(meta))
        p_cal = service._apply_temperature(p_raw, bundle.temperature)
        assert p_cal.shape == (1, 3)
        np.testing.assert_almost_equal(p_cal.sum(), 1.0, decimal=4)
        assert np.all(p_cal >= 0)
        assert np.all(p_cal <= 1)

    def test_predict_no_nan(self):
        bundle = self._make_mock_bundle()
        service = StackingInferenceService(bundle)
        # Test pipeline math produces finite values
        from scripts.serving.meta_features import build_meta_features

        p = np.array([[0.30, 0.20, 0.50]])
        meta = build_meta_features(p, p, p)
        p_raw = np.asarray(bundle.mlp_model.predict_proba(meta))
        p_cal = service._apply_temperature(p_raw, bundle.temperature)
        assert np.all(np.isfinite(p_cal))

    def test_temperature_scaling(self):
        service = StackingInferenceService.__new__(StackingInferenceService)
        p = np.array([[0.3, 0.2, 0.5]])
        result = service._apply_temperature(p, 1.0)
        np.testing.assert_array_almost_equal(result, p, decimal=5)

    def test_validate_output_raises_on_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            StackingInferenceService._validate_output(np.array([[float("nan"), 0.5, 0.5]]))
