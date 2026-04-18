"""Tests for HF Hub model loading (ISO 29119 / ISO 27001).

Document ID: ALICE-TEST-MODEL-LOADER
Version: 1.0.0
Count: 3 test methods
"""

import pytest

from scripts.serving.model_loader import ModelBundle, load_models


class TestModelBundle:
    """Tests for ModelBundle dataclass construction."""

    def test_bundle_has_required_fields(self):
        bundle = ModelBundle(
            lgb_model=None,
            xgb_model=None,
            cb_model=None,
            mlp_model=None,
            temperature=1.0,
            draw_rate_lookup=None,
            encoders=None,
            fallback_mode=False,
            version="test",
        )
        assert bundle.fallback_mode is False
        assert bundle.temperature == 1.0

    def test_fallback_mode_flag(self):
        bundle = ModelBundle(
            lgb_model="mock",
            xgb_model=None,
            cb_model=None,
            mlp_model=None,
            temperature=1.0,
            draw_rate_lookup=None,
            encoders=None,
            fallback_mode=True,
            version="test",
        )
        assert bundle.fallback_mode is True


class TestLoadModels:
    """Tests for load_models() with various cache states."""

    def test_missing_cache_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_models(
                cache_dir=tmp_path / "nonexistent",
                hf_repo_id="fake/repo",
                download=False,
            )
