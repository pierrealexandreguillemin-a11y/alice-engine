"""Tests Search Space Parsing - ISO 29119.

Document ID: ALICE-TEST-TRAINING-OPTUNA-SEARCH-SPACE
Version: 1.0.0

Tests pour le parsing du search space.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""


class TestSearchSpaceParsing:
    """Tests pour le parsing du search space."""

    def test_catboost_uses_min_data_in_leaf(self, sample_data, sample_config):
        """Vérifie que CatBoost utilise min_data_in_leaf."""
        assert "min_data_in_leaf" in sample_config["optuna"]["catboost_search_space"]

    def test_xgboost_uses_subsampling(self, sample_data, sample_config):
        """Vérifie que XGBoost utilise colsample_bytree et subsample."""
        search_space = sample_config["optuna"]["xgboost_search_space"]

        assert "colsample_bytree" in search_space
        assert "subsample" in search_space

    def test_lightgbm_uses_bagging(self, sample_data, sample_config):
        """Vérifie que LightGBM utilise feature_fraction et bagging_fraction."""
        search_space = sample_config["optuna"]["lightgbm_search_space"]

        assert "feature_fraction" in search_space
        assert "bagging_fraction" in search_space
        assert "min_child_samples" in search_space
