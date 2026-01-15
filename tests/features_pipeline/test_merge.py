"""Tests Merge All Features - ISO 5259.

Document ID: ALICE-TEST-FEATURES-PIPELINE-MERGE
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.pipeline import extract_all_features, merge_all_features


class TestMergeAllFeatures:
    """Tests pour merge_all_features."""

    def test_basic_merge(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test merge basique des features."""
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )
        result = merge_all_features(sample_target.copy(), features, include_advanced=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_target)

    def test_merge_preserves_original_columns(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test merge preserve colonnes originales."""
        original_cols = set(sample_target.columns)
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )
        result = merge_all_features(sample_target.copy(), features, include_advanced=False)

        for col in original_cols:
            assert col in result.columns, f"Colonne originale perdue: {col}"

    def test_merge_adds_feature_columns(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test merge ajoute colonnes features."""
        original_col_count = len(sample_target.columns)
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )
        result = merge_all_features(sample_target.copy(), features, include_advanced=False)

        assert len(result.columns) >= original_col_count

    def test_merge_with_advanced_features(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test merge avec features avancees."""
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=True
        )
        result = merge_all_features(sample_target.copy(), features, include_advanced=True)

        assert isinstance(result, pd.DataFrame)

    def test_merge_handles_empty_features(self, sample_target: pd.DataFrame) -> None:
        """Test merge gere features vides gracieusement."""
        empty_features = {
            "club_reliability": pd.DataFrame(),
            "player_reliability": pd.DataFrame(),
            "recent_form": pd.DataFrame(),
            "board_position": pd.DataFrame(),
            "color_perf": pd.DataFrame(),
            "ffe_regulatory": pd.DataFrame(),
            "team_enjeu": pd.DataFrame(),
        }

        result = merge_all_features(sample_target.copy(), empty_features, include_advanced=False)
        assert isinstance(result, pd.DataFrame)
