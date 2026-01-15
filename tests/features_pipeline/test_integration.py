"""Tests Pipeline Integration - ISO 5259.

Document ID: ALICE-TEST-FEATURES-PIPELINE-INTEGRATION
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


class TestPipelineIntegration:
    """Tests integration complete du pipeline."""

    def test_full_pipeline_basic(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test pipeline complet basique."""
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )
        result = merge_all_features(sample_target.copy(), features, include_advanced=False)

        assert not result.empty
        assert len(result) == len(sample_target)

    def test_full_pipeline_advanced(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test pipeline complet avec features avancees."""
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=True
        )
        result = merge_all_features(sample_target.copy(), features, include_advanced=True)

        assert not result.empty

    def test_pipeline_idempotent(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test pipeline est idempotent."""
        features1 = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )
        features2 = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )

        assert set(features1.keys()) == set(features2.keys())

        for key in features1:
            assert len(features1[key]) == len(features2[key])


class TestPipelineDataQuality:
    """Tests qualite donnees ISO 5259."""

    def test_no_data_leakage(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test ISO 5259: pas de fuite de donnees futur vers passe."""
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )

        if not features["recent_form"].empty and "ronde" in features["recent_form"].columns:
            max_ronde = features["recent_form"]["ronde"].max()
            assert max_ronde < 10, "Fuite de donnees: forme contient rondes futures"

    def test_feature_types_consistent(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
    ) -> None:
        """Test types de features consistants."""
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )

        for name, df in features.items():
            assert isinstance(df, pd.DataFrame), f"{name} n'est pas DataFrame"

            for col in df.columns:
                if len(df) > 0 and df[col].dtype == "object":
                    sample = df[col].dropna()
                    if len(sample) >= 5:
                        types_in_col = {type(v).__name__ for v in sample}
                        assert (
                            len(types_in_col) <= 2
                        ), f"Types mixtes dans {name}.{col}: {types_in_col}"
