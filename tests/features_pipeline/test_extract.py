"""Tests Extract All Features - ISO 5259.

Document ID: ALICE-TEST-FEATURES-PIPELINE-EXTRACT
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.pipeline import extract_all_features


class TestExtractAllFeatures:
    """Tests pour extract_all_features."""

    def test_basic_extraction(
        self, sample_history: pd.DataFrame, sample_history_played: pd.DataFrame
    ) -> None:
        """Test extraction basique sans features avancees."""
        result = extract_all_features(sample_history, sample_history_played, include_advanced=False)

        assert isinstance(result, dict)
        expected_keys = [
            "club_reliability",
            "player_reliability",
            "recent_form",
            "board_position",
            "color_perf",
            "ffe_regulatory",
            "team_enjeu",
        ]
        for key in expected_keys:
            assert key in result, f"Feature manquante: {key}"
            assert isinstance(result[key], pd.DataFrame)

    def test_extraction_with_advanced(
        self, sample_history: pd.DataFrame, sample_history_played: pd.DataFrame
    ) -> None:
        """Test extraction avec features avancees."""
        result = extract_all_features(sample_history, sample_history_played, include_advanced=True)

        advanced_keys = ["h2h", "pressure", "trajectory"]
        for key in advanced_keys:
            assert key in result, f"Feature avancee manquante: {key}"

    def test_empty_history_returns_empty_features(self) -> None:
        """Test historique vide retourne features vides."""
        empty_df = pd.DataFrame()
        result = extract_all_features(empty_df, empty_df, include_advanced=False)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(value, pd.DataFrame), f"{key} n'est pas un DataFrame"

    def test_feature_count_without_advanced(
        self, sample_history: pd.DataFrame, sample_history_played: pd.DataFrame
    ) -> None:
        """Test nombre de features sans avancees = 7."""
        result = extract_all_features(sample_history, sample_history_played, include_advanced=False)
        assert len(result) == 7

    def test_feature_count_with_advanced(
        self, sample_history: pd.DataFrame, sample_history_played: pd.DataFrame
    ) -> None:
        """Test nombre de features avec avancees = 10."""
        result = extract_all_features(sample_history, sample_history_played, include_advanced=True)
        assert len(result) == 10
