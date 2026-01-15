"""Tests Extract Team Enjeu Fallback - ISO 5259.

Document ID: ALICE-TEST-FEATURES-ENJEU-FALLBACK
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.enjeu import extract_team_enjeu_fallback


class TestExtractTeamEnjeuFallback:
    """Tests pour extract_team_enjeu_fallback."""

    def test_fallback_basic(self, sample_matches: pd.DataFrame) -> None:
        """Test fallback extraction basique."""
        result = extract_team_enjeu_fallback(sample_matches)

        assert not result.empty
        assert "equipe" in result.columns
        assert "zone_enjeu" in result.columns
        assert "is_fallback" in result.columns

    def test_fallback_marks_as_fallback(self, sample_matches: pd.DataFrame) -> None:
        """Test ISO 5259: fallback marque comme estimation."""
        result = extract_team_enjeu_fallback(sample_matches)

        assert result["is_fallback"].all()

    def test_fallback_estimates_middle_position(self, sample_matches: pd.DataFrame) -> None:
        """Test fallback estime position mi-tableau."""
        result = extract_team_enjeu_fallback(sample_matches)

        for _, row in result.iterrows():
            expected_mid = row["nb_equipes"] // 2
            assert row["position"] == expected_mid

    def test_fallback_deduplicated(self, sample_matches: pd.DataFrame) -> None:
        """Test fallback deduplique par equipe/saison."""
        result = extract_team_enjeu_fallback(sample_matches)

        duplicates = result.duplicated(subset=["equipe", "saison"])
        assert not duplicates.any()

    def test_fallback_empty_df(self) -> None:
        """Test fallback avec DataFrame vide."""
        result = extract_team_enjeu_fallback(pd.DataFrame())
        assert result.empty

    def test_fallback_missing_equipe_columns(self) -> None:
        """Test fallback sans colonnes equipe."""
        df = pd.DataFrame({"autre": [1], "saison": [2025], "ronde": [1]})
        result = extract_team_enjeu_fallback(df)
        assert result.empty

    def test_fallback_counts_rondes(self, sample_matches: pd.DataFrame) -> None:
        """Test fallback compte les rondes par equipe."""
        result = extract_team_enjeu_fallback(sample_matches)

        club_a = result[result["equipe"] == "N1 Club A"]
        if len(club_a) > 0:
            assert club_a.iloc[0]["nb_rondes"] == 2
