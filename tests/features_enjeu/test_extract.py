"""Tests Extract Team Enjeu Features - ISO 5259.

Document ID: ALICE-TEST-FEATURES-ENJEU-EXTRACT
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.enjeu import extract_team_enjeu_features


class TestExtractTeamEnjeuFeatures:
    """Tests pour extract_team_enjeu_features."""

    def test_basic_extraction(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test extraction basique avec classement valide."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        assert not result.empty
        assert "equipe" in result.columns
        assert "zone_enjeu" in result.columns
        assert "niveau_hierarchique" in result.columns

    def test_zone_enjeu_values(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test valeurs zone_enjeu valides."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        valid_zones = {"promotion", "maintien", "mi_tableau", "montee", "danger"}
        assert all(z in valid_zones for z in result["zone_enjeu"].unique())

    def test_first_position_is_promotion(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test position 1 = zone promotion/montee."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        equipe_a = result[result["equipe"] == "N1 Equipe A"]
        assert len(equipe_a) == 1
        assert equipe_a.iloc[0]["zone_enjeu"] in ["promotion", "montee"]

    def test_last_position_is_relegation(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test derniere position = zone maintien/danger."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        equipe_c = result[result["equipe"] == "N1 Equipe C"]
        assert len(equipe_c) == 1
        assert equipe_c.iloc[0]["zone_enjeu"] in ["maintien", "danger"]

    def test_middle_position_is_mi_tableau(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test position milieu = mi_tableau."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        equipe_b = result[result["equipe"] == "N1 Equipe B"]
        assert len(equipe_b) == 1
        assert equipe_b.iloc[0]["zone_enjeu"] == "mi_tableau"

    def test_niveau_hierarchique_calculated(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test calcul niveau hierarchique."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        n1_equipe = result[result["equipe"] == "N1 Equipe A"]
        n4_equipe = result[result["equipe"] == "N4 Equipe D"]

        assert n1_equipe.iloc[0]["niveau_hierarchique"] < n4_equipe.iloc[0]["niveau_hierarchique"]

    def test_empty_df_returns_empty(self, sample_standings: pd.DataFrame) -> None:
        """Test DataFrame vide retourne DataFrame vide."""
        result = extract_team_enjeu_features(pd.DataFrame(), sample_standings)
        assert result.empty

    def test_empty_standings_uses_fallback(self, sample_matches: pd.DataFrame) -> None:
        """Test classement vide utilise fallback."""
        result = extract_team_enjeu_features(sample_matches, pd.DataFrame())
        assert not result.empty

    def test_missing_columns_returns_empty(self) -> None:
        """Test colonnes manquantes retourne vide."""
        df = pd.DataFrame({"autre_colonne": [1, 2, 3]})
        result = extract_team_enjeu_features(df, pd.DataFrame())
        assert result.empty

    def test_all_required_columns_present(
        self, sample_matches: pd.DataFrame, sample_standings: pd.DataFrame
    ) -> None:
        """Test toutes les colonnes requises presentes."""
        result = extract_team_enjeu_features(sample_matches, sample_standings)

        required_cols = [
            "equipe",
            "saison",
            "competition",
            "division",
            "groupe",
            "ronde",
            "position",
            "points_cumules",
            "nb_equipes",
            "ecart_premier",
            "ecart_dernier",
            "zone_enjeu",
            "niveau_hierarchique",
        ]

        for col in required_cols:
            assert col in result.columns, f"Colonne manquante: {col}"
