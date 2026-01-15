"""Tests Calculate Recent Form - ISO 29119.

Document ID: ALICE-TEST-FEATURES-RECENT-FORM-FORM
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.recent_form import calculate_recent_form


class TestCalculateRecentForm:
    """Tests pour calculate_recent_form."""

    def test_basic_form_calculation(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test calcul forme basique avec donnees valides."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        assert not result.empty
        assert "joueur_nom" in result.columns
        assert "forme_recente" in result.columns
        assert "nb_matchs_forme" in result.columns
        assert "forme_tendance" in result.columns

    def test_form_values_in_valid_range(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test que forme_recente est dans [0, 1]."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        assert result["forme_recente"].min() >= 0.0
        assert result["forme_recente"].max() <= 1.0

    def test_joueur_a_hausse_tendance(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test Joueur A avec forme croissante = tendance hausse."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        assert len(joueur_a) == 1
        assert joueur_a.iloc[0]["forme_tendance"] == "hausse"

    def test_joueur_b_baisse_tendance(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test Joueur B avec forme decroissante = tendance baisse."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        joueur_b = result[result["joueur_nom"] == "Joueur B"]
        assert len(joueur_b) == 1
        assert joueur_b.iloc[0]["forme_tendance"] == "baisse"

    def test_joueur_excluded_if_insufficient_matches(
        self, sample_matches_for_form: pd.DataFrame
    ) -> None:
        """Test Joueur C avec <5 matchs est exclu."""
        result = calculate_recent_form(sample_matches_for_form, window=5)

        joueur_c = result[result["joueur_nom"] == "Joueur C"]
        assert len(joueur_c) == 0

    def test_empty_dataframe_returns_empty(self) -> None:
        """Test DataFrame vide retourne DataFrame vide."""
        result = calculate_recent_form(pd.DataFrame(), window=5)

        assert result.empty

    def test_forfaits_excluded_from_calculation(self, matches_with_forfaits: pd.DataFrame) -> None:
        """Test ISO 5259: forfaits exclus du calcul forme."""
        result = calculate_recent_form(matches_with_forfaits, window=5)

        joueur_d = result[result["joueur_nom"] == "Joueur D"]
        assert len(joueur_d) == 1
        assert joueur_d.iloc[0]["forme_recente"] == 1.0

    def test_custom_window_size(self, sample_matches_for_form: pd.DataFrame) -> None:
        """Test avec taille de fenetre personnalisee."""
        result = calculate_recent_form(sample_matches_for_form, window=3)

        joueur_c = result[result["joueur_nom"] == "Joueur C"]
        assert len(joueur_c) == 1

    def test_missing_player_columns_returns_empty(self) -> None:
        """Test colonnes joueur manquantes retourne vide."""
        df = pd.DataFrame(
            {
                "autre_colonne": [1, 2, 3],
                "type_resultat": ["victoire_blanc", "nulle", "victoire_noir"],
            }
        )
        result = calculate_recent_form(df, window=5)

        assert result.empty
