"""Tests Calculate Selection Patterns - ISO 29119.

Document ID: ALICE-TEST-FEATURES-ALI-SELECTION
Version: 1.0.0

Tests pour calculate_selection_patterns.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd

from scripts.features.ali import calculate_selection_patterns


class TestCalculateSelectionPatterns:
    """Tests pour calculate_selection_patterns()."""

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test avec DataFrame vide retourne DataFrame vide."""
        result = calculate_selection_patterns(empty_df)
        assert result.empty

    def test_missing_echiquier_column(self) -> None:
        """Test sans colonne echiquier retourne vide."""
        df = pd.DataFrame({"saison": [2025], "ronde": [1], "blanc_nom": ["A"]})
        result = calculate_selection_patterns(df)
        assert result.empty

    def test_returns_dataframe(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test retourne un DataFrame."""
        result = calculate_selection_patterns(sample_echiquiers)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test colonnes requises présentes."""
        result = calculate_selection_patterns(sample_echiquiers)

        required_cols = [
            "joueur_nom",
            "saison",
            "role_type",
            "echiquier_prefere",
            "flexibilite_echiquier",
            "nb_echiquiers_differents",
        ]
        for col in required_cols:
            assert col in result.columns, f"Colonne manquante: {col}"

    def test_role_titulaire(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test classification 'titulaire' (>70% présence)."""
        result = calculate_selection_patterns(sample_echiquiers)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        assert len(joueur_a) == 1
        assert joueur_a.iloc[0]["role_type"] == "titulaire"

    def test_role_rotation(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test classification 'rotation' (30-70% présence)."""
        result = calculate_selection_patterns(sample_echiquiers)

        joueur_b = result[result["joueur_nom"] == "Joueur B"]
        assert len(joueur_b) == 1
        assert joueur_b.iloc[0]["role_type"] == "rotation"

    def test_role_remplacant(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test classification 'remplacant' (<30% présence)."""
        result = calculate_selection_patterns(sample_echiquiers)

        joueur_c = result[result["joueur_nom"] == "Joueur C"]
        assert len(joueur_c) == 1
        assert joueur_c.iloc[0]["role_type"] == "remplacant"

    def test_role_polyvalent(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test classification 'polyvalent' (std > 2, 3+ échiquiers)."""
        result = calculate_selection_patterns(sample_echiquiers)

        joueur_d = result[result["joueur_nom"] == "Joueur D"]
        assert len(joueur_d) == 1
        assert joueur_d.iloc[0]["role_type"] == "polyvalent"

    def test_echiquier_prefere(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test calcul échiquier préféré (modal)."""
        result = calculate_selection_patterns(sample_echiquiers)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        assert joueur_a.iloc[0]["echiquier_prefere"] == 1

    def test_flexibilite_zero_same_board(self) -> None:
        """Test flexibilité = 0 si toujours même échiquier."""
        df = pd.DataFrame(
            [
                {"saison": 2025, "ronde": 1, "blanc_nom": "Test", "noir_nom": "X", "echiquier": 1},
                {"saison": 2025, "ronde": 2, "blanc_nom": "Test", "noir_nom": "Y", "echiquier": 1},
            ]
        )
        result = calculate_selection_patterns(df)

        assert result.iloc[0]["flexibilite_echiquier"] == 0.0

    def test_flexibilite_high_varied_boards(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test flexibilité élevée si échiquiers variés."""
        result = calculate_selection_patterns(sample_echiquiers)

        joueur_d = result[result["joueur_nom"] == "Joueur D"]
        assert joueur_d.iloc[0]["flexibilite_echiquier"] > 2

    def test_nb_echiquiers_differents(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test calcul nombre échiquiers différents."""
        result = calculate_selection_patterns(sample_echiquiers)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        assert joueur_a.iloc[0]["nb_echiquiers_differents"] == 2
