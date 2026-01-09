"""Module: test_features_ali.py - Tests ALI Features.

Tests unitaires pour les features ALI (Adversarial Lineup Inference).

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (unit tests, boundary values)
- ISO/IEC 5259:2024 - Data Quality for ML (feature validation)
- ISO/IEC 25010:2023 - System Quality (testability)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

import pandas as pd
import pytest

from scripts.features.ali import calculate_presence_features, calculate_selection_patterns

# ==============================================================================
# FIXTURES (ISO 29119-3: Test Design)
# ==============================================================================


@pytest.fixture
def sample_echiquiers() -> pd.DataFrame:
    """Fixture DataFrame échiquiers pour tests."""
    return pd.DataFrame(
        [
            # Joueur A: présent 7/9 rondes (régulier, titulaire)
            {"saison": 2025, "ronde": 1, "blanc_nom": "Joueur A", "noir_nom": "X", "echiquier": 1},
            {"saison": 2025, "ronde": 2, "blanc_nom": "Joueur A", "noir_nom": "Y", "echiquier": 1},
            {"saison": 2025, "ronde": 3, "blanc_nom": "Joueur A", "noir_nom": "Z", "echiquier": 1},
            {"saison": 2025, "ronde": 4, "blanc_nom": "Joueur A", "noir_nom": "W", "echiquier": 1},
            {"saison": 2025, "ronde": 5, "blanc_nom": "Joueur A", "noir_nom": "V", "echiquier": 1},
            {"saison": 2025, "ronde": 6, "blanc_nom": "Joueur A", "noir_nom": "U", "echiquier": 2},
            {"saison": 2025, "ronde": 7, "blanc_nom": "Joueur A", "noir_nom": "T", "echiquier": 1},
            # Joueur B: présent 4/9 rondes (rotation)
            {"saison": 2025, "ronde": 1, "blanc_nom": "X", "noir_nom": "Joueur B", "echiquier": 3},
            {"saison": 2025, "ronde": 3, "blanc_nom": "X", "noir_nom": "Joueur B", "echiquier": 4},
            {"saison": 2025, "ronde": 5, "blanc_nom": "X", "noir_nom": "Joueur B", "echiquier": 5},
            {"saison": 2025, "ronde": 7, "blanc_nom": "X", "noir_nom": "Joueur B", "echiquier": 6},
            # Joueur C: présent 2/9 rondes (rare, remplaçant)
            {"saison": 2025, "ronde": 8, "blanc_nom": "Joueur C", "noir_nom": "S", "echiquier": 8},
            {"saison": 2025, "ronde": 9, "blanc_nom": "Joueur C", "noir_nom": "R", "echiquier": 8},
            # Joueur D: polyvalent (plusieurs échiquiers)
            {"saison": 2025, "ronde": 1, "blanc_nom": "Joueur D", "noir_nom": "Q", "echiquier": 1},
            {"saison": 2025, "ronde": 2, "blanc_nom": "Joueur D", "noir_nom": "P", "echiquier": 4},
            {"saison": 2025, "ronde": 3, "blanc_nom": "Joueur D", "noir_nom": "O", "echiquier": 7},
            {"saison": 2025, "ronde": 4, "blanc_nom": "Joueur D", "noir_nom": "N", "echiquier": 2},
            {"saison": 2025, "ronde": 5, "blanc_nom": "Joueur D", "noir_nom": "M", "echiquier": 5},
            {"saison": 2025, "ronde": 6, "blanc_nom": "Joueur D", "noir_nom": "L", "echiquier": 8},
            {"saison": 2025, "ronde": 7, "blanc_nom": "Joueur D", "noir_nom": "K", "echiquier": 3},
            {"saison": 2025, "ronde": 8, "blanc_nom": "Joueur D", "noir_nom": "J", "echiquier": 6},
        ]
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Fixture DataFrame vide."""
    return pd.DataFrame()


# ==============================================================================
# TESTS: calculate_presence_features
# ==============================================================================


class TestCalculatePresenceFeatures:
    """Tests pour calculate_presence_features()."""

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test avec DataFrame vide retourne DataFrame vide."""
        result = calculate_presence_features(empty_df)
        assert result.empty

    def test_returns_dataframe(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test retourne un DataFrame."""
        result = calculate_presence_features(sample_echiquiers)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test colonnes requises présentes."""
        result = calculate_presence_features(sample_echiquiers)

        required_cols = [
            "joueur_nom",
            "saison",
            "taux_presence_saison",
            "derniere_presence",
            "nb_rondes_jouees",
            "nb_rondes_total",
            "regularite",
        ]
        for col in required_cols:
            assert col in result.columns, f"Colonne manquante: {col}"

    def test_regularite_regulier_threshold(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test seuil régularité 'regulier' (>70%)."""
        result = calculate_presence_features(sample_echiquiers)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        assert len(joueur_a) == 1
        # 7 rondes sur 9 = 77.8% > 70%
        assert joueur_a.iloc[0]["regularite"] == "regulier"

    def test_regularite_rare_threshold(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test seuil régularité 'rare' (<30%)."""
        result = calculate_presence_features(sample_echiquiers)

        joueur_c = result[result["joueur_nom"] == "Joueur C"]
        assert len(joueur_c) == 1
        # 2 rondes sur 9 = 22.2% < 30%
        assert joueur_c.iloc[0]["regularite"] == "rare"

    def test_regularite_occasionnel_threshold(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test seuil régularité 'occasionnel' (30-70%)."""
        result = calculate_presence_features(sample_echiquiers)

        joueur_b = result[result["joueur_nom"] == "Joueur B"]
        assert len(joueur_b) == 1
        # 4 rondes sur 9 = 44.4% → occasionnel
        assert joueur_b.iloc[0]["regularite"] == "occasionnel"

    def test_taux_presence_calculation(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test calcul taux de présence correct."""
        result = calculate_presence_features(sample_echiquiers)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        # 7 rondes jouées sur 9 total
        assert joueur_a.iloc[0]["nb_rondes_jouees"] == 7
        assert joueur_a.iloc[0]["taux_presence_saison"] == pytest.approx(7 / 9, rel=0.01)

    def test_derniere_presence_calculation(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test calcul dernière présence."""
        result = calculate_presence_features(sample_echiquiers)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        # Dernière ronde jouée = 7, max = 9, donc 2 rondes d'absence
        assert joueur_a.iloc[0]["derniere_presence"] == 2

    def test_filter_by_saison(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test filtrage par saison."""
        result = calculate_presence_features(sample_echiquiers, saison=2025)
        assert len(result) > 0
        assert all(result["saison"] == 2025)

    def test_filter_nonexistent_saison(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test filtrage saison inexistante retourne vide."""
        result = calculate_presence_features(sample_echiquiers, saison=2020)
        assert result.empty

    def test_deduplication_blanc_noir(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test déduplication joueurs jouant blanc et noir."""
        result = calculate_presence_features(sample_echiquiers)
        # Chaque joueur ne doit apparaître qu'une fois par saison
        assert result.groupby(["joueur_nom", "saison"]).size().max() == 1


# ==============================================================================
# TESTS: calculate_selection_patterns
# ==============================================================================


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
        # Joueur A joue 6 fois échiquier 1, 1 fois échiquier 2
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
        # Joueur D joue sur 8 échiquiers différents
        assert joueur_d.iloc[0]["flexibilite_echiquier"] > 2

    def test_nb_echiquiers_differents(self, sample_echiquiers: pd.DataFrame) -> None:
        """Test calcul nombre échiquiers différents."""
        result = calculate_selection_patterns(sample_echiquiers)

        joueur_a = result[result["joueur_nom"] == "Joueur A"]
        # Joueur A joue sur 2 échiquiers différents (1 et 2)
        assert joueur_a.iloc[0]["nb_echiquiers_differents"] == 2


# ==============================================================================
# TESTS: Edge Cases (ISO 29119-4: Boundary Value Analysis)
# ==============================================================================


class TestALIEdgeCases:
    """Tests edge cases pour features ALI."""

    def test_single_game_player(self) -> None:
        """Test joueur avec une seule partie."""
        df = pd.DataFrame(
            [{"saison": 2025, "ronde": 1, "blanc_nom": "Solo", "noir_nom": "X", "echiquier": 1}]
        )

        presence = calculate_presence_features(df)
        patterns = calculate_selection_patterns(df)

        assert len(presence) >= 1
        assert len(patterns) >= 1
        # Flexibilité = 0 pour une seule partie
        solo = patterns[patterns["joueur_nom"] == "Solo"]
        if not solo.empty:
            assert solo.iloc[0]["flexibilite_echiquier"] == 0.0

    def test_boundary_70_percent(self) -> None:
        """Test frontière exacte 70% (7 sur 10 rondes)."""
        # 7/10 = 70% exactement - doit être occasionnel (not > 0.7)
        df = pd.DataFrame(
            [
                {"saison": 2025, "ronde": i, "blanc_nom": "Border", "noir_nom": "X", "echiquier": 1}
                for i in range(1, 8)
            ]  # 7 rondes
            + [
                {"saison": 2025, "ronde": i, "blanc_nom": "Other", "noir_nom": "Y", "echiquier": 2}
                for i in range(1, 11)
            ]  # 10 rondes total
        )

        result = calculate_presence_features(df)
        border = result[result["joueur_nom"] == "Border"]
        # 7/10 = 70% exactement → occasionnel (seuil strict >70%)
        assert border.iloc[0]["regularite"] == "occasionnel"

    def test_boundary_30_percent(self) -> None:
        """Test frontière exacte 30% (3 sur 10 rondes)."""
        # 3/10 = 30% exactement - doit être occasionnel (>= 0.3)
        df = pd.DataFrame(
            [
                {"saison": 2025, "ronde": i, "blanc_nom": "Border", "noir_nom": "X", "echiquier": 1}
                for i in range(1, 4)
            ]  # 3 rondes
            + [
                {"saison": 2025, "ronde": i, "blanc_nom": "Other", "noir_nom": "Y", "echiquier": 2}
                for i in range(1, 11)
            ]  # 10 rondes total
        )

        result = calculate_presence_features(df)
        border = result[result["joueur_nom"] == "Border"]
        # 3/10 = 30% exactement → occasionnel (seuil >= 0.3)
        assert border.iloc[0]["regularite"] == "occasionnel"

    def test_multiple_seasons(self) -> None:
        """Test avec plusieurs saisons."""
        df = pd.DataFrame(
            [
                {"saison": 2024, "ronde": 1, "blanc_nom": "A", "noir_nom": "X", "echiquier": 1},
                {"saison": 2025, "ronde": 1, "blanc_nom": "A", "noir_nom": "Y", "echiquier": 2},
            ]
        )

        result = calculate_presence_features(df)
        # Doit avoir 2 entrées pour joueur A (une par saison)
        a_entries = result[result["joueur_nom"] == "A"]
        assert len(a_entries) == 2
        assert set(a_entries["saison"].unique()) == {2024, 2025}
