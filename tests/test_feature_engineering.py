# tests/test_feature_engineering.py
"""Tests pour feature_engineering.py - ISO 29119.

Tests unitaires pour l'extraction des features ML.
"""

import pandas as pd
import pytest

from scripts.feature_engineering import (
    calculate_color_performance,
    calculate_standings,
    extract_team_enjeu_features,
)

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """Fixture avec matchs simples pour test classement."""
    return pd.DataFrame(
        [
            # Ronde 1: A bat B (4-2), C bat D (5-1)
            {
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 1,
                "equipe_dom": "Equipe A",
                "equipe_ext": "Equipe B",
                "score_dom": 4,
                "score_ext": 2,
                "echiquier": 1,
                "blanc_nom": "Joueur 1",
                "noir_nom": "Joueur 2",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            },
            {
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 1,
                "equipe_dom": "Equipe C",
                "equipe_ext": "Equipe D",
                "score_dom": 5,
                "score_ext": 1,
                "echiquier": 1,
                "blanc_nom": "Joueur 3",
                "noir_nom": "Joueur 4",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            },
            # Ronde 2: A bat C (3-3 nul), B bat D (4-2)
            {
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 2,
                "equipe_dom": "Equipe A",
                "equipe_ext": "Equipe C",
                "score_dom": 3,
                "score_ext": 3,
                "echiquier": 1,
                "blanc_nom": "Joueur 1",
                "noir_nom": "Joueur 3",
                "resultat_blanc": 0.5,
                "resultat_noir": 0.5,
                "type_resultat": "nulle",
            },
            {
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 2,
                "equipe_dom": "Equipe B",
                "equipe_ext": "Equipe D",
                "score_dom": 4,
                "score_ext": 2,
                "echiquier": 1,
                "blanc_nom": "Joueur 2",
                "noir_nom": "Joueur 4",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            },
        ]
    )


@pytest.fixture
def sample_color_games() -> pd.DataFrame:
    """Fixture avec parties pour test performance couleur."""
    games = []

    # Joueur X: 10 parties blancs (8 victoires), 10 parties noirs (4 victoires)
    # => prefere blancs
    for i in range(8):
        games.append(
            {
                "blanc_nom": "Joueur X",
                "noir_nom": f"Adversaire {i}",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            }
        )
    for i in range(2):
        games.append(
            {
                "blanc_nom": "Joueur X",
                "noir_nom": f"Adversaire {8 + i}",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
            }
        )
    for i in range(4):
        games.append(
            {
                "blanc_nom": f"Adversaire B{i}",
                "noir_nom": "Joueur X",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
            }
        )
    for i in range(6):
        games.append(
            {
                "blanc_nom": f"Adversaire B{4 + i}",
                "noir_nom": "Joueur X",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            }
        )

    # Joueur Y: equilibre (5 victoires blancs sur 10, 5 victoires noirs sur 10)
    for i in range(5):
        games.append(
            {
                "blanc_nom": "Joueur Y",
                "noir_nom": f"Adv Y{i}",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            }
        )
    for i in range(5):
        games.append(
            {
                "blanc_nom": "Joueur Y",
                "noir_nom": f"Adv Y{5 + i}",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
            }
        )
    for i in range(5):
        games.append(
            {
                "blanc_nom": f"Adv YB{i}",
                "noir_nom": "Joueur Y",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
            }
        )
    for i in range(5):
        games.append(
            {
                "blanc_nom": f"Adv YB{5 + i}",
                "noir_nom": "Joueur Y",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            }
        )

    return pd.DataFrame(games)


# ==============================================================================
# TESTS CALCULATE_STANDINGS
# ==============================================================================


class TestCalculateStandings:
    """Tests pour calculate_standings()."""

    def test_standings_basic(self, sample_matches: pd.DataFrame) -> None:
        """Test classement basique apres 2 rondes."""
        result = calculate_standings(sample_matches)

        assert not result.empty
        assert "position" in result.columns
        assert "points_cumules" in result.columns

    def test_standings_points_correct(self, sample_matches: pd.DataFrame) -> None:
        """Test points corrects (2 pts victoire, 1 pt nul)."""
        result = calculate_standings(sample_matches)

        # Apres ronde 2:
        # A: 2 (victoire R1) + 1 (nul R2) = 3 pts
        # C: 2 (victoire R1) + 1 (nul R2) = 3 pts
        # B: 0 (defaite R1) + 2 (victoire R2) = 2 pts
        # D: 0 + 0 = 0 pts

        # Filtrer ronde 2
        r2 = result[result["ronde"] == 2]

        a_pts = r2[r2["equipe"] == "Equipe A"]["points_cumules"].values[0]
        c_pts = r2[r2["equipe"] == "Equipe C"]["points_cumules"].values[0]
        b_pts = r2[r2["equipe"] == "Equipe B"]["points_cumules"].values[0]
        d_pts = r2[r2["equipe"] == "Equipe D"]["points_cumules"].values[0]

        assert a_pts == 3  # 2 + 1
        assert c_pts == 3  # 2 + 1
        assert b_pts == 2  # 0 + 2
        assert d_pts == 0  # 0 + 0

    def test_standings_position_order(self, sample_matches: pd.DataFrame) -> None:
        """Test que les positions sont ordonnees correctement."""
        result = calculate_standings(sample_matches)
        r2 = result[result["ronde"] == 2]

        # D doit etre dernier (position 4)
        d_pos = r2[r2["equipe"] == "Equipe D"]["position"].values[0]
        assert d_pos == 4

        # A ou C doit etre 1er ou 2e (memes points)
        a_pos = r2[r2["equipe"] == "Equipe A"]["position"].values[0]
        c_pos = r2[r2["equipe"] == "Equipe C"]["position"].values[0]
        assert a_pos in [1, 2]
        assert c_pos in [1, 2]

    def test_standings_ecart_premier(self, sample_matches: pd.DataFrame) -> None:
        """Test ecart_premier calcule correctement."""
        result = calculate_standings(sample_matches)
        r2 = result[result["ronde"] == 2]

        # D a 0 pts, premier a 3 pts => ecart = 3
        d_ecart = r2[r2["equipe"] == "Equipe D"]["ecart_premier"].values[0]
        assert d_ecart == 3

        # A a 3 pts, premier a 3 pts => ecart = 0
        a_ecart = r2[r2["equipe"] == "Equipe A"]["ecart_premier"].values[0]
        assert a_ecart == 0

    def test_standings_nb_equipes(self, sample_matches: pd.DataFrame) -> None:
        """Test nb_equipes correct."""
        result = calculate_standings(sample_matches)

        # 4 equipes dans ce groupe
        assert (result["nb_equipes"] == 4).all()

    def test_standings_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_standings(pd.DataFrame())
        assert result.empty

    def test_standings_missing_columns(self) -> None:
        """Test avec colonnes manquantes."""
        df = pd.DataFrame({"saison": [2025], "equipe_dom": ["A"]})
        result = calculate_standings(df)
        assert result.empty


# ==============================================================================
# TESTS EXTRACT_TEAM_ENJEU_FEATURES
# ==============================================================================


class TestExtractTeamEnjeuFeatures:
    """Tests pour extract_team_enjeu_features()."""

    def test_enjeu_uses_real_position(self, sample_matches: pd.DataFrame) -> None:
        """Test que zone_enjeu utilise position reelle."""
        result = extract_team_enjeu_features(sample_matches)

        assert not result.empty
        assert "position" in result.columns
        assert "zone_enjeu" in result.columns

        # D est dernier => zone danger ou descente
        d_zone = result[result["equipe"] == "Equipe D"]["zone_enjeu"].unique()
        assert len(d_zone) > 0

    def test_enjeu_has_ecarts(self, sample_matches: pd.DataFrame) -> None:
        """Test que ecart_premier et ecart_dernier sont presents."""
        result = extract_team_enjeu_features(sample_matches)

        assert "ecart_premier" in result.columns
        assert "ecart_dernier" in result.columns


# ==============================================================================
# TESTS CALCULATE_COLOR_PERFORMANCE
# ==============================================================================


class TestCalculateColorPerformance:
    """Tests pour calculate_color_performance()."""

    def test_color_basic(self, sample_color_games: pd.DataFrame) -> None:
        """Test calcul performance couleur basique."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        assert not result.empty
        assert "score_blancs" in result.columns
        assert "score_noirs" in result.columns
        assert "avantage_blancs" in result.columns
        assert "couleur_preferee" in result.columns

    def test_color_joueur_x_prefere_blanc(self, sample_color_games: pd.DataFrame) -> None:
        """Test Joueur X prefere blancs."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        x_row = result[result["joueur_nom"] == "Joueur X"]
        assert len(x_row) == 1

        # Score blancs = 8/10 = 0.8
        assert x_row["score_blancs"].values[0] == pytest.approx(0.8, abs=0.01)

        # Score noirs = 4/10 = 0.4
        assert x_row["score_noirs"].values[0] == pytest.approx(0.4, abs=0.01)

        # Avantage blancs = 0.8 - 0.4 = 0.4
        assert x_row["avantage_blancs"].values[0] == pytest.approx(0.4, abs=0.01)

        # Prefere blanc
        assert x_row["couleur_preferee"].values[0] == "blanc"

    def test_color_joueur_y_neutre(self, sample_color_games: pd.DataFrame) -> None:
        """Test Joueur Y est neutre."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        y_row = result[result["joueur_nom"] == "Joueur Y"]
        assert len(y_row) == 1

        # Score blancs = 5/10 = 0.5
        assert y_row["score_blancs"].values[0] == pytest.approx(0.5, abs=0.01)

        # Score noirs = 5/10 = 0.5
        assert y_row["score_noirs"].values[0] == pytest.approx(0.5, abs=0.01)

        # Avantage ~0
        assert abs(y_row["avantage_blancs"].values[0]) < 0.05

        # Neutre
        assert y_row["couleur_preferee"].values[0] == "neutre"

    def test_color_min_games_filter(self, sample_color_games: pd.DataFrame) -> None:
        """Test filtre min_games fonctionne."""
        # Avec min_games=100, personne ne devrait passer
        result = calculate_color_performance(sample_color_games, min_games=100)
        assert result.empty

    def test_color_counts(self, sample_color_games: pd.DataFrame) -> None:
        """Test comptage parties correct."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        x_row = result[result["joueur_nom"] == "Joueur X"]
        assert x_row["nb_blancs"].values[0] == 10
        assert x_row["nb_noirs"].values[0] == 10

        y_row = result[result["joueur_nom"] == "Joueur Y"]
        assert y_row["nb_blancs"].values[0] == 10
        assert y_row["nb_noirs"].values[0] == 10
