"""Module: test_feature_engineering.py - Tests Feature Engineering.

Tests unitaires et edge cases pour l'extraction des features ML.
Couvre tous les modules: reliability, performance, standings, advanced.

ISO Compliance:
- ISO/IEC 29119 - Software Testing (unit tests, edge cases)
- ISO/IEC 5259:2024 - Data Quality for ML (feature validation)
- ISO/IEC 25010 - System Quality (testabilite)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

import numpy as np
import pandas as pd
import pytest

from scripts.features.advanced import (
    calculate_elo_trajectory,
    calculate_head_to_head,
    calculate_pressure_performance,
)
from scripts.features.performance import (
    calculate_color_performance,
    calculate_recent_form,
)

# Import from refactored modules
from scripts.features.reliability import (
    extract_club_reliability,
    extract_player_reliability,
)
from scripts.features.standings import (
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
            # Ronde 2: A vs C (3-3 nul), B bat D (4-2)
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

    # Joueur Y: équilibré (5 victoires blancs sur 10, 5 victoires noirs sur 10)
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

    # Joueur Z: seulement blancs (pas assez de noirs)
    for i in range(10):
        games.append(
            {
                "blanc_nom": "Joueur Z",
                "noir_nom": f"Adv Z{i}",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            }
        )
    # Seulement 2 parties noirs
    for i in range(2):
        games.append(
            {
                "blanc_nom": f"Adv ZB{i}",
                "noir_nom": "Joueur Z",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
            }
        )

    return pd.DataFrame(games)


@pytest.fixture
def sample_h2h_games() -> pd.DataFrame:
    """Fixture pour tests H2H."""
    games = []
    # A vs B: 5 confrontations, A gagne 4
    for i in range(4):
        games.append(
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Joueur B",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            }
        )
    games.append(
        {
            "blanc_nom": "Joueur B",
            "noir_nom": "Joueur A",
            "resultat_blanc": 1.0,
            "resultat_noir": 0.0,
            "type_resultat": "victoire_blanc",
        }
    )
    return pd.DataFrame(games)


@pytest.fixture
def sample_dated_games() -> pd.DataFrame:
    """Fixture avec dates pour tests fatigue/trajectoire."""
    return pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "blanc_nom": "Joueur Test",
                "noir_nom": "Adv 1",
                "blanc_elo": 1500,
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            },
            {
                "date": "2025-01-02",  # 1 jour après = fatigué
                "blanc_nom": "Joueur Test",
                "noir_nom": "Adv 2",
                "blanc_elo": 1510,
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            },
            {
                "date": "2025-01-10",  # 8 jours après = reposé
                "blanc_nom": "Joueur Test",
                "noir_nom": "Adv 3",
                "blanc_elo": 1520,
                "resultat_blanc": 0.5,
                "resultat_noir": 0.5,
                "type_resultat": "nulle",
            },
            {
                "date": "2025-01-15",  # 5 jours après = normal
                "blanc_nom": "Joueur Test",
                "noir_nom": "Adv 4",
                "blanc_elo": 1530,
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            },
            {
                "date": "2025-01-20",
                "blanc_nom": "Joueur Test",
                "noir_nom": "Adv 5",
                "blanc_elo": 1550,
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            },
            {
                "date": "2025-01-25",
                "blanc_nom": "Joueur Test",
                "noir_nom": "Adv 6",
                "blanc_elo": 1560,
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            },
        ]
    )


# ==============================================================================
# TESTS CALCULATE_STANDINGS
# ==============================================================================


class TestCalculateStandings:
    """Tests pour calculate_standings()."""

    def test_standings_basic(self, sample_matches: pd.DataFrame) -> None:
        """Test classement basique après 2 rondes."""
        result = calculate_standings(sample_matches)

        assert not result.empty
        assert "position" in result.columns
        assert "points_cumules" in result.columns

    def test_standings_points_correct(self, sample_matches: pd.DataFrame) -> None:
        """Test points corrects (2 pts victoire, 1 pt nul)."""
        result = calculate_standings(sample_matches)

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
        """Test que les positions sont ordonnées correctement."""
        result = calculate_standings(sample_matches)
        r2 = result[result["ronde"] == 2]

        d_pos = r2[r2["equipe"] == "Equipe D"]["position"].values[0]
        assert d_pos == 4

        a_pos = r2[r2["equipe"] == "Equipe A"]["position"].values[0]
        c_pos = r2[r2["equipe"] == "Equipe C"]["position"].values[0]
        assert a_pos in [1, 2]
        assert c_pos in [1, 2]

    def test_standings_ecart_premier(self, sample_matches: pd.DataFrame) -> None:
        """Test ecart_premier calculé correctement."""
        result = calculate_standings(sample_matches)
        r2 = result[result["ronde"] == 2]

        d_ecart = r2[r2["equipe"] == "Equipe D"]["ecart_premier"].values[0]
        assert d_ecart == 3

        a_ecart = r2[r2["equipe"] == "Equipe A"]["ecart_premier"].values[0]
        assert a_ecart == 0

    def test_standings_nb_equipes(self, sample_matches: pd.DataFrame) -> None:
        """Test nb_equipes correct."""
        result = calculate_standings(sample_matches)
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

    def test_standings_has_tiebreaker_columns(self, sample_matches: pd.DataFrame) -> None:
        """Test que les colonnes tie-breaker sont présentes."""
        result = calculate_standings(sample_matches)

        assert "victoires" in result.columns
        assert "diff_points_matchs" in result.columns


# ==============================================================================
# TESTS EXTRACT_TEAM_ENJEU_FEATURES
# ==============================================================================


class TestExtractTeamEnjeuFeatures:
    """Tests pour extract_team_enjeu_features()."""

    def test_enjeu_uses_real_position(self, sample_matches: pd.DataFrame) -> None:
        """Test que zone_enjeu utilise position réelle."""
        from scripts.features import calculate_standings

        standings = calculate_standings(sample_matches)
        result = extract_team_enjeu_features(sample_matches, standings)

        assert not result.empty
        assert "position" in result.columns
        assert "zone_enjeu" in result.columns

        d_zone = result[result["equipe"] == "Equipe D"]["zone_enjeu"].unique()
        assert len(d_zone) > 0

    def test_enjeu_has_ecarts(self, sample_matches: pd.DataFrame) -> None:
        """Test que ecart_premier et ecart_dernier sont présents."""
        from scripts.features import calculate_standings

        standings = calculate_standings(sample_matches)
        result = extract_team_enjeu_features(sample_matches, standings)

        assert "ecart_premier" in result.columns
        assert "ecart_dernier" in result.columns


# ==============================================================================
# TESTS CALCULATE_COLOR_PERFORMANCE (FIXED)
# ==============================================================================


class TestCalculateColorPerformance:
    """Tests pour calculate_color_performance() - version corrigée."""

    def test_color_basic(self, sample_color_games: pd.DataFrame) -> None:
        """Test calcul performance couleur basique."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        assert not result.empty
        assert "score_blancs" in result.columns
        assert "score_noirs" in result.columns
        assert "avantage_blancs" in result.columns
        assert "couleur_preferee" in result.columns
        assert "data_quality" in result.columns  # NEW

    def test_color_joueur_x_prefere_blanc(self, sample_color_games: pd.DataFrame) -> None:
        """Test Joueur X préfère blancs."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        x_row = result[result["joueur_nom"] == "Joueur X"]
        assert len(x_row) == 1

        assert x_row["score_blancs"].values[0] == pytest.approx(0.8, abs=0.01)
        assert x_row["score_noirs"].values[0] == pytest.approx(0.4, abs=0.01)
        assert x_row["avantage_blancs"].values[0] == pytest.approx(0.4, abs=0.01)
        assert x_row["couleur_preferee"].values[0] == "blanc"
        assert x_row["data_quality"].values[0] == "complet"

    def test_color_joueur_y_neutre(self, sample_color_games: pd.DataFrame) -> None:
        """Test Joueur Y est neutre."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        y_row = result[result["joueur_nom"] == "Joueur Y"]
        assert len(y_row) == 1

        assert y_row["score_blancs"].values[0] == pytest.approx(0.5, abs=0.01)
        assert y_row["score_noirs"].values[0] == pytest.approx(0.5, abs=0.01)
        assert abs(y_row["avantage_blancs"].values[0]) < 0.05
        assert y_row["couleur_preferee"].values[0] == "neutre"

    def test_color_min_games_filter(self, sample_color_games: pd.DataFrame) -> None:
        """Test filtre min_games fonctionne."""
        result = calculate_color_performance(sample_color_games, min_games=100)
        assert result.empty

    def test_color_counts(self, sample_color_games: pd.DataFrame) -> None:
        """Test comptage parties correct."""
        result = calculate_color_performance(sample_color_games, min_games=10)

        x_row = result[result["joueur_nom"] == "Joueur X"]
        assert x_row["nb_blancs"].values[0] == 10
        assert x_row["nb_noirs"].values[0] == 10

    def test_color_insufficient_data_no_fillna(self, sample_color_games: pd.DataFrame) -> None:
        """Test ISO 5259: pas de fillna(0.5) - données insuffisantes marquées."""
        result = calculate_color_performance(sample_color_games, min_games=10, min_per_color=5)

        # Joueur Z: 10 blancs, 2 noirs => données insuffisantes pour noirs
        z_row = result[result["joueur_nom"] == "Joueur Z"]
        assert len(z_row) == 1

        # data_quality doit indiquer le problème
        assert z_row["data_quality"].values[0] == "partiel_noirs"

        # couleur_preferee = donnees_insuffisantes (pas d'estimation!)
        assert z_row["couleur_preferee"].values[0] == "donnees_insuffisantes"

        # avantage_blancs doit être NaN (pas 0.5!)
        assert pd.isna(z_row["avantage_blancs"].values[0])

    def test_color_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_color_performance(pd.DataFrame())
        assert result.empty


# ==============================================================================
# TESTS CALCULATE_RECENT_FORM
# ==============================================================================


class TestCalculateRecentForm:
    """Tests pour calculate_recent_form()."""

    def test_form_basic(self, sample_color_games: pd.DataFrame) -> None:
        """Test calcul forme récente basique."""
        result = calculate_recent_form(sample_color_games, window=5)

        assert not result.empty
        assert "forme_recente" in result.columns
        assert "forme_tendance" in result.columns

    def test_form_tendance_values(self, sample_color_games: pd.DataFrame) -> None:
        """Test que tendance a les bonnes valeurs."""
        result = calculate_recent_form(sample_color_games, window=5)

        tendances = result["forme_tendance"].unique()
        for t in tendances:
            assert t in ["hausse", "baisse", "stable"]

    def test_form_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_recent_form(pd.DataFrame())
        assert result.empty


# ==============================================================================
# TESTS CALCULATE_HEAD_TO_HEAD
# ==============================================================================


class TestCalculateHeadToHead:
    """Tests pour calculate_head_to_head()."""

    def test_h2h_basic(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test H2H basique."""
        result = calculate_head_to_head(sample_h2h_games, min_games=3)

        assert not result.empty
        assert "joueur_a" in result.columns
        assert "joueur_b" in result.columns
        assert "avantage_a" in result.columns

    def test_h2h_scores(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test que A domine B."""
        result = calculate_head_to_head(sample_h2h_games, min_games=3)

        # A vs B: A gagne 4/5
        row = result[
            ((result["joueur_a"] == "Joueur A") & (result["joueur_b"] == "Joueur B"))
            | ((result["joueur_a"] == "Joueur B") & (result["joueur_b"] == "Joueur A"))
        ]
        assert len(row) == 1
        assert row["nb_confrontations"].values[0] == 5

    def test_h2h_min_games_filter(self, sample_h2h_games: pd.DataFrame) -> None:
        """Test filtre min_games."""
        result = calculate_head_to_head(sample_h2h_games, min_games=10)
        assert result.empty

    def test_h2h_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_head_to_head(pd.DataFrame())
        assert result.empty


# ==============================================================================
# TESTS CALCULATE_ELO_TRAJECTORY
# ==============================================================================


class TestCalculateEloTrajectory:
    """Tests pour calculate_elo_trajectory()."""

    def test_trajectory_basic(self, sample_dated_games: pd.DataFrame) -> None:
        """Test trajectoire Elo basique."""
        result = calculate_elo_trajectory(sample_dated_games, window=4)

        assert not result.empty
        assert "elo_trajectory" in result.columns
        assert "momentum" in result.columns

    def test_trajectory_progression(self, sample_dated_games: pd.DataFrame) -> None:
        """Test détection progression Elo."""
        result = calculate_elo_trajectory(sample_dated_games, window=4)

        # Joueur Test: 1500 -> 1560 = +60 = progression
        row = result[result["joueur_nom"] == "Joueur Test"]
        if not row.empty:
            assert row["elo_trajectory"].values[0] == "progression"
            assert row["momentum"].values[0] > 0

    def test_trajectory_no_date(self) -> None:
        """Test sans colonne date."""
        df = pd.DataFrame({"blanc_nom": ["A"], "blanc_elo": [1500]})
        result = calculate_elo_trajectory(df)
        assert result.empty


# ==============================================================================
# TESTS CALCULATE_PRESSURE_PERFORMANCE
# ==============================================================================


class TestCalculatePressurePerformance:
    """Tests pour calculate_pressure_performance()."""

    def test_pressure_basic(self) -> None:
        """Test performance pression basique."""
        games = []
        # Matchs normaux (ronde < 7)
        for i in range(10):
            games.append(
                {
                    "ronde": 3,
                    "score_dom": 4,
                    "score_ext": 2,
                    "blanc_nom": "Joueur P",
                    "noir_nom": f"Adv {i}",
                    "resultat_blanc": 0.5,
                    "resultat_noir": 0.5,
                    "type_resultat": "nulle",
                }
            )
        # Matchs décisifs (ronde >= 7)
        for i in range(5):
            games.append(
                {
                    "ronde": 8,
                    "score_dom": 3,
                    "score_ext": 3,
                    "blanc_nom": "Joueur P",
                    "noir_nom": f"Adv D{i}",
                    "resultat_blanc": 1.0,
                    "resultat_noir": 0.0,
                    "type_resultat": "victoire_blanc",
                }
            )

        df = pd.DataFrame(games)
        result = calculate_pressure_performance(df, min_games=3)

        assert not result.empty
        p_row = result[result["joueur_nom"] == "Joueur P"]
        assert len(p_row) == 1
        # Score normal = 0.5, score pression = 1.0 => clutch
        assert p_row["pressure_type"].values[0] == "clutch"

    def test_pressure_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        result = calculate_pressure_performance(pd.DataFrame())
        assert result.empty


# ==============================================================================
# TESTS RELIABILITY FEATURES
# ==============================================================================


class TestReliabilityFeatures:
    """Tests pour features de fiabilité."""

    def test_club_reliability(self) -> None:
        """Test extract_club_reliability."""
        df = pd.DataFrame(
            [
                {"equipe_dom": "Club A", "equipe_ext": "Club B", "type_resultat": "victoire_blanc"},
                {"equipe_dom": "Club A", "equipe_ext": "Club C", "type_resultat": "forfait_blanc"},
                {"equipe_dom": "Club B", "equipe_ext": "Club A", "type_resultat": "nulle"},
            ]
        )

        result = extract_club_reliability(df)

        assert not result.empty
        assert "fiabilite_score" in result.columns

    def test_player_reliability(self) -> None:
        """Test extract_player_reliability."""
        df = pd.DataFrame(
            [
                {
                    "blanc_nom": "Player 1",
                    "noir_nom": "Player 2",
                    "type_resultat": "victoire_blanc",
                },
                {"blanc_nom": "Player 1", "noir_nom": "Player 3", "type_resultat": "forfait_blanc"},
                {"blanc_nom": "Player 2", "noir_nom": "Player 1", "type_resultat": "nulle"},
            ]
        )

        result = extract_player_reliability(df)

        assert not result.empty
        assert "taux_presence" in result.columns
        assert "joueur_fantome" in result.columns

    def test_reliability_empty_df(self) -> None:
        """Test avec DataFrame vide."""
        assert extract_club_reliability(pd.DataFrame()).empty
        assert extract_player_reliability(pd.DataFrame()).empty


# ==============================================================================
# TESTS EDGE CASES ISO 29119
# ==============================================================================


class TestEdgeCases:
    """Tests edge cases ISO 29119."""

    def test_nan_scores(self) -> None:
        """Test avec scores NaN."""
        df = pd.DataFrame(
            [
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 1,
                    "equipe_dom": "A",
                    "equipe_ext": "B",
                    "score_dom": np.nan,
                    "score_ext": np.nan,
                }
            ]
        )
        # Ne doit pas crasher
        result = calculate_standings(df)
        # Peut être vide ou ignorer les NaN
        assert isinstance(result, pd.DataFrame)

    def test_single_ronde(self) -> None:
        """Test avec une seule ronde."""
        df = pd.DataFrame(
            [
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 1,
                    "equipe_dom": "A",
                    "equipe_ext": "B",
                    "score_dom": 4,
                    "score_ext": 2,
                }
            ]
        )
        result = calculate_standings(df)
        assert not result.empty
        assert len(result) == 2  # 2 équipes

    def test_multi_saison(self, sample_matches: pd.DataFrame) -> None:
        """Test avec plusieurs saisons."""
        df2 = sample_matches.copy()
        df2["saison"] = 2024

        df_combined = pd.concat([sample_matches, df2])
        result = calculate_standings(df_combined)

        assert not result.empty
        assert 2024 in result["saison"].values
        assert 2025 in result["saison"].values

    def test_tiebreaker_same_points(self) -> None:
        """Test tie-breaker quand même nombre de points."""
        df = pd.DataFrame(
            [
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 1,
                    "equipe_dom": "Equipe A",
                    "equipe_ext": "Equipe B",
                    "score_dom": 4,
                    "score_ext": 2,  # A gagne
                },
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 1,
                    "equipe_dom": "Equipe C",
                    "equipe_ext": "Equipe D",
                    "score_dom": 4,
                    "score_ext": 2,  # C gagne
                },
                # Ronde 2: A vs C (A gagne), B vs D (B gagne)
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 2,
                    "equipe_dom": "Equipe A",
                    "equipe_ext": "Equipe C",
                    "score_dom": 5,
                    "score_ext": 1,  # A gagne avec meilleure diff
                },
                {
                    "saison": 2025,
                    "competition": "Test",
                    "division": "N1",
                    "groupe": "A",
                    "ronde": 2,
                    "equipe_dom": "Equipe B",
                    "equipe_ext": "Equipe D",
                    "score_dom": 4,
                    "score_ext": 2,  # B gagne
                },
            ]
        )

        result = calculate_standings(df)
        r2 = result[result["ronde"] == 2]

        # A: 4 pts, C: 2 pts, B: 2 pts, D: 0 pts
        # B et C ont 2 pts mais positions stables grâce au tie-breaker
        a_pos = r2[r2["equipe"] == "Equipe A"]["position"].values[0]
        assert a_pos == 1

        b_pos = r2[r2["equipe"] == "Equipe B"]["position"].values[0]
        c_pos = r2[r2["equipe"] == "Equipe C"]["position"].values[0]
        # Les deux ont 2 pts, mais un doit être 2e et l'autre 3e
        assert {b_pos, c_pos} == {2, 3}


# ==============================================================================
# TESTS TEMPORAL_SPLIT - scripts/feature_engineering.py
# ==============================================================================


class TestTemporalSplit:
    """Tests pour temporal_split() du module principal feature_engineering."""

    def test_temporal_split_basic(self) -> None:
        """Test split temporel basique."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023, 2024, 2025],
                "data": ["a", "b", "c", "d", "e", "f"],
            }
        )

        train, valid, test = temporal_split(df, train_end=2022, valid_end=2023)

        assert len(train) == 3  # 2020, 2021, 2022
        assert len(valid) == 1  # 2023
        assert len(test) == 2  # 2024, 2025

    def test_temporal_split_train_contains_correct_years(self) -> None:
        """Test que train contient les bonnes années."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023, 2024],
                "value": range(5),
            }
        )

        train, _, _ = temporal_split(df, train_end=2022, valid_end=2023)

        assert set(train["saison"].unique()) == {2020, 2021, 2022}

    def test_temporal_split_valid_contains_correct_years(self) -> None:
        """Test que valid contient les bonnes années."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023, 2024],
                "value": range(5),
            }
        )

        _, valid, _ = temporal_split(df, train_end=2022, valid_end=2023)

        assert set(valid["saison"].unique()) == {2023}

    def test_temporal_split_test_contains_correct_years(self) -> None:
        """Test que test contient les bonnes années."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023, 2024, 2025],
                "value": range(6),
            }
        )

        _, _, test = temporal_split(df, train_end=2022, valid_end=2023)

        assert set(test["saison"].unique()) == {2024, 2025}

    def test_temporal_split_empty_valid(self) -> None:
        """Test avec valid vide (pas de données pour cette période)."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2025],
                "value": range(3),
            }
        )

        train, valid, test = temporal_split(df, train_end=2022, valid_end=2023)

        assert len(train) == 2
        assert len(valid) == 0  # Pas de 2023
        assert len(test) == 1

    def test_temporal_split_empty_test(self) -> None:
        """Test avec test vide."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023],
                "value": range(4),
            }
        )

        _, _, test = temporal_split(df, train_end=2022, valid_end=2023)

        assert len(test) == 0

    def test_temporal_split_custom_boundaries(self) -> None:
        """Test avec boundaries personnalisées."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2018, 2019, 2020, 2021, 2022],
                "value": range(5),
            }
        )

        train, valid, test = temporal_split(df, train_end=2019, valid_end=2020)

        assert len(train) == 2  # 2018, 2019
        assert len(valid) == 1  # 2020
        assert len(test) == 2  # 2021, 2022

    def test_temporal_split_no_data_leakage(self) -> None:
        """Test ISO 5259: pas de data leakage entre splits."""
        from scripts.feature_engineering import temporal_split

        df = pd.DataFrame(
            {
                "saison": [2020, 2021, 2022, 2023, 2024],
                "value": range(5),
            }
        )

        train, valid, test = temporal_split(df, train_end=2022, valid_end=2023)

        # Vérifier qu'il n'y a pas de chevauchement
        train_years = set(train["saison"].unique())
        valid_years = set(valid["saison"].unique())
        test_years = set(test["saison"].unique())

        assert train_years.isdisjoint(valid_years)
        assert train_years.isdisjoint(test_years)
        assert valid_years.isdisjoint(test_years)
