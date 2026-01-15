"""Fixtures CE Features - ISO 29119.

Document ID: ALICE-TEST-CE-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_standings() -> pd.DataFrame:
    """Fixture DataFrame classement pour tests."""
    return pd.DataFrame(
        [
            {
                "equipe": "Leader",
                "saison": 2025,
                "ronde": 7,
                "position": 1,
                "points_cumules": 14,
                "nb_equipes": 8,
                "ecart_premier": 0,
                "ecart_dernier": 10,
            },
            {
                "equipe": "Dauphin",
                "saison": 2025,
                "ronde": 7,
                "position": 2,
                "points_cumules": 12,
                "nb_equipes": 8,
                "ecart_premier": 2,
                "ecart_dernier": 8,
            },
            {
                "equipe": "Milieu",
                "saison": 2025,
                "ronde": 7,
                "position": 4,
                "points_cumules": 8,
                "nb_equipes": 8,
                "ecart_premier": 6,
                "ecart_dernier": 4,
            },
            {
                "equipe": "Danger",
                "saison": 2025,
                "ronde": 7,
                "position": 6,
                "points_cumules": 5,
                "nb_equipes": 8,
                "ecart_premier": 9,
                "ecart_dernier": 1,
            },
            {
                "equipe": "Relegable",
                "saison": 2025,
                "ronde": 7,
                "position": 7,
                "points_cumules": 4,
                "nb_equipes": 8,
                "ecart_premier": 10,
                "ecart_dernier": 0,
            },
            {
                "equipe": "Dernier",
                "saison": 2025,
                "ronde": 7,
                "position": 8,
                "points_cumules": 2,
                "nb_equipes": 8,
                "ecart_premier": 12,
                "ecart_dernier": 0,
            },
        ]
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Fixture DataFrame vide."""
    return pd.DataFrame()


@pytest.fixture
def sample_player_stats() -> pd.DataFrame:
    """Fixture stats joueurs pour transferability."""
    return pd.DataFrame(
        [
            {
                "joueur_nom": "Expert",
                "nb_parties": 50,
                "score_moyen": 0.75,
                "elo_moyen": 2200,
                "victoires": 35,
                "nulles": 10,
                "defaites": 5,
            },
            {
                "joueur_nom": "Regular",
                "nb_parties": 30,
                "score_moyen": 0.55,
                "elo_moyen": 1800,
                "victoires": 15,
                "nulles": 5,
                "defaites": 10,
            },
            {
                "joueur_nom": "Novice",
                "nb_parties": 5,
                "score_moyen": 0.4,
                "elo_moyen": 1500,
                "victoires": 2,
                "nulles": 0,
                "defaites": 3,
            },
        ]
    )


@pytest.fixture
def sample_team_rosters() -> pd.DataFrame:
    """Fixture compositions equipes."""
    return pd.DataFrame(
        [
            {"equipe": "TeamA", "joueur_nom": "Expert", "echiquier": 1},
            {"equipe": "TeamA", "joueur_nom": "Regular", "echiquier": 2},
            {"equipe": "TeamB", "joueur_nom": "Novice", "echiquier": 1},
        ]
    )
