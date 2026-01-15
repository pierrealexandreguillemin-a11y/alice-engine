"""Fixtures Feature Engineering - ISO 29119.

Document ID: ALICE-TEST-FE-CONFTEST
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
                "date": "2025-01-02",
                "blanc_nom": "Joueur Test",
                "noir_nom": "Adv 2",
                "blanc_elo": 1510,
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
            },
            {
                "date": "2025-01-10",
                "blanc_nom": "Joueur Test",
                "noir_nom": "Adv 3",
                "blanc_elo": 1520,
                "resultat_blanc": 0.5,
                "resultat_noir": 0.5,
                "type_resultat": "nulle",
            },
            {
                "date": "2025-01-15",
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
