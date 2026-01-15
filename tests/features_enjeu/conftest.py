"""Fixtures Features Enjeu - ISO 29119.

Document ID: ALICE-TEST-FEATURES-ENJEU-CONFTEST
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
    """Fixture avec classement pour tester zones enjeu."""
    return pd.DataFrame(
        [
            {
                "equipe": "N1 Equipe A",
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 5,
                "position": 1,
                "points_cumules": 8,
                "nb_equipes": 10,
                "ecart_premier": 0,
                "ecart_dernier": 6,
            },
            {
                "equipe": "N1 Equipe B",
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 5,
                "position": 5,
                "points_cumules": 5,
                "nb_equipes": 10,
                "ecart_premier": 3,
                "ecart_dernier": 3,
            },
            {
                "equipe": "N1 Equipe C",
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 5,
                "position": 9,
                "points_cumules": 2,
                "nb_equipes": 10,
                "ecart_premier": 6,
                "ecart_dernier": 0,
            },
            {
                "equipe": "N4 Equipe D",
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 4",
                "groupe": "Groupe B",
                "ronde": 5,
                "position": 1,
                "points_cumules": 10,
                "nb_equipes": 8,
                "ecart_premier": 0,
                "ecart_dernier": 8,
            },
            {
                "equipe": "N4 Equipe E",
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 4",
                "groupe": "Groupe B",
                "ronde": 5,
                "position": 8,
                "points_cumules": 2,
                "nb_equipes": 8,
                "ecart_premier": 8,
                "ecart_dernier": 0,
            },
        ]
    )


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """Fixture avec matchs pour test fallback."""
    return pd.DataFrame(
        [
            {
                "equipe_dom": "N1 Club A",
                "equipe_ext": "N1 Club B",
                "saison": 2025,
                "ronde": 1,
            },
            {
                "equipe_dom": "N1 Club A",
                "equipe_ext": "N1 Club C",
                "saison": 2025,
                "ronde": 2,
            },
            {
                "equipe_dom": "N4 Club D",
                "equipe_ext": "N4 Club E",
                "saison": 2025,
                "ronde": 1,
            },
        ]
    )
