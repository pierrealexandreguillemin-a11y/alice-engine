"""Fixtures Features Pipeline - ISO 29119.

Document ID: ALICE-TEST-FEATURES-PIPELINE-CONFTEST
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
def sample_history() -> pd.DataFrame:
    """Fixture avec historique complet pour pipeline."""
    base = [
        {
            "saison": 2025,
            "competition": "Interclubs",
            "division": "Nationale 1",
            "groupe": "Groupe A",
            "ronde": 1,
            "echiquier": 1,
            "equipe_dom": "Club A",
            "equipe_ext": "Club B",
            "blanc_nom": "Joueur 1",
            "blanc_club": "Club A",
            "blanc_elo": 2200,
            "noir_nom": "Joueur 2",
            "noir_club": "Club B",
            "noir_elo": 2150,
            "resultat_blanc": 1.0,
            "resultat_noir": 0.0,
            "type_resultat": "victoire_blanc",
            "date": "2025-01-15",
            "score_dom": 4,
            "score_ext": 2,
        },
        {
            "saison": 2025,
            "competition": "Interclubs",
            "division": "Nationale 1",
            "groupe": "Groupe A",
            "ronde": 1,
            "echiquier": 2,
            "equipe_dom": "Club A",
            "equipe_ext": "Club B",
            "blanc_nom": "Joueur 3",
            "blanc_club": "Club A",
            "blanc_elo": 2100,
            "noir_nom": "Joueur 4",
            "noir_club": "Club B",
            "noir_elo": 2080,
            "resultat_blanc": 0.5,
            "resultat_noir": 0.5,
            "type_resultat": "nulle",
            "date": "2025-01-15",
            "score_dom": 4,
            "score_ext": 2,
        },
        {
            "saison": 2025,
            "competition": "Interclubs",
            "division": "Nationale 1",
            "groupe": "Groupe A",
            "ronde": 2,
            "echiquier": 1,
            "equipe_dom": "Club B",
            "equipe_ext": "Club A",
            "blanc_nom": "Joueur 2",
            "blanc_club": "Club B",
            "blanc_elo": 2150,
            "noir_nom": "Joueur 1",
            "noir_club": "Club A",
            "noir_elo": 2200,
            "resultat_blanc": 0.0,
            "resultat_noir": 1.0,
            "type_resultat": "victoire_noir",
            "date": "2025-01-22",
            "score_dom": 2,
            "score_ext": 4,
        },
    ]
    extra = [
        {
            "saison": 2025,
            "competition": "Interclubs",
            "division": "Nationale 1",
            "groupe": "Groupe A",
            "ronde": r,
            "echiquier": 1,
            "equipe_dom": "Club A",
            "equipe_ext": "Club C",
            "blanc_nom": "Joueur 1",
            "blanc_club": "Club A",
            "blanc_elo": 2200,
            "noir_nom": "Joueur 5",
            "noir_club": "Club C",
            "noir_elo": 2000,
            "resultat_blanc": 1.0,
            "resultat_noir": 0.0,
            "type_resultat": "victoire_blanc",
            "date": f"2025-02-{r:02d}",
            "score_dom": 5,
            "score_ext": 1,
        }
        for r in range(3, 9)
    ]
    return pd.DataFrame(base + extra)


@pytest.fixture
def sample_history_played(sample_history: pd.DataFrame) -> pd.DataFrame:
    """Fixture avec parties jouees uniquement (sans forfaits)."""
    return sample_history[
        ~sample_history["type_resultat"].isin(
            ["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"]
        )
    ].copy()


@pytest.fixture
def sample_target() -> pd.DataFrame:
    """Fixture DataFrame cible pour merge."""
    return pd.DataFrame(
        [
            {
                "saison": 2025,
                "ronde": 10,
                "echiquier": 1,
                "equipe_dom": "Club A",
                "equipe_ext": "Club D",
                "blanc_nom": "Joueur 1",
                "blanc_club": "Club A",
                "noir_nom": "Joueur 6",
                "noir_club": "Club D",
            }
        ]
    )
