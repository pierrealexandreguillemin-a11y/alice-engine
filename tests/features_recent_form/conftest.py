"""Fixtures Features Recent Form - ISO 29119.

Document ID: ALICE-TEST-FEATURES-RECENT-FORM-CONFTEST
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
def sample_matches_for_form() -> pd.DataFrame:
    """Fixture avec matchs pour tester le calcul de forme recente."""
    return pd.DataFrame(
        [
            # Joueur A - 6 matchs avec blanc (forme croissante)
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 1",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
                "date": "2025-01-01",
            },
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 2",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
                "date": "2025-01-08",
            },
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 3",
                "resultat_blanc": 0.5,
                "resultat_noir": 0.5,
                "type_resultat": "nulle",
                "date": "2025-01-15",
            },
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 4",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-22",
            },
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 5",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-29",
            },
            {
                "blanc_nom": "Joueur A",
                "noir_nom": "Adversaire 6",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-02-05",
            },
            # Joueur B - 5 matchs avec noir (forme decroissante)
            {
                "blanc_nom": "Adversaire 7",
                "noir_nom": "Joueur B",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
                "date": "2025-01-01",
            },
            {
                "blanc_nom": "Adversaire 8",
                "noir_nom": "Joueur B",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
                "date": "2025-01-08",
            },
            {
                "blanc_nom": "Adversaire 9",
                "noir_nom": "Joueur B",
                "resultat_blanc": 0.5,
                "resultat_noir": 0.5,
                "type_resultat": "nulle",
                "date": "2025-01-15",
            },
            {
                "blanc_nom": "Adversaire 10",
                "noir_nom": "Joueur B",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-22",
            },
            {
                "blanc_nom": "Adversaire 11",
                "noir_nom": "Joueur B",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-29",
            },
            # Joueur C - seulement 3 matchs (pas assez pour window=5)
            {
                "blanc_nom": "Joueur C",
                "noir_nom": "Adversaire 12",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-01",
            },
            {
                "blanc_nom": "Joueur C",
                "noir_nom": "Adversaire 13",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-08",
            },
            {
                "blanc_nom": "Joueur C",
                "noir_nom": "Adversaire 14",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-15",
            },
        ]
    )


@pytest.fixture
def matches_with_forfaits() -> pd.DataFrame:
    """Fixture avec forfaits pour tester filtrage."""
    return pd.DataFrame(
        [
            # Matchs joues
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 1",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-01",
            },
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 2",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-08",
            },
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 3",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-15",
            },
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 4",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-22",
            },
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 5",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-29",
            },
            # Forfaits (doivent etre ignores)
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 6",
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "forfait_noir",
                "date": "2025-02-05",
            },
            {
                "blanc_nom": "Adversaire 7",
                "noir_nom": "Joueur D",
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "forfait_blanc",
                "date": "2025-02-12",
            },
            {
                "blanc_nom": "Joueur D",
                "noir_nom": "Adversaire 8",
                "resultat_blanc": 0.0,
                "resultat_noir": 0.0,
                "type_resultat": "non_joue",
                "date": "2025-02-19",
            },
        ]
    )
