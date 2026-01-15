"""Fixtures FFE Rules Features - ISO 29119.

Document ID: ALICE-TEST-FFE-RULES-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import pytest

from scripts.ffe_rules_features import Joueur, Sexe


@pytest.fixture
def joueurs_a02() -> list[Joueur]:
    """Composition valide pour A02."""
    return [
        Joueur(1, "Joueur 1", 2400, Sexe.MASCULIN, "FRA", False),
        Joueur(2, "Joueur 2", 2300, Sexe.MASCULIN, "FRA", False),
        Joueur(3, "Joueur 3", 2200, Sexe.MASCULIN, "FRA", False),
        Joueur(4, "Joueur 4", 2100, Sexe.FEMININ, "FRA", False),
        Joueur(5, "Joueur 5", 2000, Sexe.MASCULIN, "FRA", False),
        Joueur(6, "Joueur 6", 1900, Sexe.MASCULIN, "FRA", False),
        Joueur(7, "Joueur 7", 1800, Sexe.MASCULIN, "FRA", True),
        Joueur(8, "Joueur 8", 1700, Sexe.MASCULIN, "FRA", False),
    ]
