"""Tests Noyau - ISO 25012.

Document ID: ALICE-TEST-FFE-RULES-NOYAU
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from scripts.ffe_rules_features import (
    Equipe,
    calculer_pct_noyau,
    get_noyau,
    valide_noyau,
)


class TestGetNoyau:
    """Tests pour get_noyau."""

    def test_noyau_vide(self) -> None:
        historique: dict[str, set[int]] = {}
        assert get_noyau("Equipe A", historique) == set()

    def test_noyau_existant(self) -> None:
        historique = {"Equipe A": {1, 2, 3, 4}}
        assert get_noyau("Equipe A", historique) == {1, 2, 3, 4}


class TestCalculerPctNoyau:
    """Tests pour calculer_pct_noyau."""

    def test_pct_noyau_100(self) -> None:
        historique = {"Equipe A": {1, 2, 3, 4}}
        pct = calculer_pct_noyau([1, 2, 3, 4], "Equipe A", historique)
        assert pct == 1.0

    def test_pct_noyau_50(self) -> None:
        historique = {"Equipe A": {1, 2}}
        pct = calculer_pct_noyau([1, 2, 5, 6], "Equipe A", historique)
        assert pct == 0.5

    def test_pct_noyau_0(self) -> None:
        historique = {"Equipe A": {1, 2, 3, 4}}
        pct = calculer_pct_noyau([5, 6, 7, 8], "Equipe A", historique)
        assert pct == 0.0

    def test_pct_noyau_vide(self) -> None:
        historique: dict[str, set[int]] = {}
        pct = calculer_pct_noyau([], "Equipe A", historique)
        assert pct == 0.0


class TestValideNoyau:
    """Tests pour valide_noyau."""

    def test_ronde_1_toujours_valide(self) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="N2", ronde=1)
        historique: dict[str, set[int]] = {}
        regles = {"noyau": 50, "noyau_type": "pourcentage"}
        assert valide_noyau([5, 6, 7, 8], equipe, historique, regles) is True

    def test_noyau_50pct_valide(self) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="N2", ronde=3)
        historique = {"Equipe A": {1, 2, 3, 4}}
        regles = {"noyau": 50, "noyau_type": "pourcentage"}
        assert valide_noyau([1, 2, 3, 4, 5, 6, 7, 8], equipe, historique, regles) is True

    def test_noyau_50pct_invalide(self) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="N2", ronde=3)
        historique = {"Equipe A": {1, 2}}
        regles = {"noyau": 50, "noyau_type": "pourcentage"}
        assert valide_noyau([1, 2, 3, 4, 5, 6, 7, 8], equipe, historique, regles) is False

    def test_noyau_absolu_valide(self) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="Regionale", ronde=3)
        historique = {"Equipe A": {1, 2, 3}}
        regles = {"noyau": 2, "noyau_type": "absolu"}
        assert valide_noyau([1, 2, 3, 4, 5], equipe, historique, regles) is True

    def test_noyau_absolu_invalide(self) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="Regionale", ronde=3)
        historique = {"Equipe A": {1}}
        regles = {"noyau": 2, "noyau_type": "absolu"}
        assert valide_noyau([1, 4, 5, 6, 7], equipe, historique, regles) is False
