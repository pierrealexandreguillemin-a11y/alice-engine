"""Tests Detection Type Competition - ISO 25012.

Document ID: ALICE-TEST-FFE-RULES-COMPETITION
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from scripts.ffe_rules_features import (
    TypeCompetition,
    detecter_type_competition,
    get_niveau_equipe,
)


class TestDetecterTypeCompetition:
    """Tests pour detecter_type_competition."""

    def test_national_n1(self) -> None:
        assert detecter_type_competition("Nationale 1") == TypeCompetition.A02

    def test_national_n2(self) -> None:
        assert detecter_type_competition("N2 - Groupe A") == TypeCompetition.A02

    def test_top16(self) -> None:
        assert detecter_type_competition("Top 16 hommes") == TypeCompetition.A02

    def test_feminin(self) -> None:
        assert detecter_type_competition("Top 12 Féminin") == TypeCompetition.F01

    def test_coupe_france(self) -> None:
        assert detecter_type_competition("Coupe de France") == TypeCompetition.C01

    def test_coupe_loubatiere(self) -> None:
        assert detecter_type_competition("Coupe Loubatière") == TypeCompetition.C03

    def test_coupe_parite(self) -> None:
        assert detecter_type_competition("Coupe de la Parité") == TypeCompetition.C04

    def test_jeunes(self) -> None:
        assert detecter_type_competition("Top Jeunes") == TypeCompetition.J02

    def test_scolaire(self) -> None:
        assert detecter_type_competition("Championnat Scolaire") == TypeCompetition.J03

    def test_regionale(self) -> None:
        assert detecter_type_competition("Régionale PACA") == TypeCompetition.REG

    def test_departemental(self) -> None:
        assert detecter_type_competition("Départemental BdR") == TypeCompetition.DEP

    def test_default_inconnu(self) -> None:
        assert detecter_type_competition("Tournoi inconnu") == TypeCompetition.A02


class TestGetNiveauEquipe:
    """Tests pour get_niveau_equipe."""

    def test_top16(self) -> None:
        assert get_niveau_equipe("Top 16 Equipe 1") == 1

    def test_n1(self) -> None:
        assert get_niveau_equipe("Nationale 1 - Groupe A") == 2

    def test_n2(self) -> None:
        assert get_niveau_equipe("N2 Ouest") == 3

    def test_n3(self) -> None:
        assert get_niveau_equipe("N3 Groupe 5") == 4

    def test_n4(self) -> None:
        assert get_niveau_equipe("N4 PACA") == 5

    def test_regionale(self) -> None:
        assert get_niveau_equipe("Regionale 1") == 6

    def test_departemental(self) -> None:
        assert get_niveau_equipe("Departemental BdR") == 9

    def test_inconnu(self) -> None:
        assert get_niveau_equipe("Equipe mystere") == 10
