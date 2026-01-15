"""Tests Joueur Brule - ISO 25012.

Document ID: ALICE-TEST-FFE-RULES-BRULAGE
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from scripts.ffe_rules_features import est_brule, matchs_avant_brulage


class TestEstBrule:
    """Tests pour est_brule."""

    def test_joueur_non_brule_sans_historique(self) -> None:
        historique: dict[int, dict[str, int]] = {}
        assert est_brule(12345, "N2 Equipe", historique) is False

    def test_joueur_non_brule_2_matchs(self) -> None:
        historique = {12345: {"N1 Equipe": 2}}
        assert est_brule(12345, "N2 Equipe", historique) is False

    def test_joueur_brule_3_matchs(self) -> None:
        historique = {12345: {"N1 Equipe": 3}}
        assert est_brule(12345, "N2 Equipe", historique) is True

    def test_joueur_brule_plus_de_3_matchs(self) -> None:
        historique = {12345: {"Top 16 Equipe": 5}}
        assert est_brule(12345, "N3 Equipe", historique) is True

    def test_joueur_non_brule_equipe_inferieure(self) -> None:
        historique = {12345: {"N3 Equipe": 10}}
        assert est_brule(12345, "N2 Equipe", historique) is False

    def test_joueur_brule_seuil_1_feminin(self) -> None:
        historique = {12345: {"Top12F": 1}}
        assert est_brule(12345, "N1F Equipe", historique, seuil_brulage=1) is True


class TestMatchsAvantBrulage:
    """Tests pour matchs_avant_brulage."""

    def test_aucun_match_3_restants(self) -> None:
        historique: dict[int, dict[str, int]] = {}
        assert matchs_avant_brulage(12345, "N1 Equipe", historique) == 3

    def test_2_matchs_1_restant(self) -> None:
        historique = {12345: {"N1 Equipe": 2}}
        assert matchs_avant_brulage(12345, "N1 Equipe", historique) == 1

    def test_3_matchs_0_restant(self) -> None:
        historique = {12345: {"N1 Equipe": 3}}
        assert matchs_avant_brulage(12345, "N1 Equipe", historique) == 0

    def test_plus_de_3_matchs(self) -> None:
        historique = {12345: {"N1 Equipe": 5}}
        assert matchs_avant_brulage(12345, "N1 Equipe", historique) == 0
