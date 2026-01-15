"""Tests Zones Enjeu et Mouvements - ISO 25012.

Document ID: ALICE-TEST-FFE-RULES-ZONES
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from scripts.ffe_rules_features import calculer_zone_enjeu, detecter_mouvement_joueur


class TestCalculerZoneEnjeu:
    """Tests pour calculer_zone_enjeu."""

    def test_montee(self) -> None:
        assert calculer_zone_enjeu(1, 10, "N2") == "montee"

    def test_course_titre(self) -> None:
        assert calculer_zone_enjeu(2, 10, "N2") == "course_titre"
        assert calculer_zone_enjeu(3, 10, "N2") == "course_titre"

    def test_mi_tableau(self) -> None:
        assert calculer_zone_enjeu(5, 10, "N2") == "mi_tableau"

    def test_danger_n2(self) -> None:
        assert calculer_zone_enjeu(8, 10, "N2") == "danger"

    def test_descente_n2(self) -> None:
        assert calculer_zone_enjeu(9, 10, "N2") == "descente"
        assert calculer_zone_enjeu(10, 10, "N2") == "descente"

    def test_descente_top16(self) -> None:
        assert calculer_zone_enjeu(13, 16, "Top16") == "descente"
        assert calculer_zone_enjeu(16, 16, "Top16") == "descente"

    def test_descente_n3(self) -> None:
        assert calculer_zone_enjeu(8, 10, "N3") == "descente"


class TestDetecterMouvementJoueur:
    """Tests pour detecter_mouvement_joueur."""

    def test_promotion_n2_vers_n1(self) -> None:
        mouvement = detecter_mouvement_joueur(2200, "N2 Equipe", "N1 Equipe")
        assert mouvement["type"] == "promotion"
        assert mouvement["equipe_renforcee"] == "N1 Equipe"
        assert mouvement["equipe_affaiblie"] == "N2 Equipe"
        assert mouvement["impact"] == 2200

    def test_relegation_n1_vers_n2(self) -> None:
        mouvement = detecter_mouvement_joueur(2200, "N1 Equipe", "N2 Equipe")
        assert mouvement["type"] == "relegation"
        assert mouvement["equipe_renforcee"] == "N2 Equipe"
        assert mouvement["equipe_affaiblie"] == "N1 Equipe"

    def test_lateral_n2_vers_n2(self) -> None:
        mouvement = detecter_mouvement_joueur(2200, "N2 Equipe A", "N2 Equipe B")
        assert mouvement["type"] == "lateral"
        assert mouvement["equipe_renforcee"] is None
        assert mouvement["impact"] == 0
