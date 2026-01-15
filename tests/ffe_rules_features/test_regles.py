"""Tests Regles Competition - ISO 25012.

Document ID: ALICE-TEST-FFE-RULES-REGLES
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from scripts.ffe_rules_features import TypeCompetition, get_regles_competition


class TestGetReglesCompetition:
    """Tests pour get_regles_competition."""

    def test_a02_taille_equipe(self) -> None:
        regles = get_regles_competition(TypeCompetition.A02)
        assert regles["taille_equipe"] == 8

    def test_f01_seuil_brulage(self) -> None:
        regles = get_regles_competition(TypeCompetition.F01)
        assert regles["seuil_brulage"] == 1

    def test_c01_ordre_libre(self) -> None:
        regles = get_regles_competition(TypeCompetition.C01)
        assert regles["ordre_elo_obligatoire"] is False

    def test_c03_elo_max(self) -> None:
        regles = get_regles_competition(TypeCompetition.C03)
        assert regles["elo_max"] == 1800

    def test_c04_elo_total(self) -> None:
        regles = get_regles_competition(TypeCompetition.C04)
        assert regles["elo_total_max"] == 8000

    def test_j02_seuil_brulage(self) -> None:
        regles = get_regles_competition(TypeCompetition.J02)
        assert regles["seuil_brulage"] == 4

    def test_reg_noyau_absolu(self) -> None:
        regles = get_regles_competition(TypeCompetition.REG)
        assert regles["noyau"] == 2
        assert regles["noyau_type"] == "absolu"
