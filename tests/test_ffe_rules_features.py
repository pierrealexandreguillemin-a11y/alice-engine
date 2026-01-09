"""Module: test_ffe_rules_features.py - Tests Regles FFE.

Tests unitaires pour le module ffe_rules_features.
66 tests couvrant toutes les regles FFE.

ISO Compliance:
- ISO/IEC 29119 - Software Testing (unit tests, coverage)
- ISO/IEC 25012 - Data Quality (validation regles metier)
- ISO/IEC 5259:2024 - Data Quality for ML (business rules)

Couverture:
- Detection type competition
- Calcul joueur brule
- Verification noyau
- Zones d'enjeu
- Validation composition

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from __future__ import annotations

import pytest

from scripts.ffe_rules_features import (
    Equipe,
    Joueur,
    Sexe,
    TypeCompetition,
    calculer_pct_noyau,
    calculer_zone_enjeu,
    detecter_mouvement_joueur,
    detecter_type_competition,
    est_brule,
    get_niveau_equipe,
    get_noyau,
    get_regles_competition,
    matchs_avant_brulage,
    valide_noyau,
    valider_composition,
)

# ==============================================================================
# TESTS DETECTION TYPE COMPETITION
# ==============================================================================


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


# ==============================================================================
# TESTS JOUEUR BRULE
# ==============================================================================


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
        # Jouer en N3 ne brule pas pour N2
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


# ==============================================================================
# TESTS NOYAU
# ==============================================================================


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
        # 4 sur 8 = 50%
        assert valide_noyau([1, 2, 3, 4, 5, 6, 7, 8], equipe, historique, regles) is True

    def test_noyau_50pct_invalide(self) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="N2", ronde=3)
        historique = {"Equipe A": {1, 2}}
        regles = {"noyau": 50, "noyau_type": "pourcentage"}
        # 2 sur 8 = 25% < 50%
        assert valide_noyau([1, 2, 3, 4, 5, 6, 7, 8], equipe, historique, regles) is False

    def test_noyau_absolu_valide(self) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="Regionale", ronde=3)
        historique = {"Equipe A": {1, 2, 3}}
        regles = {"noyau": 2, "noyau_type": "absolu"}
        # 3 >= 2
        assert valide_noyau([1, 2, 3, 4, 5], equipe, historique, regles) is True

    def test_noyau_absolu_invalide(self) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="Regionale", ronde=3)
        historique = {"Equipe A": {1}}
        regles = {"noyau": 2, "noyau_type": "absolu"}
        # 1 < 2
        assert valide_noyau([1, 4, 5, 6, 7], equipe, historique, regles) is False


# ==============================================================================
# TESTS ZONES D'ENJEU
# ==============================================================================


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


# ==============================================================================
# TESTS MOUVEMENT JOUEUR
# ==============================================================================


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


# ==============================================================================
# TESTS REGLES PAR COMPETITION
# ==============================================================================


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


# ==============================================================================
# TESTS VALIDATION COMPOSITION
# ==============================================================================


class TestValiderComposition:
    """Tests pour valider_composition."""

    @pytest.fixture
    def joueurs_a02(self) -> list[Joueur]:
        """Composition valide pour A02."""
        return [
            Joueur(1, "Joueur 1", 2400, Sexe.MASCULIN, "FRA", False),
            Joueur(2, "Joueur 2", 2300, Sexe.MASCULIN, "FRA", False),
            Joueur(3, "Joueur 3", 2200, Sexe.MASCULIN, "FRA", False),
            Joueur(4, "Joueur 4", 2100, Sexe.FEMININ, "FRA", False),
            Joueur(5, "Joueur 5", 2000, Sexe.MASCULIN, "FRA", False),
            Joueur(6, "Joueur 6", 1900, Sexe.MASCULIN, "FRA", False),
            Joueur(7, "Joueur 7", 1800, Sexe.MASCULIN, "FRA", True),  # mute
            Joueur(8, "Joueur 8", 1700, Sexe.MASCULIN, "FRA", False),
        ]

    def test_composition_vide(self) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="N2", ronde=3)
        regles = get_regles_competition(TypeCompetition.A02)
        erreurs = valider_composition([], equipe, {}, {}, regles)
        assert erreurs == ["Composition vide"]

    def test_composition_valide(self, joueurs_a02: list[Joueur]) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="N2", ronde=1)
        regles = get_regles_competition(TypeCompetition.A02)
        # Ronde 1 = pas de contrainte noyau
        erreurs = valider_composition(joueurs_a02, equipe, {}, {}, regles)
        assert erreurs == []

    def test_ordre_elo_invalide(self) -> None:
        joueurs = [
            Joueur(1, "Joueur 1", 1800, Sexe.MASCULIN, "FRA", False),
            Joueur(2, "Joueur 2", 2200, Sexe.MASCULIN, "FRA", False),  # Plus fort
        ]
        equipe = Equipe(nom="Equipe A", club="Club", division="N2", ronde=1)
        regles = get_regles_competition(TypeCompetition.A02)
        erreurs = valider_composition(joueurs, equipe, {}, {}, regles)
        assert any("Ordre Elo" in e for e in erreurs)

    def test_joueur_brule(self) -> None:
        joueurs = [
            Joueur(1, "Joueur Brule", 2200, Sexe.MASCULIN, "FRA", False),
        ]
        equipe = Equipe(nom="N2 Equipe", club="Club", division="N2", ronde=3)
        # Joueur a joue 3 fois en N1
        historique_brulage = {1: {"N1 Equipe": 3}}
        regles = get_regles_competition(TypeCompetition.A02)
        erreurs = valider_composition(joueurs, equipe, historique_brulage, {}, regles)
        assert any("brule" in e.lower() for e in erreurs)

    def test_trop_de_mutes(self) -> None:
        joueurs = [
            Joueur(1, "Mute 1", 2200, Sexe.MASCULIN, "FRA", True),
            Joueur(2, "Mute 2", 2100, Sexe.MASCULIN, "FRA", True),
            Joueur(3, "Mute 3", 2000, Sexe.MASCULIN, "FRA", True),
            Joueur(4, "Mute 4", 1900, Sexe.MASCULIN, "FRA", True),  # 4 mutes > 3
        ]
        equipe = Equipe(nom="Equipe A", club="Club", division="N2", ronde=1)
        regles = get_regles_competition(TypeCompetition.A02)
        erreurs = valider_composition(joueurs, equipe, {}, {}, regles)
        assert any("mutes" in e.lower() for e in erreurs)

    def test_elo_max_loubatiere(self) -> None:
        joueurs = [
            Joueur(1, "Joueur Fort", 2000, Sexe.MASCULIN, "FRA", False),  # > 1800
        ]
        equipe = Equipe(nom="Loubatiere", club="Club", division="Coupe", ronde=1)
        regles = get_regles_competition(TypeCompetition.C03)
        erreurs = valider_composition(joueurs, equipe, {}, {}, regles)
        assert any("2000 > 1800" in e for e in erreurs)

    def test_elo_total_parite(self) -> None:
        joueurs = [
            Joueur(1, "H1", 2200, Sexe.MASCULIN, "FRA", False),
            Joueur(2, "H2", 2100, Sexe.MASCULIN, "FRA", False),
            Joueur(3, "F1", 2000, Sexe.FEMININ, "FRA", False),
            Joueur(4, "F2", 1900, Sexe.FEMININ, "FRA", False),  # Total = 8200 > 8000
        ]
        equipe = Equipe(nom="Parite", club="Club", division="Coupe", ronde=1)
        regles = get_regles_competition(TypeCompetition.C04)
        erreurs = valider_composition(joueurs, equipe, {}, {}, regles)
        assert any("8200 > 8000" in e for e in erreurs)

    def test_composition_parite_invalide(self) -> None:
        joueurs = [
            Joueur(1, "H1", 1800, Sexe.MASCULIN, "FRA", False),
            Joueur(2, "H2", 1700, Sexe.MASCULIN, "FRA", False),
            Joueur(3, "H3", 1600, Sexe.MASCULIN, "FRA", False),  # 3H + 1F
            Joueur(4, "F1", 1500, Sexe.FEMININ, "FRA", False),
        ]
        equipe = Equipe(nom="Parite", club="Club", division="Coupe", ronde=1)
        regles = get_regles_competition(TypeCompetition.C04)
        erreurs = valider_composition(joueurs, equipe, {}, {}, regles)
        # Devrait signaler le non-respect de 2H + 2F
        assert any("hommes" in e.lower() or "femmes" in e.lower() for e in erreurs)
