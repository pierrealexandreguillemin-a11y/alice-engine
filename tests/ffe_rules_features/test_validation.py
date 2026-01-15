"""Tests Validation Composition - ISO 25012.

Document ID: ALICE-TEST-FFE-RULES-VALIDATION
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
    Joueur,
    Sexe,
    TypeCompetition,
    get_regles_competition,
    valider_composition,
)


class TestValiderComposition:
    """Tests pour valider_composition."""

    def test_composition_vide(self) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="N2", ronde=3)
        regles = get_regles_competition(TypeCompetition.A02)
        erreurs = valider_composition([], equipe, {}, {}, regles)
        assert erreurs == ["Composition vide"]

    def test_composition_valide(self, joueurs_a02: list[Joueur]) -> None:
        equipe = Equipe(nom="Equipe A", club="Club", division="N2", ronde=1)
        regles = get_regles_competition(TypeCompetition.A02)
        erreurs = valider_composition(joueurs_a02, equipe, {}, {}, regles)
        assert erreurs == []

    def test_ordre_elo_invalide(self) -> None:
        joueurs = [
            Joueur(1, "Joueur 1", 1800, Sexe.MASCULIN, "FRA", False),
            Joueur(2, "Joueur 2", 2200, Sexe.MASCULIN, "FRA", False),
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
        historique_brulage = {1: {"N1 Equipe": 3}}
        regles = get_regles_competition(TypeCompetition.A02)
        erreurs = valider_composition(joueurs, equipe, historique_brulage, {}, regles)
        assert any("brule" in e.lower() for e in erreurs)

    def test_trop_de_mutes(self) -> None:
        joueurs = [
            Joueur(1, "Mute 1", 2200, Sexe.MASCULIN, "FRA", True),
            Joueur(2, "Mute 2", 2100, Sexe.MASCULIN, "FRA", True),
            Joueur(3, "Mute 3", 2000, Sexe.MASCULIN, "FRA", True),
            Joueur(4, "Mute 4", 1900, Sexe.MASCULIN, "FRA", True),
        ]
        equipe = Equipe(nom="Equipe A", club="Club", division="N2", ronde=1)
        regles = get_regles_competition(TypeCompetition.A02)
        erreurs = valider_composition(joueurs, equipe, {}, {}, regles)
        assert any("mutes" in e.lower() for e in erreurs)

    def test_elo_max_loubatiere(self) -> None:
        joueurs = [
            Joueur(1, "Joueur Fort", 2000, Sexe.MASCULIN, "FRA", False),
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
            Joueur(4, "F2", 1900, Sexe.FEMININ, "FRA", False),
        ]
        equipe = Equipe(nom="Parite", club="Club", division="Coupe", ronde=1)
        regles = get_regles_competition(TypeCompetition.C04)
        erreurs = valider_composition(joueurs, equipe, {}, {}, regles)
        assert any("8200 > 8000" in e for e in erreurs)

    def test_composition_parite_invalide(self) -> None:
        joueurs = [
            Joueur(1, "H1", 1800, Sexe.MASCULIN, "FRA", False),
            Joueur(2, "H2", 1700, Sexe.MASCULIN, "FRA", False),
            Joueur(3, "H3", 1600, Sexe.MASCULIN, "FRA", False),
            Joueur(4, "F1", 1500, Sexe.FEMININ, "FRA", False),
        ]
        equipe = Equipe(nom="Parite", club="Club", division="Coupe", ronde=1)
        regles = get_regles_competition(TypeCompetition.C04)
        erreurs = valider_composition(joueurs, equipe, {}, {}, regles)
        assert any("hommes" in e.lower() or "femmes" in e.lower() for e in erreurs)
