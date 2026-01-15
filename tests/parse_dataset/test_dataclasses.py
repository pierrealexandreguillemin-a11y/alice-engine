"""Tests Dataclasses - ISO 29119.

Document ID: ALICE-TEST-PARSE-DATACLASSES
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from datetime import datetime

from scripts.parse_dataset import (
    Echiquier,
    Joueur,
    JoueurLicencie,
    Match,
    Metadata,
)

# ==============================================================================
# TESTS CONSTANTS
# ==============================================================================


class TestDataclasses:
    """Tests pour les dataclasses."""

    def test_joueur_creation(self) -> None:
        """Test création Joueur."""
        joueur = Joueur(
            nom="KASPAROV",
            prenom="Garry",
            nom_complet="KASPAROV Garry",
            elo=2850,
            titre="g",
            titre_fide="GM",
        )
        assert joueur.nom == "KASPAROV"
        assert joueur.prenom == "Garry"
        assert joueur.elo == 2850
        assert joueur.titre_fide == "GM"

    def test_joueur_defaults(self) -> None:
        """Test valeurs par défaut Joueur."""
        joueur = Joueur(
            nom="DOE",
            prenom="John",
            nom_complet="DOE John",
            elo=1500,
        )
        assert joueur.titre == ""
        assert joueur.titre_fide == ""

    def test_echiquier_creation(self) -> None:
        """Test création Echiquier."""
        blanc = Joueur("A", "B", "A B", 1600)
        noir = Joueur("C", "D", "C D", 1500)
        echiquier = Echiquier(
            numero=1,
            blanc=blanc,
            noir=noir,
            equipe_blanc="Club A",
            equipe_noir="Club B",
            resultat_blanc=1.0,
            resultat_noir=0.0,
            resultat_text="1 - 0",
            type_resultat="victoire_blanc",
            diff_elo=100,
        )
        assert echiquier.numero == 1
        assert echiquier.blanc.elo == 1600
        assert echiquier.resultat_blanc == 1.0
        assert echiquier.diff_elo == 100

    def test_match_creation(self) -> None:
        """Test création Match."""
        match = Match(
            ronde=1,
            equipe_dom="Club A",
            equipe_ext="Club B",
            score_dom=4,
            score_ext=2,
        )
        assert match.ronde == 1
        assert match.score_dom == 4
        assert match.echiquiers == []

    def test_match_with_date(self) -> None:
        """Test Match avec date."""
        date = datetime(2025, 1, 15, 14, 0)
        match = Match(
            ronde=1,
            equipe_dom="A",
            equipe_ext="B",
            score_dom=3,
            score_ext=3,
            date=date,
            heure="14:00",
            jour_semaine="Samedi",
        )
        assert match.date == date
        assert match.heure == "14:00"

    def test_metadata_creation(self) -> None:
        """Test création Metadata."""
        meta = Metadata(
            saison=2025,
            competition="Interclubs",
            division="Nationale 1",
            groupe="Groupe A",
            ligue="",
            ligue_code="",
            niveau=1,
            type_competition="national",
        )
        assert meta.saison == 2025
        assert meta.niveau == 1

    def test_joueur_licencie_creation(self) -> None:
        """Test création JoueurLicencie."""
        joueur = JoueurLicencie(
            nr_ffe="K59857",
            id_ffe=12345,
            nom="DUPONT",
            prenom="Jean",
            nom_complet="DUPONT Jean",
            affiliation="A",
            elo=1800,
            elo_type="F",
            elo_rapide=1750,
            elo_rapide_type="N",
            elo_blitz=1700,
            elo_blitz_type="E",
            categorie="SenM",
            mute=False,
            club="Échiquier Club Paris",
        )
        assert joueur.nr_ffe == "K59857"
        assert joueur.elo_type == "F"
        assert joueur.mute is False


# ==============================================================================
# TESTS PARSING_UTILS
# ==============================================================================
