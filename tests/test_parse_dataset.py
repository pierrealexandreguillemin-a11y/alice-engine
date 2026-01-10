"""Module: test_parse_dataset.py - Tests Parsing Dataset.

Tests unitaires complets pour le parsing des fichiers FFE HTML.
Couvre: constants, dataclasses, parsing_utils, compositions, players.

ISO Compliance:
- ISO/IEC 29119 - Software Testing (unit tests, coverage)
- ISO/IEC 5259:2024 - Data Quality for ML (parsing validation)
- ISO/IEC 25012 - Data Quality (exactitude, coherence)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.parse_dataset import (
    CATEGORIES_AGE,
    LIGUES_REGIONALES,
    TITRES_FIDE,
    TYPES_COMPETITION,
    Echiquier,
    Joueur,
    JoueurLicencie,
    Match,
    Metadata,
    _invert_type_resultat,
    extract_metadata_from_path,
    joueur_to_dict,
    parse_board_number,
    parse_calendrier,
    parse_elo_value,
    parse_groupe,
    parse_player_page,
    parse_player_text,
    parse_result,
    parse_ronde,
)

# ==============================================================================
# TESTS CONSTANTS
# ==============================================================================


class TestConstants:
    """Tests pour les constantes du module."""

    def test_titres_fide_lowercase_keys(self) -> None:
        """Test que toutes les clés sont en minuscules."""
        for key in TITRES_FIDE:
            assert key == key.lower()

    def test_titres_fide_uppercase_values(self) -> None:
        """Test que toutes les valeurs sont en majuscules."""
        for value in TITRES_FIDE.values():
            assert value == value.upper()

    def test_titres_fide_standard_titles(self) -> None:
        """Test les titres FIDE standards."""
        assert TITRES_FIDE["g"] == "GM"
        assert TITRES_FIDE["m"] == "IM"
        assert TITRES_FIDE["f"] == "FM"
        assert TITRES_FIDE["c"] == "CM"

    def test_titres_fide_women_titles(self) -> None:
        """Test les titres FIDE féminins."""
        assert TITRES_FIDE["gf"] == "WGM"
        assert TITRES_FIDE["mf"] == "WIM"
        assert TITRES_FIDE["ff"] == "WFM"
        assert TITRES_FIDE["cf"] == "WCM"

    def test_ligues_regionales_all_codes(self) -> None:
        """Test que tous les codes régionaux sont présents."""
        expected_codes = {
            "ARA",
            "BFC",
            "BRE",
            "COR",
            "GUY",
            "REU",
            "IDF",
            "NOR",
            "NAQ",
            "PACA",
            "HDF",
            "PDL",
            "OCC",
            "CVL",
            "GES",
        }
        actual_codes = set(LIGUES_REGIONALES.values())
        assert actual_codes == expected_codes

    def test_ligues_regionales_idf(self) -> None:
        """Test code Île-de-France."""
        assert LIGUES_REGIONALES["Ile_de_France"] == "IDF"

    def test_categories_age_seniors(self) -> None:
        """Test catégorie Seniors."""
        assert "SenM" in CATEGORIES_AGE
        assert CATEGORIES_AGE["SenM"]["age_min"] == 20
        assert CATEGORIES_AGE["SenM"]["age_max"] == 49
        assert CATEGORIES_AGE["SenM"]["genre"] == "M"

    def test_categories_age_all_have_code_ffe(self) -> None:
        """Test que toutes les catégories ont un code FFE."""
        for cat, info in CATEGORIES_AGE.items():
            assert "code_ffe" in info, f"Catégorie {cat} manque code_ffe"
            assert "genre" in info, f"Catégorie {cat} manque genre"

    def test_categories_age_youth_categories(self) -> None:
        """Test les catégories jeunes."""
        # Poussins U10
        assert CATEGORIES_AGE["PouM"]["code_ffe"] == "U10"
        assert CATEGORIES_AGE["PouM"]["age_min"] == 8
        assert CATEGORIES_AGE["PouM"]["age_max"] == 9

        # Minimes U16
        assert CATEGORIES_AGE["MinM"]["code_ffe"] == "U16"
        assert CATEGORIES_AGE["MinM"]["age_min"] == 14
        assert CATEGORIES_AGE["MinM"]["age_max"] == 15

    def test_types_competition_national(self) -> None:
        """Test types de compétition nationaux."""
        assert TYPES_COMPETITION["Interclubs"] == "national"
        assert TYPES_COMPETITION["Interclubs_Feminins"] == "national_feminin"
        assert TYPES_COMPETITION["Coupe_de_France"] == "coupe"


# ==============================================================================
# TESTS DATACLASSES
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


class TestParsePlayerText:
    """Tests pour parse_player_text."""

    def test_parse_with_title_and_elo(self) -> None:
        """Test parsing joueur avec titre et elo."""
        result = parse_player_text("g KASPAROV Garry  2850")
        assert result.titre == "g"
        assert result.titre_fide == "GM"
        assert result.elo == 2850
        assert "KASPAROV" in result.nom_complet

    def test_parse_without_title(self) -> None:
        """Test parsing joueur sans titre."""
        result = parse_player_text("DUPONT Jean  1650")
        assert result.titre == ""
        assert result.titre_fide == ""
        assert result.elo == 1650

    def test_parse_im_title(self) -> None:
        """Test parsing titre IM."""
        result = parse_player_text("m POLGAR Judit  2735")
        assert result.titre == "m"
        assert result.titre_fide == "IM"
        assert result.elo == 2735

    def test_parse_fm_title(self) -> None:
        """Test parsing titre FM."""
        result = parse_player_text("f MARTIN Pierre  2350")
        assert result.titre == "f"
        assert result.titre_fide == "FM"

    def test_parse_women_titles(self) -> None:
        """Test parsing titres féminins."""
        result = parse_player_text("ff KOSINTSEVA Nadezhda  2450")
        assert result.titre == "ff"
        assert result.titre_fide == "WFM"

    def test_parse_no_elo(self) -> None:
        """Test parsing sans elo reconnu."""
        result = parse_player_text("UNKNOWN Player")
        assert result.elo == 0
        assert result.nom_complet == "UNKNOWN Player"

    def test_parse_whitespace_handling(self) -> None:
        """Test gestion des espaces multiples."""
        result = parse_player_text("  DUPONT  Jean   1500  ")
        assert result.elo == 1500
        assert "DUPONT" in result.nom_complet

    def test_parse_3_digit_elo(self) -> None:
        """Test parsing elo à 3 chiffres."""
        result = parse_player_text("DEBUTANT Marc  999")
        assert result.elo == 999


class TestParseResult:
    """Tests pour parse_result."""

    def test_victoire_blanc_standard(self) -> None:
        """Test victoire blancs format standard."""
        score_b, score_n, type_res = parse_result("1 - 0")
        assert score_b == 1.0
        assert score_n == 0.0
        assert type_res == "victoire_blanc"

    def test_victoire_blanc_compact(self) -> None:
        """Test victoire blancs format compact."""
        score_b, score_n, type_res = parse_result("1-0")
        assert score_b == 1.0
        assert score_n == 0.0
        assert type_res == "victoire_blanc"

    def test_victoire_noir_standard(self) -> None:
        """Test victoire noirs format standard."""
        score_b, score_n, type_res = parse_result("0 - 1")
        assert score_b == 0.0
        assert score_n == 1.0
        assert type_res == "victoire_noir"

    def test_nulle_x(self) -> None:
        """Test nulle format X - X."""
        score_b, score_n, type_res = parse_result("X - X")
        assert score_b == 0.5
        assert score_n == 0.5
        assert type_res == "nulle"

    def test_nulle_half(self) -> None:
        """Test nulle format 1/2."""
        score_b, score_n, type_res = parse_result("1/2")
        assert score_b == 0.5
        assert score_n == 0.5
        assert type_res == "nulle"

    def test_forfait_blanc(self) -> None:
        """Test forfait blancs."""
        score_b, score_n, type_res = parse_result("F - 1")
        assert score_b == 0.0
        assert score_n == 1.0
        assert type_res == "forfait_blanc"

    def test_forfait_noir(self) -> None:
        """Test forfait noirs."""
        score_b, score_n, type_res = parse_result("1 - F")
        assert score_b == 1.0
        assert score_n == 0.0
        assert type_res == "forfait_noir"

    def test_double_forfait(self) -> None:
        """Test double forfait."""
        score_b, score_n, type_res = parse_result("F-F")
        assert score_b == 0.0
        assert score_n == 0.0
        assert type_res == "double_forfait"

    def test_ajournement(self) -> None:
        """Test ajournement."""
        score_b, score_n, type_res = parse_result("A")
        assert score_b == 0.0
        assert score_n == 0.0
        assert type_res == "ajournement"

    def test_victoire_blanc_apres_ajournement(self) -> None:
        """Test victoire blanc après ajournement."""
        score_b, score_n, type_res = parse_result("1 - A")
        assert score_b == 1.0
        assert score_n == 0.0
        assert type_res == "victoire_blanc_ajournement"

    def test_empty_result(self) -> None:
        """Test résultat vide."""
        score_b, score_n, type_res = parse_result("")
        assert type_res == "non_joue"

    def test_unknown_result(self) -> None:
        """Test résultat inconnu."""
        score_b, score_n, type_res = parse_result("???")
        assert type_res == "inconnu"


class TestParseBoardNumber:
    """Tests pour parse_board_number."""

    def test_board_blanc(self) -> None:
        """Test échiquier avec blancs."""
        num, couleur = parse_board_number("1 B")
        assert num == 1
        assert couleur == "B"

    def test_board_noir(self) -> None:
        """Test échiquier avec noirs."""
        num, couleur = parse_board_number("2 N")
        assert num == 2
        assert couleur == "N"

    def test_board_no_space(self) -> None:
        """Test échiquier sans espace."""
        num, couleur = parse_board_number("3B")
        assert num == 3
        assert couleur == "B"

    def test_board_lowercase(self) -> None:
        """Test échiquier minuscule."""
        num, couleur = parse_board_number("4 n")
        assert num == 4
        assert couleur == "N"

    def test_board_invalid(self) -> None:
        """Test échiquier invalide."""
        num, couleur = parse_board_number("invalid")
        assert num == 0
        assert couleur == ""

    def test_board_double_digit(self) -> None:
        """Test échiquier à deux chiffres."""
        num, couleur = parse_board_number("12 B")
        assert num == 12
        assert couleur == "B"


class TestParseEloValue:
    """Tests pour parse_elo_value."""

    def test_elo_fide(self) -> None:
        """Test elo FIDE."""
        elo, type_elo = parse_elo_value("1567 F")
        assert elo == 1567
        assert type_elo == "F"

    def test_elo_national(self) -> None:
        """Test elo national."""
        elo, type_elo = parse_elo_value("1500 N")
        assert elo == 1500
        assert type_elo == "N"

    def test_elo_estime(self) -> None:
        """Test elo estimé."""
        elo, type_elo = parse_elo_value("1200 E")
        assert elo == 1200
        assert type_elo == "E"

    def test_elo_without_type(self) -> None:
        """Test elo sans type."""
        elo, type_elo = parse_elo_value("1800")
        assert elo == 1800
        assert type_elo == ""

    def test_elo_with_nbsp(self) -> None:
        """Test elo avec espace insécable."""
        elo, type_elo = parse_elo_value("1600\xa0F")
        assert elo == 1600
        assert type_elo == "F"

    def test_elo_invalid(self) -> None:
        """Test elo invalide."""
        elo, type_elo = parse_elo_value("invalid")
        assert elo == 0
        assert type_elo == ""


# ==============================================================================
# TESTS COMPOSITIONS
# ==============================================================================


class TestExtractMetadataFromPath:
    """Tests pour extract_metadata_from_path."""

    def test_national_competition(self, tmp_path: Path) -> None:
        """Test extraction metadata compétition nationale."""
        data_root = tmp_path / "dataset"
        data_root.mkdir()
        groupe_dir = data_root / "2025" / "Interclubs" / "Nationale_1" / "Groupe_A"
        groupe_dir.mkdir(parents=True)

        meta = extract_metadata_from_path(groupe_dir, data_root)

        assert meta.saison == 2025
        assert "Interclubs" in meta.competition
        assert meta.type_competition == "national"
        assert meta.niveau == 1

    def test_regional_competition(self, tmp_path: Path) -> None:
        """Test extraction metadata compétition régionale."""
        data_root = tmp_path / "dataset"
        data_root.mkdir()
        groupe_dir = data_root / "2025" / "Ligue_de_Ile_de_France" / "R1" / "Groupe_1"
        groupe_dir.mkdir(parents=True)

        meta = extract_metadata_from_path(groupe_dir, data_root)

        assert meta.saison == 2025
        assert meta.type_competition == "regional"

    def test_minimal_path(self, tmp_path: Path) -> None:
        """Test extraction avec chemin minimal."""
        data_root = tmp_path / "dataset"
        data_root.mkdir()
        groupe_dir = data_root / "2025"
        groupe_dir.mkdir(parents=True)

        meta = extract_metadata_from_path(groupe_dir, data_root)

        assert meta.saison == 2025
        assert meta.competition == ""

    def test_invalid_saison(self, tmp_path: Path) -> None:
        """Test extraction avec saison invalide."""
        data_root = tmp_path / "dataset"
        data_root.mkdir()
        groupe_dir = data_root / "invalid" / "Interclubs"
        groupe_dir.mkdir(parents=True)

        meta = extract_metadata_from_path(groupe_dir, data_root)

        assert meta.saison == 0

    def test_coupe_de_france(self, tmp_path: Path) -> None:
        """Test extraction Coupe de France."""
        data_root = tmp_path / "dataset"
        data_root.mkdir()
        groupe_dir = data_root / "2025" / "Coupe_de_France" / "Tour_1" / "Groupe"
        groupe_dir.mkdir(parents=True)

        meta = extract_metadata_from_path(groupe_dir, data_root)

        assert meta.type_competition == "coupe"


class TestParseCalendrier:
    """Tests pour parse_calendrier."""

    def test_calendrier_not_found(self, tmp_path: Path) -> None:
        """Test calendrier inexistant."""
        result = parse_calendrier(tmp_path / "nonexistent.html")
        assert result == {}

    def test_calendrier_basic(self, tmp_path: Path) -> None:
        """Test parsing calendrier basique."""
        calendrier_html = """
        <html>
        <body>
        <table>
            <tr id="RowRonde1"><td><a>Ronde 1</a></td></tr>
            <tr id="RowMatch1">
                <td>Club A</td>
                <td></td>
                <td></td>
                <td>Club B</td>
                <td>Samedi 15/01/2025 14:00</td>
                <td>Salle des fêtes</td>
            </tr>
        </table>
        </body>
        </html>
        """
        calendrier_path = tmp_path / "calendrier.html"
        calendrier_path.write_text(calendrier_html, encoding="utf-8")

        result = parse_calendrier(calendrier_path)

        assert len(result) == 1
        key = (1, "Club A", "Club B")
        assert key in result
        assert result[key]["ronde"] == 1
        assert result[key]["lieu"] == "Salle des fêtes"


class TestParseRonde:
    """Tests pour parse_ronde."""

    def test_parse_ronde_empty_file(self, tmp_path: Path) -> None:
        """Test parsing fichier ronde vide."""
        ronde_file = tmp_path / "ronde_1.html"
        ronde_file.write_text("<html><body></body></html>")

        result = parse_ronde(ronde_file)

        assert result == []

    def test_parse_ronde_basic_match(self, tmp_path: Path) -> None:
        """Test parsing match basique."""
        ronde_html = """
        <html>
        <body>
        <table>
            <tr id="RowEnTeteDetail1">
                <td>Club A</td>
                <td>4 - 2</td>
                <td>Club B</td>
            </tr>
            <tr id="RowMatchDetail1">
                <td>1 B</td>
                <td>DUPONT Jean  1650</td>
                <td>1 - 0</td>
                <td>MARTIN Pierre  1550</td>
                <td>1 N</td>
            </tr>
        </table>
        </body>
        </html>
        """
        ronde_file = tmp_path / "ronde_1.html"
        ronde_file.write_text(ronde_html, encoding="utf-8")

        result = parse_ronde(ronde_file)

        assert len(result) == 1
        assert result[0].equipe_dom == "Club A"
        assert result[0].score_dom == 4
        assert len(result[0].echiquiers) == 1

    def test_parse_ronde_nonexistent_file(self, tmp_path: Path) -> None:
        """Test parsing fichier inexistant."""
        result = parse_ronde(tmp_path / "nonexistent.html")
        assert result == []


class TestParseGroupe:
    """Tests pour parse_groupe."""

    def test_parse_groupe_no_files(self, tmp_path: Path) -> None:
        """Test parsing groupe sans fichiers."""
        groupe_dir = tmp_path / "groupe"
        groupe_dir.mkdir()

        result = list(parse_groupe(groupe_dir, tmp_path))

        assert result == []

    def test_parse_groupe_with_ronde(self, tmp_path: Path) -> None:
        """Test parsing groupe avec une ronde."""
        groupe_dir = tmp_path / "2025" / "Interclubs" / "N1" / "Groupe_A"
        groupe_dir.mkdir(parents=True)

        ronde_html = """
        <html>
        <body>
        <table>
            <tr id="RowEnTeteDetail1">
                <td>Club A</td>
                <td>4 - 2</td>
                <td>Club B</td>
            </tr>
            <tr id="RowMatchDetail1">
                <td>1 B</td>
                <td>DUPONT Jean  1650</td>
                <td>1 - 0</td>
                <td>MARTIN Pierre  1550</td>
                <td>1 N</td>
            </tr>
        </table>
        </body>
        </html>
        """
        ronde_file = groupe_dir / "ronde_1.html"
        ronde_file.write_text(ronde_html, encoding="utf-8")

        result = list(parse_groupe(groupe_dir, tmp_path))

        assert len(result) == 1
        assert result[0]["saison"] == 2025
        assert result[0]["ronde"] == 1
        assert result[0]["blanc_nom"] == "DUPONT Jean"


# ==============================================================================
# TESTS PLAYERS
# ==============================================================================


class TestParsePlayerPage:
    """Tests pour parse_player_page."""

    def test_parse_player_page_not_found(self, tmp_path: Path) -> None:
        """Test parsing page joueur inexistante."""
        result = list(parse_player_page(tmp_path / "nonexistent.html"))
        assert result == []

    def test_parse_player_page_basic(self, tmp_path: Path) -> None:
        """Test parsing page joueur basique."""
        player_html = """
        <html>
        <body>
        <table>
            <tr class="liste_clair">
                <td>K59857</td>
                <td>DUPONT Jean</td>
                <td>A</td>
                <td><a href="FicheJoueur?Id=12345">Info</a></td>
                <td>1650 F</td>
                <td>1600 N</td>
                <td>1550 E</td>
                <td>SenM</td>
                <td></td>
                <td>Club Paris</td>
            </tr>
        </table>
        </body>
        </html>
        """
        page_path = tmp_path / "page_001.html"
        page_path.write_text(player_html, encoding="utf-8")

        result = list(parse_player_page(page_path))

        assert len(result) == 1
        assert result[0].nr_ffe == "K59857"
        assert result[0].elo == 1650
        assert result[0].elo_type == "F"
        assert result[0].categorie == "SenM"
        assert result[0].id_ffe == 12345

    def test_parse_player_page_multiple_players(self, tmp_path: Path) -> None:
        """Test parsing page avec plusieurs joueurs."""
        player_html = """
        <html>
        <body>
        <table>
            <tr class="liste_clair">
                <td>K59857</td>
                <td>DUPONT Jean</td>
                <td>A</td>
                <td><a href="FicheJoueur?Id=12345">Info</a></td>
                <td>1650 F</td>
                <td>1600 N</td>
                <td>1550 E</td>
                <td>SenM</td>
                <td></td>
                <td>Club Paris</td>
            </tr>
            <tr class="liste_fonce">
                <td>K59858</td>
                <td>MARTIN Marie</td>
                <td>B</td>
                <td><a href="FicheJoueur?Id=12346">Info</a></td>
                <td>1450 N</td>
                <td>1400 N</td>
                <td>1350 E</td>
                <td>SenF</td>
                <td>M</td>
                <td>Club Lyon</td>
            </tr>
        </table>
        </body>
        </html>
        """
        page_path = tmp_path / "page_001.html"
        page_path.write_text(player_html, encoding="utf-8")

        result = list(parse_player_page(page_path))

        assert len(result) == 2
        assert result[1].nr_ffe == "K59858"
        assert result[1].mute is True


class TestJoueurToDict:
    """Tests pour joueur_to_dict."""

    def test_joueur_to_dict_basic(self) -> None:
        """Test conversion joueur en dict."""
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
            club="Club Paris",
        )

        result = joueur_to_dict(joueur)

        assert result["nr_ffe"] == "K59857"
        assert result["elo"] == 1800
        assert result["code_ffe"] == "Sen"
        assert result["genre"] == "M"
        assert result["age_min"] == 20
        assert result["age_max"] == 49

    def test_joueur_to_dict_unknown_category(self) -> None:
        """Test conversion avec catégorie inconnue."""
        joueur = JoueurLicencie(
            nr_ffe="X00001",
            id_ffe=1,
            nom="TEST",
            prenom="Test",
            nom_complet="TEST Test",
            affiliation="A",
            elo=1500,
            elo_type="E",
            elo_rapide=1500,
            elo_rapide_type="E",
            elo_blitz=1500,
            elo_blitz_type="E",
            categorie="Unknown",
            mute=False,
            club="Test Club",
        )

        result = joueur_to_dict(joueur)

        assert result["code_ffe"] == ""
        assert result["genre"] == ""
        assert result["age_min"] is None


# ==============================================================================
# TESTS MAIN PARSE_DATASET
# ==============================================================================


class TestParseDatasetMain:
    """Tests pour les fonctions du module principal parse_dataset.py."""

    def test_find_all_groupes_empty(self, tmp_path: Path) -> None:
        """Test find_all_groupes sur répertoire vide."""
        from scripts.parse_dataset import find_all_groupes

        result = find_all_groupes(tmp_path)
        assert result == []

    def test_find_all_groupes_with_rondes(self, tmp_path: Path) -> None:
        """Test find_all_groupes avec fichiers ronde."""
        from scripts.parse_dataset import find_all_groupes

        groupe1 = tmp_path / "2025" / "Interclubs" / "N1" / "Groupe_A"
        groupe1.mkdir(parents=True)
        (groupe1 / "ronde_1.html").write_text("<html></html>")

        groupe2 = tmp_path / "2025" / "Interclubs" / "N1" / "Groupe_B"
        groupe2.mkdir(parents=True)
        (groupe2 / "ronde_1.html").write_text("<html></html>")

        result = find_all_groupes(tmp_path)

        assert len(result) == 2

    def test_find_player_pages_empty(self, tmp_path: Path) -> None:
        """Test find_player_pages sur répertoire vide."""
        from scripts.parse_dataset import find_player_pages

        result = find_player_pages(tmp_path)
        assert result == []

    def test_find_player_pages_nonexistent(self, tmp_path: Path) -> None:
        """Test find_player_pages sur répertoire inexistant."""
        from scripts.parse_dataset import find_player_pages

        result = find_player_pages(tmp_path / "nonexistent")
        assert result == []

    def test_find_player_pages_with_files(self, tmp_path: Path) -> None:
        """Test find_player_pages avec fichiers."""
        from scripts.parse_dataset import find_player_pages

        players_dir = tmp_path / "players"
        players_dir.mkdir()
        (players_dir / "page_001.html").write_text("<html></html>")
        (players_dir / "page_002.html").write_text("<html></html>")

        result = find_player_pages(players_dir)

        assert len(result) == 2


class TestParseCompositions:
    """Tests pour parse_compositions."""

    @pytest.fixture
    def mock_pyarrow(self) -> None:
        """Mock pyarrow pour les tests."""
        pass

    def test_parse_compositions_no_data(self, tmp_path: Path) -> None:
        """Test parse_compositions sans données."""
        from scripts.parse_dataset import parse_compositions

        output_path = tmp_path / "output.parquet"

        with patch("pyarrow.parquet") as mock_pq:
            stats = parse_compositions(tmp_path, output_path)

        assert stats["nb_groupes"] == 0
        assert stats["nb_echiquiers"] == 0


class TestTestParseGroupe:
    """Tests pour test_parse_groupe."""

    def test_test_parse_groupe_nonexistent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Test test_parse_groupe avec chemin inexistant."""
        from scripts.parse_dataset import test_parse_groupe

        test_parse_groupe(str(tmp_path / "nonexistent"), tmp_path)

        # La fonction doit gérer gracieusement l'erreur


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestEdgeCases:
    """Tests edge cases."""

    def test_parse_result_nulle_unicode(self) -> None:
        """Test nulle avec caractère unicode."""
        score_b, score_n, type_res = parse_result("½")
        assert score_b == 0.5
        assert score_n == 0.5
        assert type_res == "nulle"

    def test_parse_player_text_composite_name(self) -> None:
        """Test parsing nom composé."""
        result = parse_player_text("g VAN DER WIEL John  2550")
        assert result.titre_fide == "GM"
        assert result.elo == 2550

    def test_parse_result_case_insensitive(self) -> None:
        """Test résultat insensible à la casse."""
        _, _, type_res1 = parse_result("f - 1")
        _, _, type_res2 = parse_result("F - 1")
        assert type_res1 == type_res2 == "forfait_blanc"

    def test_parse_result_zero_zero_is_non_joue(self) -> None:
        """Test 0-0 est classifié comme non_joue, pas nulle.

        ISO 5259: Cohérence données - 0-0 représente un échiquier vide,
        pas une partie nulle. Bug corrigé 11/01/2026.
        """
        score_b, score_n, type_res = parse_result("0 - 0")
        assert score_b == 0.0
        assert score_n == 0.0
        assert type_res == "non_joue"

    def test_parse_result_zero_zero_compact(self) -> None:
        """Test 0-0 format compact."""
        score_b, score_n, type_res = parse_result("0-0")
        assert type_res == "non_joue"

    def test_categories_paired_genders(self) -> None:
        """Test que chaque catégorie a un équivalent masculin et féminin."""
        male_cats = {k for k in CATEGORIES_AGE if k.endswith("M")}
        female_cats = {k for k in CATEGORIES_AGE if k.endswith("F")}

        for m_cat in male_cats:
            f_cat = m_cat[:-1] + "F"
            assert f_cat in female_cats, f"Catégorie féminine manquante: {f_cat}"


# ==============================================================================
# TESTS _invert_type_resultat (ISO 5259 - Data Quality)
# ==============================================================================


class TestInvertTypeResultat:
    """Tests pour _invert_type_resultat - correction bug type_resultat.

    ISO 5259: Cohérence données - type_resultat doit correspondre aux scores.
    Bug découvert 10/01/2026: type_resultat n'était pas inversé quand DOM joue Noir.
    """

    def test_victoire_blanc_to_noir(self) -> None:
        """Test inversion victoire_blanc -> victoire_noir."""
        assert _invert_type_resultat("victoire_blanc") == "victoire_noir"

    def test_victoire_noir_to_blanc(self) -> None:
        """Test inversion victoire_noir -> victoire_blanc."""
        assert _invert_type_resultat("victoire_noir") == "victoire_blanc"

    def test_forfait_blanc_to_noir(self) -> None:
        """Test inversion forfait_blanc -> forfait_noir."""
        assert _invert_type_resultat("forfait_blanc") == "forfait_noir"

    def test_forfait_noir_to_blanc(self) -> None:
        """Test inversion forfait_noir -> forfait_blanc."""
        assert _invert_type_resultat("forfait_noir") == "forfait_blanc"

    def test_victoire_blanc_ajournement_to_noir(self) -> None:
        """Test inversion victoire_blanc_ajournement -> victoire_noir_ajournement."""
        assert _invert_type_resultat("victoire_blanc_ajournement") == "victoire_noir_ajournement"

    def test_victoire_noir_ajournement_to_blanc(self) -> None:
        """Test inversion victoire_noir_ajournement -> victoire_blanc_ajournement."""
        assert _invert_type_resultat("victoire_noir_ajournement") == "victoire_blanc_ajournement"

    def test_nulle_unchanged(self) -> None:
        """Test que nulle reste nulle (symétrique)."""
        assert _invert_type_resultat("nulle") == "nulle"

    def test_ajournement_unchanged(self) -> None:
        """Test que ajournement reste ajournement (symétrique)."""
        assert _invert_type_resultat("ajournement") == "ajournement"

    def test_double_forfait_unchanged(self) -> None:
        """Test que double_forfait reste double_forfait (symétrique)."""
        assert _invert_type_resultat("double_forfait") == "double_forfait"

    def test_non_joue_unchanged(self) -> None:
        """Test que non_joue reste non_joue (symétrique)."""
        assert _invert_type_resultat("non_joue") == "non_joue"

    def test_inconnu_unchanged(self) -> None:
        """Test que inconnu reste inconnu (symétrique)."""
        assert _invert_type_resultat("inconnu") == "inconnu"

    def test_unknown_type_passthrough(self) -> None:
        """Test que type inconnu est retourné tel quel."""
        assert _invert_type_resultat("type_inconnu_xyz") == "type_inconnu_xyz"


# ==============================================================================
# TESTS PARSING ECHIQUIER AVEC DOM NOIR (ISO 5259)
# ==============================================================================


class TestParseEchiquierDomNoir:
    """Tests parsing échiquier quand domicile joue Noir.

    Ces tests vérifient le fix du bug type_resultat (corrigé 10/01/2026).
    ISO 5259: Validation que type_resultat correspond aux scores après swap.
    """

    def test_dom_noir_victoire_ext(self, tmp_path: Path) -> None:
        """Test DOM joue Noir et EXT gagne (victoire_blanc dans données).

        HTML: DOM=Noir perd 0-1 contre EXT=Blanc
        Attendu: resultat_blanc=1.0, resultat_noir=0.0, type_resultat=victoire_blanc
        """
        ronde_html = """
        <html>
        <body>
        <table>
            <tr id="RowEnTeteDetail1">
                <td>Club DOM</td>
                <td>0 - 1</td>
                <td>Club EXT</td>
            </tr>
            <tr id="RowMatchDetail1">
                <td>2 N</td>
                <td>LEGRAND Pierre  1500</td>
                <td>0 - 1</td>
                <td>DUBOIS Jean  1600</td>
                <td>2 B</td>
            </tr>
        </table>
        </body>
        </html>
        """
        ronde_file = tmp_path / "ronde_1.html"
        ronde_file.write_text(ronde_html, encoding="utf-8")

        result = parse_ronde(ronde_file)

        assert len(result) == 1
        assert len(result[0].echiquiers) == 1
        ech = result[0].echiquiers[0]

        # DOM (LEGRAND) joue Noir, EXT (DUBOIS) joue Blanc
        assert ech.blanc.nom_complet == "DUBOIS Jean"
        assert ech.noir.nom_complet == "LEGRAND Pierre"
        # EXT (Blanc) gagne 1-0
        assert ech.resultat_blanc == 1.0
        assert ech.resultat_noir == 0.0
        # type_resultat doit être victoire_blanc (cohérent avec scores)
        assert ech.type_resultat == "victoire_blanc"

    def test_dom_noir_victoire_dom(self, tmp_path: Path) -> None:
        """Test DOM joue Noir et DOM gagne (victoire_noir dans données).

        HTML: DOM=Noir gagne 1-0 contre EXT=Blanc
        Attendu: resultat_blanc=0.0, resultat_noir=1.0, type_resultat=victoire_noir
        """
        ronde_html = """
        <html>
        <body>
        <table>
            <tr id="RowEnTeteDetail1">
                <td>Club DOM</td>
                <td>1 - 0</td>
                <td>Club EXT</td>
            </tr>
            <tr id="RowMatchDetail1">
                <td>2 N</td>
                <td>LEGRAND Pierre  1600</td>
                <td>1 - 0</td>
                <td>DUBOIS Jean  1500</td>
                <td>2 B</td>
            </tr>
        </table>
        </body>
        </html>
        """
        ronde_file = tmp_path / "ronde_1.html"
        ronde_file.write_text(ronde_html, encoding="utf-8")

        result = parse_ronde(ronde_file)

        ech = result[0].echiquiers[0]

        # DOM (LEGRAND) joue Noir et gagne
        assert ech.blanc.nom_complet == "DUBOIS Jean"
        assert ech.noir.nom_complet == "LEGRAND Pierre"
        assert ech.resultat_blanc == 0.0
        assert ech.resultat_noir == 1.0
        # type_resultat doit être victoire_noir (cohérent avec scores)
        assert ech.type_resultat == "victoire_noir"

    def test_dom_noir_nulle(self, tmp_path: Path) -> None:
        """Test DOM joue Noir et nulle (nulle reste nulle).

        HTML: DOM=Noir fait nulle avec EXT=Blanc
        Attendu: resultat_blanc=0.5, resultat_noir=0.5, type_resultat=nulle
        """
        ronde_html = """
        <html>
        <body>
        <table>
            <tr id="RowEnTeteDetail1">
                <td>Club DOM</td>
                <td>0 - 0</td>
                <td>Club EXT</td>
            </tr>
            <tr id="RowMatchDetail1">
                <td>2 N</td>
                <td>LEGRAND Pierre  1550</td>
                <td>X - X</td>
                <td>DUBOIS Jean  1550</td>
                <td>2 B</td>
            </tr>
        </table>
        </body>
        </html>
        """
        ronde_file = tmp_path / "ronde_1.html"
        ronde_file.write_text(ronde_html, encoding="utf-8")

        result = parse_ronde(ronde_file)

        ech = result[0].echiquiers[0]

        assert ech.resultat_blanc == 0.5
        assert ech.resultat_noir == 0.5
        # Nulle reste nulle (symétrique)
        assert ech.type_resultat == "nulle"

    def test_dom_blanc_victoire_dom(self, tmp_path: Path) -> None:
        """Test DOM joue Blanc et gagne (cas normal sans swap).

        HTML: DOM=Blanc gagne 1-0 contre EXT=Noir
        Attendu: resultat_blanc=1.0, resultat_noir=0.0, type_resultat=victoire_blanc
        """
        ronde_html = """
        <html>
        <body>
        <table>
            <tr id="RowEnTeteDetail1">
                <td>Club DOM</td>
                <td>1 - 0</td>
                <td>Club EXT</td>
            </tr>
            <tr id="RowMatchDetail1">
                <td>1 B</td>
                <td>LEGRAND Pierre  1600</td>
                <td>1 - 0</td>
                <td>DUBOIS Jean  1500</td>
                <td>1 N</td>
            </tr>
        </table>
        </body>
        </html>
        """
        ronde_file = tmp_path / "ronde_1.html"
        ronde_file.write_text(ronde_html, encoding="utf-8")

        result = parse_ronde(ronde_file)

        ech = result[0].echiquiers[0]

        # DOM (LEGRAND) joue Blanc
        assert ech.blanc.nom_complet == "LEGRAND Pierre"
        assert ech.noir.nom_complet == "DUBOIS Jean"
        assert ech.resultat_blanc == 1.0
        assert ech.resultat_noir == 0.0
        assert ech.type_resultat == "victoire_blanc"
