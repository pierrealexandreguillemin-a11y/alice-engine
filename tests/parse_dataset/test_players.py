"""Tests Players - ISO 29119.

Document ID: ALICE-TEST-PARSE-PLAYERS
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

from scripts.parse_dataset import (
    JoueurLicencie,
    joueur_to_dict,
    parse_player_page,
)

# ==============================================================================
# TESTS CONSTANTS
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
