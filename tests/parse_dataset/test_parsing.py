"""Tests Parsing - ISO 29119.

Document ID: ALICE-TEST-PARSE-PARSING
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from scripts.parse_dataset import (
    parse_board_number,
    parse_elo_value,
    parse_player_text,
    parse_result,
)

# ==============================================================================
# TESTS CONSTANTS
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
