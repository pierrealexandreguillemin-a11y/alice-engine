"""Tests Calendrier - ISO 29119.

Document ID: ALICE-TEST-PARSE-CALENDRIER
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

from scripts.parse_dataset import (
    parse_calendrier,
    parse_ronde,
)

# ==============================================================================
# TESTS CONSTANTS
# ==============================================================================


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
