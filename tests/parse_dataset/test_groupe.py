"""Tests Groupe - ISO 29119.

Document ID: ALICE-TEST-PARSE-GROUPE
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

from scripts.parse_dataset import (
    parse_groupe,
)

# ==============================================================================
# TESTS CONSTANTS
# ==============================================================================


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
