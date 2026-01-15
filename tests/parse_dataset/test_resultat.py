"""Tests Resultat Inversion - ISO 29119/5259.

Document ID: ALICE-TEST-PARSE-RESULTAT
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)
- ISO/IEC 5259:2024 - Data Quality for ML

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

from scripts.parse_dataset import _invert_type_resultat, parse_ronde


class TestInvertTypeResultat:
    """Tests pour _invert_type_resultat."""

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
        """Test inversion victoire_blanc_ajournement."""
        assert _invert_type_resultat("victoire_blanc_ajournement") == "victoire_noir_ajournement"

    def test_victoire_noir_ajournement_to_blanc(self) -> None:
        """Test inversion victoire_noir_ajournement."""
        assert _invert_type_resultat("victoire_noir_ajournement") == "victoire_blanc_ajournement"

    def test_nulle_unchanged(self) -> None:
        """Test que nulle reste nulle."""
        assert _invert_type_resultat("nulle") == "nulle"

    def test_ajournement_unchanged(self) -> None:
        """Test que ajournement reste ajournement."""
        assert _invert_type_resultat("ajournement") == "ajournement"

    def test_double_forfait_unchanged(self) -> None:
        """Test que double_forfait reste double_forfait."""
        assert _invert_type_resultat("double_forfait") == "double_forfait"

    def test_non_joue_unchanged(self) -> None:
        """Test que non_joue reste non_joue."""
        assert _invert_type_resultat("non_joue") == "non_joue"

    def test_inconnu_unchanged(self) -> None:
        """Test que inconnu reste inconnu."""
        assert _invert_type_resultat("inconnu") == "inconnu"

    def test_unknown_type_passthrough(self) -> None:
        """Test que type inconnu est retourne tel quel."""
        assert _invert_type_resultat("type_inconnu_xyz") == "type_inconnu_xyz"


class TestParseEchiquierDomNoir:
    """Tests parsing echiquier quand domicile joue Noir."""

    def test_dom_noir_victoire_ext(self, tmp_path: Path) -> None:
        """Test DOM joue Noir et EXT gagne."""
        ronde_html = """<html><body><table>
            <tr id="RowEnTeteDetail1"><td>Club DOM</td><td>0 - 1</td><td>Club EXT</td></tr>
            <tr id="RowMatchDetail1">
                <td>2 N</td><td>LEGRAND Pierre  1500</td><td>0 - 1</td>
                <td>DUBOIS Jean  1600</td><td>2 B</td>
            </tr>
        </table></body></html>"""
        ronde_file = tmp_path / "ronde_1.html"
        ronde_file.write_text(ronde_html, encoding="utf-8")
        result = parse_ronde(ronde_file)
        ech = result[0].echiquiers[0]
        assert ech.blanc.nom_complet == "DUBOIS Jean"
        assert ech.noir.nom_complet == "LEGRAND Pierre"
        assert ech.resultat_blanc == 1.0
        assert ech.type_resultat == "victoire_blanc"

    def test_dom_noir_victoire_dom(self, tmp_path: Path) -> None:
        """Test DOM joue Noir et DOM gagne."""
        ronde_html = """<html><body><table>
            <tr id="RowEnTeteDetail1"><td>Club DOM</td><td>1 - 0</td><td>Club EXT</td></tr>
            <tr id="RowMatchDetail1">
                <td>2 N</td><td>LEGRAND Pierre  1600</td><td>1 - 0</td>
                <td>DUBOIS Jean  1500</td><td>2 B</td>
            </tr>
        </table></body></html>"""
        ronde_file = tmp_path / "ronde_1.html"
        ronde_file.write_text(ronde_html, encoding="utf-8")
        result = parse_ronde(ronde_file)
        ech = result[0].echiquiers[0]
        assert ech.resultat_noir == 1.0
        assert ech.type_resultat == "victoire_noir"

    def test_dom_noir_nulle(self, tmp_path: Path) -> None:
        """Test DOM joue Noir et nulle."""
        ronde_html = """<html><body><table>
            <tr id="RowEnTeteDetail1"><td>Club DOM</td><td>0 - 0</td><td>Club EXT</td></tr>
            <tr id="RowMatchDetail1">
                <td>2 N</td><td>LEGRAND Pierre  1550</td><td>X - X</td>
                <td>DUBOIS Jean  1550</td><td>2 B</td>
            </tr>
        </table></body></html>"""
        ronde_file = tmp_path / "ronde_1.html"
        ronde_file.write_text(ronde_html, encoding="utf-8")
        result = parse_ronde(ronde_file)
        ech = result[0].echiquiers[0]
        assert ech.resultat_blanc == 0.5
        assert ech.type_resultat == "nulle"

    def test_dom_blanc_victoire_dom(self, tmp_path: Path) -> None:
        """Test DOM joue Blanc et gagne (cas normal)."""
        ronde_html = """<html><body><table>
            <tr id="RowEnTeteDetail1"><td>Club DOM</td><td>1 - 0</td><td>Club EXT</td></tr>
            <tr id="RowMatchDetail1">
                <td>1 B</td><td>LEGRAND Pierre  1600</td><td>1 - 0</td>
                <td>DUBOIS Jean  1500</td><td>1 N</td>
            </tr>
        </table></body></html>"""
        ronde_file = tmp_path / "ronde_1.html"
        ronde_file.write_text(ronde_html, encoding="utf-8")
        result = parse_ronde(ronde_file)
        ech = result[0].echiquiers[0]
        assert ech.blanc.nom_complet == "LEGRAND Pierre"
        assert ech.resultat_blanc == 1.0
        assert ech.type_resultat == "victoire_blanc"
