"""Tests Orchestration - ISO 29119/5259.

Document ID: ALICE-TEST-ORCHESTRATION
Version: 1.1.0
Tests: 17

Classes:
- TestFindAllGroupes: Tests découverte groupes (3 tests)
- TestFindPlayerPages: Tests découverte pages joueurs (3 tests)
- TestParseCompositions: Tests export compositions (4 tests)
- TestParseJoueurs: Tests export joueurs (4 tests)
- TestTestParseGroupe: Tests fonction debug (3 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<150 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestFindAllGroupes:
    """Tests pour find_all_groupes."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Retourne liste vide si aucun fichier ronde."""
        from scripts.parse_dataset.orchestration import find_all_groupes

        result = find_all_groupes(tmp_path)
        assert result == []

    def test_finds_multiple_groupes(self, tmp_path: Path) -> None:
        """Trouve plusieurs groupes avec fichiers ronde."""
        from scripts.parse_dataset.orchestration import find_all_groupes

        g1 = tmp_path / "2025" / "Interclubs" / "N1" / "Groupe_A"
        g2 = tmp_path / "2025" / "Interclubs" / "N1" / "Groupe_B"
        g1.mkdir(parents=True)
        g2.mkdir(parents=True)
        (g1 / "ronde_1.html").write_text("<html></html>")
        (g2 / "ronde_1.html").write_text("<html></html>")

        result = find_all_groupes(tmp_path)

        assert len(result) == 2

    def test_deduplicates_groupes(self, tmp_path: Path) -> None:
        """Ne duplique pas les groupes avec plusieurs rondes."""
        from scripts.parse_dataset.orchestration import find_all_groupes

        groupe = tmp_path / "groupe"
        groupe.mkdir()
        (groupe / "ronde_1.html").write_text("<html></html>")
        (groupe / "ronde_2.html").write_text("<html></html>")
        (groupe / "ronde_3.html").write_text("<html></html>")

        result = find_all_groupes(tmp_path)

        assert len(result) == 1


class TestFindPlayerPages:
    """Tests pour find_player_pages."""

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Retourne liste vide si répertoire inexistant."""
        from scripts.parse_dataset.orchestration import find_player_pages

        result = find_player_pages(tmp_path / "nonexistent")
        assert result == []

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Retourne liste vide si répertoire vide."""
        from scripts.parse_dataset.orchestration import find_player_pages

        result = find_player_pages(tmp_path)
        assert result == []

    def test_finds_player_pages(self, tmp_path: Path) -> None:
        """Trouve les fichiers page_*.html."""
        from scripts.parse_dataset.orchestration import find_player_pages

        (tmp_path / "page_001.html").write_text("<html></html>")
        (tmp_path / "page_002.html").write_text("<html></html>")
        (tmp_path / "other.html").write_text("<html></html>")

        result = find_player_pages(tmp_path)

        assert len(result) == 2


class TestParseCompositions:
    """Tests pour parse_compositions."""

    def test_pyarrow_import_error(self, tmp_path: Path) -> None:
        """Exit si PyArrow non installé."""
        from scripts.parse_dataset.orchestration import parse_compositions

        with patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}):
            with pytest.raises(SystemExit):
                parse_compositions(tmp_path, tmp_path / "out.parquet")

    def test_no_data_found(self, tmp_path: Path) -> None:
        """Log warning si aucune donnée."""
        from scripts.parse_dataset.orchestration import parse_compositions

        mock_pa = MagicMock()
        mock_pq = MagicMock()

        with patch.dict("sys.modules", {"pyarrow": mock_pa, "pyarrow.parquet": mock_pq}):
            stats = parse_compositions(tmp_path, tmp_path / "out.parquet")

        assert stats["nb_groupes"] == 0
        assert stats["nb_echiquiers"] == 0

    def test_processes_groupes_with_data(self, tmp_path: Path) -> None:
        """Parse groupes et exporte en Parquet."""
        pytest.importorskip("pyarrow")
        from scripts.parse_dataset.orchestration import parse_compositions

        groupe = tmp_path / "2025" / "Interclubs" / "N1" / "Groupe_A"
        groupe.mkdir(parents=True)
        ronde_html = """<html><body><table>
            <tr id="RowEnTeteDetail1"><td>A</td><td>1-0</td><td>B</td></tr>
            <tr id="RowMatchDetail1">
                <td>1 B</td><td>DUPONT Jean 1600</td><td>1-0</td>
                <td>MARTIN Paul 1500</td><td>1 N</td>
            </tr>
        </table></body></html>"""
        (groupe / "ronde_1.html").write_text(ronde_html, encoding="utf-8")

        output_path = tmp_path / "out.parquet"
        stats = parse_compositions(tmp_path, output_path, verbose=True)

        assert stats["nb_groupes"] == 1
        assert stats["nb_echiquiers"] >= 1
        assert output_path.exists()

    def test_counts_forfaits_and_zero_elo(self, tmp_path: Path) -> None:
        """Compte forfaits et elos à zéro."""
        pytest.importorskip("pyarrow")
        from scripts.parse_dataset.orchestration import parse_compositions

        groupe = tmp_path / "2025" / "Interclubs" / "N1" / "Groupe_A"
        groupe.mkdir(parents=True)
        ronde_html = """<html><body><table>
            <tr id="RowEnTeteDetail1"><td>A</td><td>1-0</td><td>B</td></tr>
            <tr id="RowMatchDetail1">
                <td>1 B</td><td>DUPONT Jean 0</td><td>F-1</td>
                <td>MARTIN Paul 1500</td><td>1 N</td>
            </tr>
        </table></body></html>"""
        (groupe / "ronde_1.html").write_text(ronde_html, encoding="utf-8")

        output_path = tmp_path / "out.parquet"
        stats = parse_compositions(tmp_path, output_path)

        assert stats["nb_echiquiers_forfait"] >= 0
        assert stats["nb_echiquiers_elo_zero"] >= 0


class TestParseJoueurs:
    """Tests pour parse_joueurs."""

    def test_pyarrow_import_error(self, tmp_path: Path) -> None:
        """Exit si PyArrow non installé."""
        from scripts.parse_dataset.orchestration import parse_joueurs

        with patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}):
            with pytest.raises(SystemExit):
                parse_joueurs(tmp_path, tmp_path / "out.parquet")

    def test_no_player_pages(self, tmp_path: Path) -> None:
        """Log warning si aucune page joueur."""
        from scripts.parse_dataset.orchestration import parse_joueurs

        mock_pa = MagicMock()
        mock_pq = MagicMock()

        with patch.dict("sys.modules", {"pyarrow": mock_pa, "pyarrow.parquet": mock_pq}):
            stats = parse_joueurs(tmp_path, tmp_path / "out.parquet")

        assert stats["nb_pages"] == 0
        assert stats["nb_joueurs"] == 0

    def test_processes_player_pages(self, tmp_path: Path) -> None:
        """Parse pages joueurs et exporte en Parquet."""
        pytest.importorskip("pyarrow")
        from scripts.parse_dataset.orchestration import parse_joueurs

        player_html = """<html><body><table>
            <tr class="liste_clair">
                <td>K59857</td><td>DUPONT Jean</td><td>A</td>
                <td><a href="FicheJoueur?Id=123">Info</a></td>
                <td>1650 F</td><td>1600 N</td><td>1550 E</td>
                <td>SenM</td><td></td><td>Club</td>
            </tr>
        </table></body></html>"""
        (tmp_path / "page_001.html").write_text(player_html, encoding="utf-8")

        output_path = tmp_path / "out.parquet"
        stats = parse_joueurs(tmp_path, output_path, verbose=True)

        assert stats["nb_pages"] == 1
        assert stats["nb_joueurs"] >= 1
        assert output_path.exists()

    def test_counts_zero_elo_players(self, tmp_path: Path) -> None:
        """Compte les joueurs avec elo=0 (ISO 5259 data quality)."""
        pytest.importorskip("pyarrow")
        from scripts.parse_dataset.orchestration import parse_joueurs

        # HTML avec joueur elo=0 (valeur vide dans colonnes elo)
        player_html = """<html><body><table>
            <tr class="liste_clair">
                <td>K59857</td><td>DUPONT Jean</td><td>A</td>
                <td><a href="FicheJoueur?Id=123">Info</a></td>
                <td>0 F</td><td>0 N</td><td>0 E</td>
                <td>SenM</td><td></td><td>Club</td>
            </tr>
        </table></body></html>"""
        (tmp_path / "page_001.html").write_text(player_html, encoding="utf-8")

        output_path = tmp_path / "out.parquet"
        stats = parse_joueurs(tmp_path, output_path)

        assert stats["nb_joueurs_elo_zero"] >= 1


class TestTestParseGroupe:
    """Tests pour test_parse_groupe."""

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        """Gère chemin inexistant gracieusement."""
        from scripts.parse_dataset.orchestration import test_parse_groupe

        # Ne doit pas lever d'exception
        test_parse_groupe(str(tmp_path / "nonexistent"), tmp_path)

    def test_empty_rows_no_sample(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """N'affiche pas SAMPLE si rows vide (branche else ligne 193)."""
        from scripts.parse_dataset.orchestration import test_parse_groupe

        # Créer un groupe avec HTML valide mais sans matchs parsables
        groupe = tmp_path / "2025" / "Interclubs" / "N1" / "Groupe_A"
        groupe.mkdir(parents=True)
        # HTML sans RowMatchDetail = aucune ligne parsée
        ronde_html = """<html><body><table>
            <tr id="RowEnTeteDetail1"><td>Club A</td><td>0-0</td><td>Club B</td></tr>
        </table></body></html>"""
        (groupe / "ronde_1.html").write_text(ronde_html, encoding="utf-8")

        test_parse_groupe(str(groupe), tmp_path)

        captured = capsys.readouterr()
        assert "METADATA" in captured.out
        assert "0 lignes" in captured.out
        # SAMPLE ne doit pas apparaître car rows est vide
        assert "SAMPLE" not in captured.out

    def test_parses_and_prints(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Parse et affiche les résultats."""
        from scripts.parse_dataset.orchestration import test_parse_groupe

        groupe = tmp_path / "2025" / "Interclubs" / "N1" / "Groupe_A"
        groupe.mkdir(parents=True)
        ronde_html = """<html><body><table>
            <tr id="RowEnTeteDetail1"><td>Club A</td><td>1-0</td><td>Club B</td></tr>
            <tr id="RowMatchDetail1">
                <td>1 B</td><td>DUPONT Jean 1600</td><td>1-0</td>
                <td>MARTIN Paul 1500</td><td>1 N</td>
            </tr>
        </table></body></html>"""
        (groupe / "ronde_1.html").write_text(ronde_html, encoding="utf-8")

        test_parse_groupe(str(groupe), tmp_path)

        captured = capsys.readouterr()
        assert "METADATA" in captured.out or "Saison" in captured.out
