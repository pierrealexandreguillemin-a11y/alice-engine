"""Tests Main - ISO 29119.

Document ID: ALICE-TEST-PARSE-MAIN
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

import pytest

# ==============================================================================
# TESTS CONSTANTS
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

    def test_parse_compositions_no_data(self, tmp_path: Path) -> None:
        """Test parse_compositions sans données - données vides retourne stats 0."""
        from scripts.parse_dataset import parse_compositions

        output_path = tmp_path / "output.parquet"

        # Appel réel - la fonction gère les données vides
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
