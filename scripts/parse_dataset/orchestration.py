"""Orchestration du parsing FFE - ISO 5055.

Ce module contient les fonctions de découverte et d'export.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from scripts.parse_dataset.compositions import extract_metadata_from_path, parse_groupe
from scripts.parse_dataset.players import joueur_to_dict, parse_player_page

logger = logging.getLogger(__name__)


def find_all_groupes(data_dir: Path) -> list[Path]:
    """Trouve tous les dossiers de groupes.

    Args:
    ----
        data_dir: Racine du dataset

    Returns:
    -------
        Liste de chemins vers les dossiers de groupes
    """
    groupes = []
    for ronde_file in data_dir.rglob("ronde_*.html"):
        groupe_dir = ronde_file.parent
        if groupe_dir not in groupes:
            groupes.append(groupe_dir)
    return sorted(groupes)


def find_player_pages(players_dir: Path) -> list[Path]:
    """Trouve toutes les pages de joueurs.

    Args:
    ----
        players_dir: Repertoire contenant les pages joueurs

    Returns:
    -------
        Liste de chemins vers les fichiers page_*.html
    """
    if not players_dir.exists():
        return []
    return sorted(players_dir.rglob("page_*.html"))


def parse_compositions(
    data_dir: Path,
    output_path: Path,
    verbose: bool = False,
) -> dict[str, int]:
    """Parse toutes les compositions et exporte en Parquet.

    Args:
    ----
        data_dir: Racine du dataset
        output_path: Chemin du fichier Parquet de sortie
        verbose: Afficher les details

    Returns:
    -------
        Stats de parsing
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("PyArrow requis: pip install pyarrow")
        sys.exit(1)

    groupes = find_all_groupes(data_dir)
    logger.info(f"Trouve {len(groupes)} groupes a parser")

    stats = {
        "nb_groupes": 0,
        "nb_matchs": 0,
        "nb_echiquiers": 0,
        "nb_echiquiers_forfait": 0,
        "nb_echiquiers_elo_zero": 0,
    }

    all_rows: list[dict[str, Any]] = []

    for i, groupe_dir in enumerate(groupes, 1):
        if verbose or i % 100 == 0:
            logger.info(f"Parsing groupe {i}/{len(groupes)}: {groupe_dir.name}")

        for row in parse_groupe(groupe_dir, data_dir):
            all_rows.append(row)
            stats["nb_echiquiers"] += 1
            if "forfait" in row["type_resultat"]:
                stats["nb_echiquiers_forfait"] += 1
            if row["blanc_elo"] == 0 or row["noir_elo"] == 0:
                stats["nb_echiquiers_elo_zero"] += 1

        stats["nb_groupes"] += 1

    logger.info(f"Conversion en Parquet ({len(all_rows)} lignes)...")

    if all_rows:
        table = pa.Table.from_pylist(all_rows)
        pq.write_table(table, output_path, compression="snappy")
        logger.info(f"Exporte: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        logger.warning("Aucune donnee trouvee!")

    return stats


def parse_joueurs(
    players_dir: Path,
    output_path: Path,
    verbose: bool = False,
) -> dict[str, int]:
    """Parse toutes les pages joueurs et exporte en Parquet.

    Args:
    ----
        players_dir: Repertoire contenant les pages joueurs
        output_path: Chemin du fichier Parquet de sortie
        verbose: Afficher les details

    Returns:
    -------
        Stats de parsing
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("PyArrow requis: pip install pyarrow")
        sys.exit(1)

    pages = find_player_pages(players_dir)
    logger.info(f"Trouve {len(pages)} pages joueurs a parser")

    stats = {"nb_pages": 0, "nb_joueurs": 0, "nb_joueurs_elo_zero": 0}
    all_rows: list[dict[str, Any]] = []

    for i, page_path in enumerate(pages, 1):
        if verbose or i % 100 == 0:
            logger.info(f"Parsing page {i}/{len(pages)}: {page_path.name}")

        for joueur in parse_player_page(page_path):
            row = joueur_to_dict(joueur)
            all_rows.append(row)
            stats["nb_joueurs"] += 1
            if joueur.elo == 0:
                stats["nb_joueurs_elo_zero"] += 1

        stats["nb_pages"] += 1

    logger.info(f"Conversion en Parquet ({len(all_rows)} lignes)...")

    if all_rows:
        table = pa.Table.from_pylist(all_rows)
        pq.write_table(table, output_path, compression="snappy")
        logger.info(f"Exporte: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        logger.warning("Aucune donnee joueur trouvee!")

    return stats


def test_parse_groupe(groupe_path: str, data_dir: Path) -> None:
    """Teste le parsing sur un groupe."""
    groupe_dir = Path(groupe_path)
    if not groupe_dir.exists():
        logger.error(f"Erreur: {groupe_dir} n'existe pas")
        return

    logger.info(f"Test parsing: {groupe_dir}")
    print("=" * 80)

    metadata = extract_metadata_from_path(groupe_dir, data_dir)
    print("\n[METADATA]")
    print(f"  Saison: {metadata.saison}")
    print(f"  Competition: {metadata.competition}")
    print(f"  Division: {metadata.division}")
    print(f"  Groupe: {metadata.groupe}")
    print(f"  Type: {metadata.type_competition}")

    rows = list(parse_groupe(groupe_dir, data_dir))
    print(f"\n[ECHIQUIERS] {len(rows)} lignes parsees")

    if rows:
        print("\n[SAMPLE]")
        for row in rows[:3]:
            print(
                f"  R{row['ronde']}: {row['blanc_nom']} ({row['blanc_elo']}) "
                f"vs {row['noir_nom']} ({row['noir_elo']}) = {row['resultat_text']}"
            )
