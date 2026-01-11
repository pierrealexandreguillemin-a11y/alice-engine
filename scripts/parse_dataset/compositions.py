"""Parsing compositions FFE - ISO 5055.

Ce module contient les fonctions principales de parsing pour les compositions:
- extract_metadata_from_path: Extrait metadata du chemin
- parse_groupe: Parse un groupe complet

Les sous-modules calendrier et ronde contiennent:
- parse_calendrier: Parse calendrier.html
- parse_ronde: Parse ronde_N.html

Conformite ISO/IEC 5055 (<300 lignes, SRP).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.parse_dataset.calendrier import parse_calendrier
from scripts.parse_dataset.constants import LIGUES_REGIONALES, TYPES_COMPETITION
from scripts.parse_dataset.dataclasses import Metadata
from scripts.parse_dataset.ronde import parse_ronde

if TYPE_CHECKING:
    from collections.abc import Iterator

# Re-export for backwards compatibility
__all__ = [
    "extract_metadata_from_path",
    "parse_calendrier",
    "parse_groupe",
    "parse_ronde",
]


def extract_metadata_from_path(groupe_dir: Path, data_root: Path) -> Metadata:
    """Extrait les metadonnees depuis le chemin du dossier.

    Args:
    ----
        groupe_dir: Chemin vers le dossier du groupe
        data_root: Racine du dataset

    Returns:
    -------
        Metadata avec saison, competition, division, groupe, etc.
    """
    try:
        rel_path = groupe_dir.relative_to(data_root)
    except ValueError:
        rel_path = groupe_dir

    parts = rel_path.parts
    metadata = Metadata(saison=0, competition="", division="", groupe="")

    _parse_saison(metadata, parts)
    _parse_competition(metadata, parts)
    _parse_division(metadata, parts)
    _parse_groupe(metadata, parts)

    return metadata


def _parse_saison(metadata: Metadata, parts: tuple[str, ...]) -> None:
    """Parse la saison depuis le chemin."""
    if len(parts) >= 1:
        try:
            metadata.saison = int(parts[0])
        except ValueError:
            pass


def _parse_competition(metadata: Metadata, parts: tuple[str, ...]) -> None:
    """Parse la competition depuis le chemin."""
    if len(parts) < 2:
        return

    comp = parts[1]
    metadata.competition = comp.replace("_", " ")

    if comp.startswith("Ligue_"):
        _parse_ligue_regionale(metadata, comp)
    else:
        metadata.type_competition = TYPES_COMPETITION.get(comp, "autre")


def _parse_ligue_regionale(metadata: Metadata, comp: str) -> None:
    """Parse une ligue regionale."""
    metadata.type_competition = "regional"
    ligue_match = re.search(r"Ligue_(?:de_|des_|du_|d'|de_la_|de_l')?(.+)", comp)
    if not ligue_match:
        return

    ligue_name = ligue_match.group(1)
    metadata.ligue = ligue_name.replace("_", " ")

    for key, code in LIGUES_REGIONALES.items():
        if key in ligue_name or key.lower() in ligue_name.lower():
            metadata.ligue_code = code
            break


def _parse_division(metadata: Metadata, parts: tuple[str, ...]) -> None:
    """Parse la division depuis le chemin."""
    if len(parts) < 3:
        return

    div = parts[2]
    metadata.division = div.replace("_", " ")
    niveau_match = re.search(r"(\d+)", div)
    if niveau_match:
        metadata.niveau = int(niveau_match.group(1))


def _parse_groupe(metadata: Metadata, parts: tuple[str, ...]) -> None:
    """Parse le groupe depuis le chemin."""
    if len(parts) >= 4:
        metadata.groupe = parts[3].replace("_", " ")


def parse_groupe(groupe_dir: Path, data_root: Path) -> Iterator[dict[str, Any]]:
    """Parse un dossier groupe complet.

    Args:
    ----
        groupe_dir: Chemin vers le dossier du groupe
        data_root: Racine du dataset

    Yields:
    ------
        Dicts representant chaque echiquier
    """
    metadata = extract_metadata_from_path(groupe_dir, data_root)
    calendrier_path = groupe_dir / "calendrier.html"
    calendrier_info = parse_calendrier(calendrier_path)

    for ronde_file in sorted(groupe_dir.glob("ronde_*.html")):
        matchs = parse_ronde(ronde_file, calendrier_info)

        for match in matchs:
            for ech in match.echiquiers:
                yield {
                    "saison": metadata.saison,
                    "competition": metadata.competition,
                    "division": metadata.division,
                    "groupe": metadata.groupe,
                    "ligue": metadata.ligue,
                    "ligue_code": metadata.ligue_code,
                    "niveau": metadata.niveau,
                    "type_competition": metadata.type_competition,
                    "ronde": match.ronde,
                    "equipe_dom": match.equipe_dom,
                    "equipe_ext": match.equipe_ext,
                    "score_dom": match.score_dom,
                    "score_ext": match.score_ext,
                    "date": match.date,
                    "date_str": match.date_str,
                    "heure": match.heure,
                    "jour_semaine": match.jour_semaine,
                    "lieu": match.lieu,
                    "echiquier": ech.numero,
                    "blanc_nom": ech.blanc.nom_complet,
                    "blanc_titre": ech.blanc.titre_fide,
                    "blanc_elo": ech.blanc.elo,
                    "blanc_equipe": ech.equipe_blanc,
                    "noir_nom": ech.noir.nom_complet,
                    "noir_titre": ech.noir.titre_fide,
                    "noir_elo": ech.noir.elo,
                    "noir_equipe": ech.equipe_noir,
                    "resultat_blanc": ech.resultat_blanc,
                    "resultat_noir": ech.resultat_noir,
                    "resultat_text": ech.resultat_text,
                    "type_resultat": ech.type_resultat,
                    "diff_elo": ech.diff_elo,
                }
