"""Parsing joueurs FFE - ISO 5055.

Ce module contient les fonctions de parsing pour les joueurs:
- parse_player_page: Parse une page HTML de joueurs
- joueur_to_dict: Convertit un JoueurLicencie en dict

Conformite ISO/IEC 5055 (<300 lignes, SRP).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup

from scripts.parse_dataset.constants import CATEGORIES_AGE
from scripts.parse_dataset.dataclasses import JoueurLicencie
from scripts.parse_dataset.parsing_utils import parse_elo_value

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


def parse_player_page(html_path: Path) -> Iterator[JoueurLicencie]:
    """Parse une page de liste de joueurs licencies.

    Args:
    ----
        html_path: Chemin vers page_XXXX.html

    Yields:
    ------
        JoueurLicencie pour chaque joueur dans la page
    """
    soup = _read_html_soup(html_path)
    if soup is None:
        return

    for tr in soup.find_all("tr", class_=["liste_clair", "liste_fonce"]):
        joueur = _parse_player_row(tr, html_path)
        if joueur:
            yield joueur


def _read_html_soup(html_path: Path) -> BeautifulSoup | None:
    """Lit et parse le fichier HTML."""
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
        return BeautifulSoup(html, "html.parser")
    except OSError as e:
        logger.warning(f"Erreur lecture {html_path}: {e}")
        return None


def _parse_player_row(tr: Any, html_path: Path) -> JoueurLicencie | None:
    """Parse une ligne de joueur."""
    tds = tr.find_all("td")
    if len(tds) < 10:
        return None

    try:
        nom, prenom, nom_complet = _parse_name(tds[1].get_text(strip=True))

        return JoueurLicencie(
            nr_ffe=tds[0].get_text(strip=True),
            id_ffe=_extract_id_ffe(tds[3]),
            nom=nom,
            prenom=prenom,
            nom_complet=nom_complet,
            affiliation=tds[2].get_text(strip=True),
            elo=parse_elo_value(tds[4].get_text())[0],
            elo_type=parse_elo_value(tds[4].get_text())[1],
            elo_rapide=parse_elo_value(tds[5].get_text())[0],
            elo_rapide_type=parse_elo_value(tds[5].get_text())[1],
            elo_blitz=parse_elo_value(tds[6].get_text())[0],
            elo_blitz_type=parse_elo_value(tds[6].get_text())[1],
            categorie=tds[7].get_text(strip=True),
            mute=bool(tds[8].get_text(strip=True)),
            club=tds[9].get_text(strip=True),
        )
    except (IndexError, ValueError) as e:
        logger.debug(f"Erreur parsing joueur dans {html_path}: {e}")
        return None


def _parse_name(nom_complet: str) -> tuple[str, str, str]:
    """Separe nom et prenom."""
    parts = nom_complet.split()
    nom, prenom = "", ""

    for i, part in enumerate(parts):
        if part.isupper():
            nom += " " + part
        else:
            prenom = " ".join(parts[i:])
            break

    nom = nom.strip()
    if not prenom and len(parts) > 1:
        prenom = parts[-1]
        nom = " ".join(parts[:-1])

    return nom, prenom, nom_complet


def _extract_id_ffe(info_cell: Any) -> int:
    """Extrait l'ID FFE du lien."""
    info_link = info_cell.find("a", href=True)
    if not info_link:
        return 0

    id_match = re.search(r"Id=(\d+)", info_link["href"])
    return int(id_match.group(1)) if id_match else 0


def joueur_to_dict(joueur: JoueurLicencie) -> dict[str, Any]:
    """Convertit un JoueurLicencie en dict pour export.

    Args:
    ----
        joueur: JoueurLicencie a convertir

    Returns:
    -------
        Dict avec toutes les informations joueur + infos categorie
    """
    # Extraire infos categorie depuis mapping FFE
    cat_info = CATEGORIES_AGE.get(joueur.categorie, {})

    return {
        "nr_ffe": joueur.nr_ffe,
        "id_ffe": joueur.id_ffe,
        "nom": joueur.nom,
        "prenom": joueur.prenom,
        "nom_complet": joueur.nom_complet,
        "affiliation": joueur.affiliation,
        "elo": joueur.elo,
        "elo_type": joueur.elo_type,
        "elo_rapide": joueur.elo_rapide,
        "elo_rapide_type": joueur.elo_rapide_type,
        "elo_blitz": joueur.elo_blitz,
        "elo_blitz_type": joueur.elo_blitz_type,
        "categorie": joueur.categorie,
        "code_ffe": cat_info.get("code_ffe", ""),
        "genre": cat_info.get("genre", ""),
        "age_min": cat_info.get("age_min"),
        "age_max": cat_info.get("age_max"),
        "mute": joueur.mute,
        "club": joueur.club,
    }
