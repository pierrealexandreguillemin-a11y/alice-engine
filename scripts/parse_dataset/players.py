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
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning(f"Erreur lecture {html_path}: {e}")
        return

    soup = BeautifulSoup(html, "html.parser")

    # Trouver toutes les lignes de joueurs (classe liste_clair ou liste_fonce)
    for tr in soup.find_all("tr", class_=["liste_clair", "liste_fonce"]):
        tds = tr.find_all("td")

        if len(tds) < 10:
            continue

        try:
            # NrFFE (licence FFE)
            nr_ffe = tds[0].get_text(strip=True)

            # Nom et Prenom
            nom_cell = tds[1]
            nom_complet = nom_cell.get_text(strip=True)

            # Separer nom et prenom
            parts = nom_complet.split()
            nom = ""
            prenom = ""
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

            # Affiliation
            affiliation = tds[2].get_text(strip=True)

            # ID FFE (lien vers FicheJoueur)
            id_ffe = 0
            info_cell = tds[3]
            info_link = info_cell.find("a", href=True)
            if info_link:
                href = info_link["href"]
                id_match = re.search(r"Id=(\d+)", href)
                if id_match:
                    id_ffe = int(id_match.group(1))

            # Elos
            elo, elo_type = parse_elo_value(tds[4].get_text())
            elo_rapide, elo_rapide_type = parse_elo_value(tds[5].get_text())
            elo_blitz, elo_blitz_type = parse_elo_value(tds[6].get_text())

            # Categorie
            categorie = tds[7].get_text(strip=True)

            # M. = Mute (transfere d'un autre club cette saison)
            mute_text = tds[8].get_text(strip=True)
            mute = bool(mute_text)

            # Club (parfois nom + ville)
            club = tds[9].get_text(strip=True)

            yield JoueurLicencie(
                nr_ffe=nr_ffe,
                id_ffe=id_ffe,
                nom=nom,
                prenom=prenom,
                nom_complet=nom_complet,
                affiliation=affiliation,
                elo=elo,
                elo_type=elo_type,
                elo_rapide=elo_rapide,
                elo_rapide_type=elo_rapide_type,
                elo_blitz=elo_blitz,
                elo_blitz_type=elo_blitz_type,
                categorie=categorie,
                mute=mute,
                club=club,
            )

        except (IndexError, ValueError) as e:
            logger.debug(f"Erreur parsing joueur dans {html_path}: {e}")
            continue


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
