"""Parsing calendrier FFE - ISO 5055.

Ce module parse les fichiers calendrier.html pour extraire:
- Dates des matchs
- Lieux de rencontre
- Heures et jours

Conformite ISO/IEC 5055 (<300 lignes, SRP).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_calendrier(calendrier_path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    """Parse le fichier calendrier.html pour extraire dates et lieux.

    Args:
    ----
        calendrier_path: Chemin vers calendrier.html

    Returns:
    -------
        Dict indexe par (ronde, equipe_dom, equipe_ext)
    """
    if not calendrier_path.exists():
        return {}

    try:
        html = calendrier_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning(f"Erreur lecture calendrier {calendrier_path}: {e}")
        return {}

    soup = BeautifulSoup(html, "html.parser")
    matchs_info: dict[tuple[int, str, str], dict[str, Any]] = {}
    current_ronde = 0

    for tr in soup.find_all("tr"):
        tr_id = tr.get("id", "")

        if "RowRonde" in tr_id:
            link = tr.find("a")
            if link:
                ronde_text = link.get_text(strip=True)
                ronde_match = re.search(r"Ronde\s*(\d+)", ronde_text, re.IGNORECASE)
                if ronde_match:
                    current_ronde = int(ronde_match.group(1))

        elif "RowMatch" in tr_id and current_ronde > 0:
            match_info = _parse_match_row(tr, current_ronde)
            if match_info:
                key, info = match_info
                matchs_info[key] = info

    return matchs_info


def _parse_match_row(
    tr: Any, current_ronde: int
) -> tuple[tuple[int, str, str], dict[str, Any]] | None:
    """Parse une ligne de match du calendrier.

    Args:
    ----
        tr: Element BeautifulSoup <tr>
        current_ronde: Numero de la ronde courante

    Returns:
    -------
        Tuple (key, info) ou None si parsing echoue
    """
    tds = tr.find_all("td")
    if len(tds) < 6:
        return None

    equipe_blancs = tds[0].get_text(strip=True)
    equipe_noirs = tds[3].get_text(strip=True)
    date_text = tds[4].get_text(strip=True)
    lieu = tds[5].get_text(strip=True)

    date_obj, heure = _parse_date_time(date_text)
    jour_semaine = _extract_jour_semaine(date_text)

    key = (current_ronde, equipe_blancs, equipe_noirs)
    info = {
        "ronde": current_ronde,
        "date": date_obj,
        "date_str": date_text,
        "heure": heure,
        "jour_semaine": jour_semaine,
        "lieu": lieu,
    }

    return key, info


def _parse_date_time(date_text: str) -> tuple[datetime | None, str]:
    """Parse date et heure depuis le texte.

    Args:
    ----
        date_text: Texte contenant date et heure

    Returns:
    -------
        Tuple (datetime ou None, heure str)
    """
    date_obj = None
    heure = ""

    date_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})\s+(\d{1,2}):(\d{2})", date_text)
    if date_match:
        jour = int(date_match.group(1))
        mois = int(date_match.group(2))
        annee = int(date_match.group(3))
        if annee < 100:
            annee += 2000
        h, m = int(date_match.group(4)), int(date_match.group(5))
        try:
            date_obj = datetime(annee, mois, jour, h, m)
            heure = f"{h:02d}:{m:02d}"
        except ValueError:
            pass

    return date_obj, heure


def _extract_jour_semaine(date_text: str) -> str:
    """Extrait le jour de la semaine depuis le texte.

    Args:
    ----
        date_text: Texte contenant potentiellement le jour

    Returns:
    -------
        Nom du jour capitalise ou chaine vide
    """
    jours = [
        "lundi",
        "mardi",
        "mercredi",
        "jeudi",
        "vendredi",
        "samedi",
        "dimanche",
    ]
    for jour_nom in jours:
        if jour_nom in date_text.lower():
            return jour_nom.capitalize()
    return ""
