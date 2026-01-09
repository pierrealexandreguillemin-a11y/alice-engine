"""Parsing ronde FFE - ISO 5055.

Ce module parse les fichiers ronde_N.html pour extraire:
- Matchs entre equipes
- Echiquiers et resultats
- Joueurs et scores

Conformite ISO/IEC 5055 (<300 lignes, SRP).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from scripts.parse_dataset.dataclasses import Echiquier, Match
from scripts.parse_dataset.parsing_utils import (
    parse_board_number,
    parse_player_text,
    parse_result,
)

logger = logging.getLogger(__name__)


def parse_ronde(
    html_path: Path,
    calendrier_info: dict[tuple[int, str, str], dict[str, Any]] | None = None,
) -> list[Match]:
    """Parse un fichier ronde HTML.

    Args:
    ----
        html_path: Chemin vers ronde_N.html
        calendrier_info: Dict avec infos dates/lieux

    Returns:
    -------
        Liste de matchs avec echiquiers
    """
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning(f"Erreur lecture {html_path}: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    ronde_match = re.search(r"ronde_(\d+)", html_path.name)
    ronde_num = int(ronde_match.group(1)) if ronde_match else 0

    matchs: list[Match] = []
    current_match: Match | None = None

    for tr in soup.find_all("tr"):
        tr_id = tr.get("id", "")

        if "RowEnTeteDetail" in tr_id:
            if current_match:
                matchs.append(current_match)

            current_match = _parse_match_header(tr, ronde_num, calendrier_info)

        elif "RowMatchDetail" in tr_id and current_match:
            echiquier = _parse_echiquier_row(tr, current_match)
            if echiquier:
                current_match.echiquiers.append(echiquier)

    if current_match:
        matchs.append(current_match)

    return matchs


def _parse_match_header(
    tr: Any,
    ronde_num: int,
    calendrier_info: dict[tuple[int, str, str], dict[str, Any]] | None,
) -> Match:
    """Parse l'en-tete d'un match (equipes et score).

    Args:
    ----
        tr: Element BeautifulSoup <tr>
        ronde_num: Numero de la ronde
        calendrier_info: Infos calendrier optionnelles

    Returns:
    -------
        Match initialise
    """
    tds = tr.find_all("td")

    equipe_dom = tds[0].get_text(strip=True) if len(tds) > 0 else ""
    score_text = tds[1].get_text(strip=True) if len(tds) > 1 else ""
    equipe_ext = tds[2].get_text(strip=True) if len(tds) > 2 else ""

    score_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", score_text)
    score_dom = int(score_match.group(1)) if score_match else 0
    score_ext = int(score_match.group(2)) if score_match else 0

    cal_info: dict[str, Any] = {}
    if calendrier_info:
        key = (ronde_num, equipe_dom, equipe_ext)
        cal_info = calendrier_info.get(key, {})

    return Match(
        ronde=ronde_num,
        equipe_dom=equipe_dom,
        equipe_ext=equipe_ext,
        score_dom=score_dom,
        score_ext=score_ext,
        date=cal_info.get("date"),
        date_str=cal_info.get("date_str", ""),
        heure=cal_info.get("heure", ""),
        jour_semaine=cal_info.get("jour_semaine", ""),
        lieu=cal_info.get("lieu", ""),
        echiquiers=[],
    )


def _parse_echiquier_row(tr: Any, current_match: Match) -> Echiquier | None:
    """Parse une ligne d'echiquier.

    Args:
    ----
        tr: Element BeautifulSoup <tr>
        current_match: Match en cours

    Returns:
    -------
        Echiquier ou None si parsing echoue
    """
    tds = tr.find_all("td")
    if len(tds) < 5:
        return None

    board_dom, couleur_dom = parse_board_number(tds[0].get_text())
    joueur_dom = parse_player_text(tds[1].get_text())
    resultat = tds[2].get_text(strip=True)
    joueur_ext = parse_player_text(tds[3].get_text())
    board_ext, _ = parse_board_number(tds[4].get_text())

    score_blanc, score_noir, type_resultat = parse_result(resultat)

    if couleur_dom == "B":
        blanc, noir = joueur_dom, joueur_ext
        equipe_blanc = current_match.equipe_dom
        equipe_noir = current_match.equipe_ext
    else:
        blanc, noir = joueur_ext, joueur_dom
        equipe_blanc = current_match.equipe_ext
        equipe_noir = current_match.equipe_dom
        score_blanc, score_noir = score_noir, score_blanc

    return Echiquier(
        numero=board_dom or board_ext,
        blanc=blanc,
        noir=noir,
        equipe_blanc=equipe_blanc,
        equipe_noir=equipe_noir,
        resultat_blanc=score_blanc,
        resultat_noir=score_noir,
        resultat_text=resultat,
        type_resultat=type_resultat,
        diff_elo=blanc.elo - noir.elo,
    )
