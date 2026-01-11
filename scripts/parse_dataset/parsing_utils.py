"""Utilitaires de parsing - ISO 5055.

Ce module contient les fonctions helper de parsing:
- parse_player_text
- parse_result
- parse_board_number
- parse_elo_value

Conformité ISO/IEC 5055.
"""

from __future__ import annotations

import re

from scripts.parse_dataset.constants import TITRES_FIDE
from scripts.parse_dataset.dataclasses import Joueur


def parse_player_text(text: str) -> Joueur:
    """Parse une chaine joueur pour extraire nom, titre et elo.

    Args:
    ----
        text: Texte du joueur (ex: "g NIKOLOV Momchil  2424")

    Returns:
    -------
        Joueur avec titre, nom, prenom, elo
    """
    text = text.strip()

    # Pattern: [titre] NOM Prenom  Elo
    pattern = r"^([gcfm]{1,2})?\s*(.+?)\s+(\d{3,4})$"
    match = re.match(pattern, text, re.IGNORECASE)

    if match:
        titre_code = (match.group(1) or "").lower()
        nom_complet = match.group(2).strip()
        elo = int(match.group(3))

        # Separer nom et prenom (NOM en majuscules, Prenom en mixed case)
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
        elif not nom:
            nom = parts[0] if parts else ""

        return Joueur(
            nom=nom,
            prenom=prenom,
            nom_complet=nom_complet,
            elo=elo,
            titre=titre_code,
            titre_fide=TITRES_FIDE.get(titre_code, ""),
        )

    # Fallback: pas d'elo detecte
    return Joueur(
        nom=text,
        prenom="",
        nom_complet=text,
        elo=0,
        titre="",
        titre_fide="",
    )


def _parse_standard_result(text: str) -> tuple[float, float, str] | None:
    """Parse resultats standards."""
    if "1 - 0" in text or "1-0" in text:
        return (1.0, 0.0, "victoire_blanc")
    if "0 - 1" in text or "0-1" in text:
        return (0.0, 1.0, "victoire_noir")
    if "X - X" in text or "1/2" in text or "½" in text:
        return (0.5, 0.5, "nulle")
    return None


def _parse_ajournement_result(text: str) -> tuple[float, float, str] | None:
    """Parse ajournements."""
    if text in ("A", "AJ.", "ADJ"):
        return (0.0, 0.0, "ajournement")

    if "A" not in text or "-" not in text:
        return None

    return _parse_ajournement_score(text)


def _parse_ajournement_score(text: str) -> tuple[float, float, str] | None:
    """Parse score d'ajournement."""
    parts = text.replace(" ", "").split("-")
    if len(parts) != 2:
        return None

    left, right = parts
    ajournement_map = {
        ("1", "A"): (1.0, 0.0, "victoire_blanc_ajournement"),
        ("A", "1"): (0.0, 1.0, "victoire_noir_ajournement"),
        ("A", "A"): (0.0, 0.0, "ajournement"),
    }
    return ajournement_map.get((left, right))


def _parse_forfait_result(text: str) -> tuple[float, float, str] | None:
    """Parse forfaits."""
    if "F" not in text:
        return None
    if text.replace(" ", "") == "F-F":
        return (0.0, 0.0, "double_forfait")
    if text.startswith("F"):
        return (0.0, 1.0, "forfait_blanc")
    return (1.0, 0.0, "forfait_noir")


def _parse_generic_score(text: str) -> tuple[float, float, str] | None:
    """Parse format generique X-Y.

    ISO 5259: Coherence donnees - 0-0 est non_joue, pas nulle.
    """
    if "-" not in text:
        return None
    parts = text.replace(" ", "").split("-")
    if len(parts) != 2:
        return None
    try:
        score_b = float(parts[0].replace(",", "."))
        score_n = float(parts[1].replace(",", "."))
        # Determine type based on score comparison
        if score_b > score_n:
            type_res = "victoire_blanc"
        elif score_n > score_b:
            type_res = "victoire_noir"
        elif score_b == 0:  # ISO 5259: 0-0 = non joue (echiquier vide)
            type_res = "non_joue"
        else:
            type_res = "nulle"
        return (score_b, score_n, type_res)
    except ValueError:
        return None


def parse_result(text: str) -> tuple[float, float, str]:
    """Parse un resultat d'echiquier.

    Args:
    ----
        text: Resultat (ex: "1 - 0", "X - X", "F", "1 - A")

    Returns:
    -------
        Tuple (score_blanc, score_noir, type_resultat)
    """
    text = text.strip().upper()

    if not text:
        return (0.0, 0.0, "non_joue")

    # Try each parser in order
    for parser in [
        _parse_standard_result,
        _parse_ajournement_result,
        _parse_forfait_result,
        _parse_generic_score,
    ]:
        result = parser(text)
        if result is not None:
            return result

    return (0.0, 0.0, "inconnu")


def parse_board_number(text: str) -> tuple[int, str]:
    """Parse numero d'echiquier et couleur.

    Args:
    ----
        text: Ex: "1 B" ou "2 N"

    Returns:
    -------
        Tuple (numero, couleur) ou (0, "") si non reconnu
    """
    text = text.strip().upper()
    match = re.match(r"(\d+)\s*([BN])", text)
    if match:
        return int(match.group(1)), match.group(2)
    return 0, ""


def parse_elo_value(text: str) -> tuple[int, str]:
    """Parse une valeur Elo avec son type.

    Args:
    ----
        text: Ex: "1567 F", "1500 N", "1500 E"

    Returns:
    -------
        Tuple (elo, type) ou (0, "") si non reconnu
    """
    text = text.strip()
    match = re.match(r"(\d+)\s*([FNE])?", text.replace("\xa0", " "))
    if match:
        elo = int(match.group(1))
        elo_type = match.group(2) or ""
        return (elo, elo_type)
    return (0, "")
