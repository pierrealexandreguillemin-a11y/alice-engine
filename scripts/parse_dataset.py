#!/usr/bin/env python3
"""
Parse le dataset FFE HTML pour extraire les donnees d'entrainement ALICE.

Ce module lit les fichiers HTML scraped du site FFE et extrait:
- Compositions: ~750k echiquiers (matchs par equipes)
- Joueurs: ~55k joueurs licencies

Sortie: fichiers Parquet optimises pour entrainement ML.
Conformite ISO/IEC 25010:2023, 25012 (Qualite donnees).

Usage:
    python scripts/parse_dataset.py [--data-dir PATH] [--output-dir PATH] [--verbose]
    python scripts/parse_dataset.py --test  # Test sur un groupe seulement
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from collections.abc import Iterator

# Configuration paths
PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "dataset_alice"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTES ET MAPPINGS
# ==============================================================================

# Mapping titres FIDE (minuscules dans HTML -> standard)
TITRES_FIDE: dict[str, str] = {
    "g": "GM",
    "m": "IM",
    "f": "FM",
    "c": "CM",
    "ff": "WFM",
    "mf": "WIM",
    "gf": "WGM",
    "cf": "WCM",
}

# Mapping ligues regionales (dossier -> code)
LIGUES_REGIONALES: dict[str, str] = {
    "Auvergne-Rhone-Alpes": "ARA",
    "Bourgogne_Franche-Comte": "BFC",
    "Bretagne": "BRE",
    "Corse": "COR",
    "Guyane": "GUY",
    "Reunion": "REU",
    "Ile_de_France": "IDF",
    "Normandie": "NOR",
    "Nouvelle_Aquitaine": "NAQ",
    "Provence_-_Alpes_-_Cote_d'Azur": "PACA",
    "Hauts-De-France": "HDF",
    "Pays_de_la_Loire": "PDL",
    "Occitanie": "OCC",
    "Centre_Val_de_Loire": "CVL",
    "Grand_Est": "GES",
}

# Mapping categories joueurs FFE (format legacy HTML -> ages officiels FFE)
# Source: Reglement FFE 2.4 Categories d'age
# Format donnees: {Cat}{Genre} ex: SenM, PouF
# Genre: M=Masculin, F=Feminin
CATEGORIES_AGE: dict[str, dict[str, Any]] = {
    # U8 - Petits Poussins (moins de 8 ans)
    "PpoM": {"age_max": 7, "genre": "M", "code_ffe": "U8"},
    "PpoF": {"age_max": 7, "genre": "F", "code_ffe": "U8F"},
    # U10 - Poussins (8 ou 9 ans)
    "PouM": {"age_min": 8, "age_max": 9, "genre": "M", "code_ffe": "U10"},
    "PouF": {"age_min": 8, "age_max": 9, "genre": "F", "code_ffe": "U10F"},
    # U12 - Pupilles (10 ou 11 ans)
    "PupM": {"age_min": 10, "age_max": 11, "genre": "M", "code_ffe": "U12"},
    "PupF": {"age_min": 10, "age_max": 11, "genre": "F", "code_ffe": "U12F"},
    # U14 - Benjamins (12 ou 13 ans)
    "BenM": {"age_min": 12, "age_max": 13, "genre": "M", "code_ffe": "U14"},
    "BenF": {"age_min": 12, "age_max": 13, "genre": "F", "code_ffe": "U14F"},
    # U16 - Minimes (14 ou 15 ans)
    "MinM": {"age_min": 14, "age_max": 15, "genre": "M", "code_ffe": "U16"},
    "MinF": {"age_min": 14, "age_max": 15, "genre": "F", "code_ffe": "U16F"},
    # U18 - Cadets (16 ou 17 ans)
    "CadM": {"age_min": 16, "age_max": 17, "genre": "M", "code_ffe": "U18"},
    "CadF": {"age_min": 16, "age_max": 17, "genre": "F", "code_ffe": "U18F"},
    # U20 - Juniors (18 ou 19 ans)
    "JunM": {"age_min": 18, "age_max": 19, "genre": "M", "code_ffe": "U20"},
    "JunF": {"age_min": 18, "age_max": 19, "genre": "F", "code_ffe": "U20F"},
    # Seniors (20 a 49 ans)
    "SenM": {"age_min": 20, "age_max": 49, "genre": "M", "code_ffe": "Sen"},
    "SenF": {"age_min": 20, "age_max": 49, "genre": "F", "code_ffe": "Sen"},
    # S50 - Seniors Plus (50 a 64 ans)
    "SepM": {"age_min": 50, "age_max": 64, "genre": "M", "code_ffe": "S50"},
    "SepF": {"age_min": 50, "age_max": 64, "genre": "F", "code_ffe": "S50"},
    # S65 - Veterans (65 ans et plus)
    "VetM": {"age_min": 65, "genre": "M", "code_ffe": "S65"},
    "VetF": {"age_min": 65, "genre": "F", "code_ffe": "S65"},
}

# Types de competitions
TYPES_COMPETITION: dict[str, str] = {
    "Interclubs": "national",
    "Interclubs_Feminins": "national_feminin",
    "Interclubs_Jeunes": "national_jeunes",
    "Interclubs_Rapide": "national_rapide",
    "Coupe_de_France": "coupe",
    "Coupe_de_la_Parite": "coupe_parite",
    "Coupe_Jean-Claude_Loubatiere": "coupe_jcl",
    "Competitions_Scolaires": "scolaire",
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass
class Joueur:
    """Representation d'un joueur."""

    nom: str
    prenom: str
    nom_complet: str
    elo: int
    titre: str = ""
    titre_fide: str = ""


@dataclass
class Echiquier:
    """Representation d'une partie sur un echiquier."""

    numero: int
    blanc: Joueur
    noir: Joueur
    equipe_blanc: str
    equipe_noir: str
    resultat_blanc: float
    resultat_noir: float
    resultat_text: str
    type_resultat: str
    diff_elo: int


@dataclass
class Match:
    """Representation d'un match entre deux equipes."""

    ronde: int
    equipe_dom: str
    equipe_ext: str
    score_dom: int
    score_ext: int
    date: datetime | None = None
    date_str: str = ""
    heure: str = ""
    jour_semaine: str = ""
    lieu: str = ""
    echiquiers: list[Echiquier] = field(default_factory=list)


@dataclass
class Metadata:
    """Metadonnees extraites du chemin."""

    saison: int
    competition: str
    division: str
    groupe: str
    ligue: str = ""
    ligue_code: str = ""
    niveau: int = 0
    type_competition: str = ""


@dataclass
class JoueurLicencie:
    """Joueur licencie FFE (depuis pages players)."""

    nr_ffe: str  # Licence FFE (ex: K59857)
    id_ffe: int  # ID interne FFE
    nom: str
    prenom: str
    nom_complet: str
    affiliation: str  # A, B ou N (type de licence FFE)
    elo: int
    elo_type: str  # F=Fide, N=National, E=Estime
    elo_rapide: int
    elo_rapide_type: str
    elo_blitz: int
    elo_blitz_type: str
    categorie: str  # SenM, MinM, etc. (format legacy)
    mute: bool  # Mute = transfere d'un autre club cette saison (reglements FFE)
    club: str  # Nom du club (parfois + ville)


# ==============================================================================
# PARSING FUNCTIONS - COMPOSITIONS
# ==============================================================================


def extract_metadata_from_path(groupe_dir: Path, data_root: Path) -> Metadata:
    """Extrait les metadonnees depuis le chemin du dossier.

    Args:
        groupe_dir: Chemin vers le dossier du groupe
        data_root: Racine du dataset (pour calculer chemin relatif)

    Returns:
        Metadata avec saison, competition, division, groupe, etc.
    """
    try:
        rel_path = groupe_dir.relative_to(data_root)
    except ValueError:
        rel_path = groupe_dir

    parts = rel_path.parts

    metadata = Metadata(
        saison=0,
        competition="",
        division="",
        groupe="",
    )

    if len(parts) >= 1:
        try:
            metadata.saison = int(parts[0])
        except ValueError:
            pass

    if len(parts) >= 2:
        comp = parts[1]
        metadata.competition = comp.replace("_", " ")

        # Detecter type de competition
        if comp.startswith("Ligue_"):
            metadata.type_competition = "regional"
            ligue_match = re.search(r"Ligue_(?:de_|des_|du_|d'|de_la_|de_l')?(.+)", comp)
            if ligue_match:
                ligue_name = ligue_match.group(1)
                metadata.ligue = ligue_name.replace("_", " ")
                for key, code in LIGUES_REGIONALES.items():
                    if key in ligue_name or key.lower() in ligue_name.lower():
                        metadata.ligue_code = code
                        break
        else:
            metadata.type_competition = TYPES_COMPETITION.get(comp, "autre")

    if len(parts) >= 3:
        div = parts[2]
        metadata.division = div.replace("_", " ")
        niveau_match = re.search(r"(\d+)", div)
        if niveau_match:
            metadata.niveau = int(niveau_match.group(1))

    if len(parts) >= 4:
        metadata.groupe = parts[3].replace("_", " ")

    return metadata


def parse_player_text(text: str) -> Joueur:
    """Parse une chaine joueur pour extraire nom, titre et elo.

    Args:
        text: Texte du joueur (ex: "g NIKOLOV Momchil  2424")

    Returns:
        Joueur avec titre, nom, prenom, elo
    """
    text = text.strip()

    # Pattern: [titre] NOM Prenom  Elo
    # Titres: g (GM), m (IM), f (FM), ff (WFM), etc.
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


def parse_result(text: str) -> tuple[float, float, str]:
    """Parse un resultat d'echiquier.

    Args:
        text: Resultat (ex: "1 - 0", "X - X", "F", "1 - A")

    Returns:
        Tuple (score_blanc, score_noir, type_resultat)
    """
    text = text.strip().upper()

    # Resultat vide = non joue / en attente
    if not text:
        return (0.0, 0.0, "non_joue")

    # Victoires classiques
    if "1 - 0" in text or "1-0" in text:
        return (1.0, 0.0, "victoire_blanc")
    elif "0 - 1" in text or "0-1" in text:
        return (0.0, 1.0, "victoire_noir")
    elif "X - X" in text or "1/2" in text or "½" in text:
        return (0.5, 0.5, "nulle")

    # Ajournements avec resultat (1-A = victoire blanc apres ajournement)
    if "A" in text and "-" in text:
        parts = text.replace(" ", "").split("-")
        if len(parts) == 2:
            left, right = parts
            if left == "1" and right == "A":
                return (1.0, 0.0, "victoire_blanc_ajournement")
            elif left == "A" and right == "1":
                return (0.0, 1.0, "victoire_noir_ajournement")
            elif left == "2" and right == "A":
                return (2.0, 0.0, "victoire_blanc_ajournement")
            elif left == "A" and right == "2":
                return (0.0, 2.0, "victoire_noir_ajournement")
            elif left == "A" and right == "A":
                return (0.0, 0.0, "ajournement")

    # Ajournement simple
    if text == "A" or text == "AJ." or text == "ADJ":
        return (0.0, 0.0, "ajournement")

    # Forfaits
    if "F" in text:
        # Double forfait (F - F) : les deux en infraction, 0-0
        if text.replace(" ", "") == "F-F":
            return (0.0, 0.0, "double_forfait")
        # Forfait simple
        if text.startswith("F"):
            return (0.0, 1.0, "forfait_blanc")
        else:
            return (1.0, 0.0, "forfait_noir")

    # Format generique X-Y (demi-points, etc.)
    if "-" in text:
        parts = text.replace(" ", "").split("-")
        if len(parts) == 2:
            try:
                score_b = float(parts[0].replace(",", "."))
                score_n = float(parts[1].replace(",", "."))
                if score_b > score_n:
                    return (score_b, score_n, "victoire_blanc")
                elif score_n > score_b:
                    return (score_b, score_n, "victoire_noir")
                else:
                    return (score_b, score_n, "nulle")
            except ValueError:
                pass

    return (0.0, 0.0, "inconnu")


def parse_board_number(text: str) -> tuple[int, str]:
    """Parse numero d'echiquier et couleur.

    Args:
        text: Ex: "1 B" ou "2 N"

    Returns:
        Tuple (numero, couleur) ou (0, "") si non reconnu
    """
    text = text.strip().upper()
    match = re.match(r"(\d+)\s*([BN])", text)
    if match:
        return int(match.group(1)), match.group(2)
    return 0, ""


def parse_calendrier(calendrier_path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    """Parse le fichier calendrier.html pour extraire dates et lieux.

    Args:
        calendrier_path: Chemin vers calendrier.html

    Returns:
        Dict indexe par (ronde, equipe_dom, equipe_ext) avec date, heure, lieu
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

        # Detecter le numero de ronde
        if "RowRonde" in tr_id:
            link = tr.find("a")
            if link:
                ronde_text = link.get_text(strip=True)
                ronde_match = re.search(r"Ronde\s*(\d+)", ronde_text, re.IGNORECASE)
                if ronde_match:
                    current_ronde = int(ronde_match.group(1))

        # Ligne de match avec date et lieu
        elif "RowMatch" in tr_id and current_ronde > 0:
            tds = tr.find_all("td")
            if len(tds) >= 6:
                equipe_blancs = tds[0].get_text(strip=True)
                equipe_noirs = tds[3].get_text(strip=True)
                date_text = tds[4].get_text(strip=True)
                lieu = tds[5].get_text(strip=True)

                # Parser la date
                date_obj = None
                heure = ""
                date_match = re.search(
                    r"(\d{1,2})/(\d{1,2})/(\d{2,4})\s+(\d{1,2}):(\d{2})", date_text
                )
                if date_match:
                    jour = int(date_match.group(1))
                    mois = int(date_match.group(2))
                    annee = int(date_match.group(3))
                    if annee < 100:
                        annee += 2000
                    h = int(date_match.group(4))
                    m = int(date_match.group(5))
                    try:
                        date_obj = datetime(annee, mois, jour, h, m)
                        heure = f"{h:02d}:{m:02d}"
                    except ValueError:
                        pass

                # Jour de la semaine
                jour_semaine = ""
                for jour_nom in [
                    "lundi",
                    "mardi",
                    "mercredi",
                    "jeudi",
                    "vendredi",
                    "samedi",
                    "dimanche",
                ]:
                    if jour_nom in date_text.lower():
                        jour_semaine = jour_nom.capitalize()
                        break

                key = (current_ronde, equipe_blancs, equipe_noirs)
                matchs_info[key] = {
                    "ronde": current_ronde,
                    "date": date_obj,
                    "date_str": date_text,
                    "heure": heure,
                    "jour_semaine": jour_semaine,
                    "lieu": lieu,
                }

    return matchs_info


def parse_ronde(
    html_path: Path, calendrier_info: dict[tuple[int, str, str], dict[str, Any]] | None = None
) -> list[Match]:
    """Parse un fichier ronde HTML.

    Args:
        html_path: Chemin vers ronde_N.html
        calendrier_info: Dict avec infos dates/lieux du calendrier

    Returns:
        Liste de matchs avec echiquiers
    """
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning(f"Erreur lecture {html_path}: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Extraire le numero de ronde du nom de fichier
    ronde_match = re.search(r"ronde_(\d+)", html_path.name)
    ronde_num = int(ronde_match.group(1)) if ronde_match else 0

    matchs: list[Match] = []
    current_match: Match | None = None

    for tr in soup.find_all("tr"):
        tr_id = tr.get("id", "")

        # Nouvelle entete de match
        if "RowEnTeteDetail" in tr_id:
            if current_match:
                matchs.append(current_match)

            tds = tr.find_all("td")
            if len(tds) >= 3:
                equipe_dom = tds[0].get_text(strip=True)
                score_text = tds[1].get_text(strip=True)
                equipe_ext = tds[2].get_text(strip=True)

                # Parser le score
                score_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", score_text)
                if score_match:
                    score_dom = int(score_match.group(1))
                    score_ext = int(score_match.group(2))
                else:
                    score_dom = score_ext = 0

                # Chercher infos calendrier
                cal_info: dict[str, Any] = {}
                if calendrier_info:
                    key = (ronde_num, equipe_dom, equipe_ext)
                    cal_info = calendrier_info.get(key, {})

                current_match = Match(
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

        # Ligne d'echiquier
        elif "RowMatchDetail" in tr_id and current_match:
            tds = tr.find_all("td")
            if len(tds) >= 5:
                board_dom, couleur_dom = parse_board_number(tds[0].get_text())
                joueur_dom = parse_player_text(tds[1].get_text())
                resultat = tds[2].get_text(strip=True)
                joueur_ext = parse_player_text(tds[3].get_text())
                board_ext, couleur_ext = parse_board_number(tds[4].get_text())

                score_blanc, score_noir, type_resultat = parse_result(resultat)

                # Determiner qui a les blancs
                if couleur_dom == "B":
                    blanc = joueur_dom
                    noir = joueur_ext
                    equipe_blanc = current_match.equipe_dom
                    equipe_noir = current_match.equipe_ext
                else:
                    blanc = joueur_ext
                    noir = joueur_dom
                    equipe_blanc = current_match.equipe_ext
                    equipe_noir = current_match.equipe_dom
                    score_blanc, score_noir = score_noir, score_blanc

                echiquier = Echiquier(
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
                current_match.echiquiers.append(echiquier)

    # Ajouter le dernier match
    if current_match:
        matchs.append(current_match)

    return matchs


def parse_groupe(groupe_dir: Path, data_root: Path) -> Iterator[dict[str, Any]]:
    """Parse un dossier groupe complet et genere des lignes pour export.

    Args:
        groupe_dir: Chemin vers le dossier du groupe
        data_root: Racine du dataset

    Yields:
        Dicts representant chaque echiquier avec toutes les infos
    """
    metadata = extract_metadata_from_path(groupe_dir, data_root)

    # Parser le calendrier pour les dates/lieux
    calendrier_path = groupe_dir / "calendrier.html"
    calendrier_info = parse_calendrier(calendrier_path)

    # Parser chaque ronde
    for ronde_file in sorted(groupe_dir.glob("ronde_*.html")):
        matchs = parse_ronde(ronde_file, calendrier_info)

        for match in matchs:
            for ech in match.echiquiers:
                yield {
                    # Metadata
                    "saison": metadata.saison,
                    "competition": metadata.competition,
                    "division": metadata.division,
                    "groupe": metadata.groupe,
                    "ligue": metadata.ligue,
                    "ligue_code": metadata.ligue_code,
                    "niveau": metadata.niveau,
                    "type_competition": metadata.type_competition,
                    # Match
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
                    # Echiquier
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


# ==============================================================================
# PARSING FUNCTIONS - JOUEURS
# ==============================================================================


def parse_elo_value(text: str) -> tuple[int, str]:
    """Parse une valeur Elo avec son type.

    Args:
        text: Ex: "1567 F", "1500 N", "1500 E"

    Returns:
        Tuple (elo, type) ou (0, "") si non reconnu
    """
    text = text.strip()
    # Format: "XXXX T" ou "XXXX\xa0T"
    match = re.match(r"(\d+)\s*([FNE])?", text.replace("\xa0", " "))
    if match:
        elo = int(match.group(1))
        elo_type = match.group(2) or ""
        return (elo, elo_type)
    return (0, "")


def parse_player_page(html_path: Path) -> Iterator[JoueurLicencie]:
    """Parse une page de liste de joueurs licencies.

    Args:
        html_path: Chemin vers page_XXXX.html

    Yields:
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
            # Important pour reglements FFE (limite de mutes par equipe)
            mute_text = tds[8].get_text(strip=True)
            mute = bool(mute_text)  # True si non-vide (contient "M" ou autre marqueur)

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
    """Convertit un JoueurLicencie en dict pour export."""
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
        "categorie": joueur.categorie,  # Format legacy: SenM, PouF, etc.
        "code_ffe": cat_info.get("code_ffe", ""),  # Format officiel: U8, U10, Sen, S50, S65
        "genre": cat_info.get("genre", ""),  # M ou F
        "age_min": cat_info.get("age_min"),  # None pour U8 (pas de min)
        "age_max": cat_info.get("age_max"),  # None pour S65 (pas de max)
        "mute": joueur.mute,  # Transfere d'un autre club (important reglements FFE)
        "club": joueur.club,  # Nom du club (parfois + ville)
    }


# ==============================================================================
# DISCOVERY FUNCTIONS
# ==============================================================================


def find_all_groupes(data_dir: Path) -> list[Path]:
    """Trouve tous les dossiers de groupes (contenant des fichiers ronde_*.html).

    Args:
        data_dir: Racine du dataset

    Returns:
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

    Supporte deux formats:
    - v1: players/page_XXXX.html (flat)
    - v2: players_v2/club_XXX/page_XX.html (hierarchique)

    Args:
        players_dir: Repertoire contenant les pages joueurs

    Returns:
        Liste de chemins vers les fichiers page_*.html
    """
    if not players_dir.exists():
        return []

    # Utilise rglob pour supporter la structure hierarchique v2
    return sorted(players_dir.rglob("page_*.html"))


# ==============================================================================
# MAIN EXPORT FUNCTIONS
# ==============================================================================


def parse_compositions(data_dir: Path, output_path: Path, verbose: bool = False) -> dict[str, int]:
    """Parse toutes les compositions et exporte en Parquet.

    Args:
        data_dir: Racine du dataset
        output_path: Chemin du fichier Parquet de sortie
        verbose: Afficher les details

    Returns:
        Stats de parsing (nb_groupes, nb_matchs, nb_echiquiers, etc.)
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

    # Convertir en table PyArrow
    if all_rows:
        table = pa.Table.from_pylist(all_rows)
        pq.write_table(table, output_path, compression="snappy")
        logger.info(f"Exporte: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        logger.warning("Aucune donnee trouvee!")

    return stats


def parse_joueurs(players_dir: Path, output_path: Path, verbose: bool = False) -> dict[str, int]:
    """Parse toutes les pages joueurs et exporte en Parquet.

    Args:
        players_dir: Repertoire contenant les pages joueurs (v1 ou v2)
        output_path: Chemin du fichier Parquet de sortie
        verbose: Afficher les details

    Returns:
        Stats de parsing (nb_pages, nb_joueurs, etc.)
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("PyArrow requis: pip install pyarrow")
        sys.exit(1)

    pages = find_player_pages(players_dir)
    logger.info(f"Trouve {len(pages)} pages joueurs a parser")

    stats = {
        "nb_pages": 0,
        "nb_joueurs": 0,
        "nb_joueurs_elo_zero": 0,
    }

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

    # Convertir en table PyArrow
    if all_rows:
        table = pa.Table.from_pylist(all_rows)
        pq.write_table(table, output_path, compression="snappy")
        logger.info(f"Exporte: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        logger.warning("Aucune donnee joueur trouvee!")

    return stats


def test_parse_groupe(groupe_path: str, data_dir: Path) -> None:
    """Teste le parsing sur un groupe et affiche les resultats."""
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
    print(f"  Ligue: {metadata.ligue} ({metadata.ligue_code})")
    print(f"  Type: {metadata.type_competition}")

    rows = list(parse_groupe(groupe_dir, data_dir))
    print("\n[STATS]")
    print(f"  Echiquiers: {len(rows)}")

    if rows:
        print("\n[EXEMPLE]")
        row = rows[0]
        print(f"  Ronde {row['ronde']}: {row['equipe_dom']} vs {row['equipe_ext']}")
        print(
            f"  Ech {row['echiquier']}: {row['blanc_nom']} ({row['blanc_elo']}) "
            f"vs {row['noir_nom']} ({row['noir_elo']})"
        )
        print(f"  Resultat: {row['resultat_text']} ({row['type_resultat']})")

    print("\n" + "=" * 80)


# ==============================================================================
# CLI
# ==============================================================================


def main() -> None:
    """Point d'entree principal."""
    parser = argparse.ArgumentParser(
        description="Parse le dataset FFE HTML pour ALICE Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python parse_dataset.py                    # Parse tout vers data/
  python parse_dataset.py --test             # Test sur un groupe
  python parse_dataset.py --verbose          # Afficher details
  python parse_dataset.py --output-dir ./out # Sortie personnalisee
  python parse_dataset.py --joueurs-only --players-dir /path/to/players_v2
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Repertoire du dataset (defaut: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Repertoire de sortie (defaut: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Afficher les details de parsing",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Mode test: parser un seul groupe et afficher les resultats",
    )
    parser.add_argument(
        "--compositions-only",
        action="store_true",
        help="Parser seulement les compositions (pas les joueurs)",
    )
    parser.add_argument(
        "--joueurs-only",
        action="store_true",
        help="Parser seulement les joueurs (pas les compositions)",
    )
    parser.add_argument(
        "--players-dir",
        type=Path,
        default=None,
        help="Repertoire des joueurs (defaut: data-dir/players). Supporte v2: players_v2/club_*/",
    )

    args = parser.parse_args()

    # Verifier le dataset
    if not args.data_dir.exists():
        logger.error(f"Dataset non trouve: {args.data_dir}")
        logger.info("Assurez-vous que le lien symbolique dataset_alice existe")
        sys.exit(1)

    # Mode test
    if args.test:
        test_groupe = args.data_dir / "2025" / "Interclubs" / "Nationale_1" / "Groupe_A"
        if not test_groupe.exists():
            # Chercher un groupe existant
            groupes = find_all_groupes(args.data_dir)
            if groupes:
                test_groupe = groupes[0]
            else:
                logger.error("Aucun groupe trouve pour le test")
                sys.exit(1)

        test_parse_groupe(str(test_groupe), args.data_dir)
        return

    # Creer le dossier de sortie
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parser
    logger.info("=" * 60)
    logger.info("ALICE Engine - Dataset Parser")
    logger.info("=" * 60)

    if not args.joueurs_only:
        logger.info("\n[1/2] Parsing des compositions...")
        comp_output = args.output_dir / "echiquiers.parquet"
        comp_stats = parse_compositions(args.data_dir, comp_output, args.verbose)
        logger.info(f"  Groupes: {comp_stats['nb_groupes']}")
        logger.info(f"  Echiquiers: {comp_stats['nb_echiquiers']}")
        logger.info(
            f"  Forfaits: {comp_stats['nb_echiquiers_forfait']} "
            f"({comp_stats['nb_echiquiers_forfait'] * 100 / max(1, comp_stats['nb_echiquiers']):.1f}%)"
        )
        logger.info(
            f"  Elo=0: {comp_stats['nb_echiquiers_elo_zero']} "
            f"({comp_stats['nb_echiquiers_elo_zero'] * 100 / max(1, comp_stats['nb_echiquiers']):.1f}%)"
        )

    if not args.compositions_only:
        logger.info("\n[2/2] Parsing des joueurs...")
        players_dir = args.players_dir if args.players_dir else args.data_dir / "players"
        joueurs_output = args.output_dir / "joueurs.parquet"
        joueurs_stats = parse_joueurs(players_dir, joueurs_output, args.verbose)
        logger.info(f"  Pages: {joueurs_stats['nb_pages']}")
        logger.info(f"  Joueurs: {joueurs_stats['nb_joueurs']}")
        logger.info(
            f"  Elo=0: {joueurs_stats['nb_joueurs_elo_zero']} "
            f"({joueurs_stats['nb_joueurs_elo_zero'] * 100 / max(1, joueurs_stats['nb_joueurs']):.1f}%)"
        )

    logger.info("\n" + "=" * 60)
    logger.info("Parsing termine!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
