"""Urgence mathématique (CE) - ISO 5055/5259.

Ce module calcule l'urgence mathématique pour le Composition Engine.
Détermine si les objectifs sont encore atteignables mathématiquement.

Features (DOCUMENTÉES ISO 5259):
- montee_possible: bool - L'équipe peut-elle encore monter?
  Calcul: points_actuels + (rondes_restantes * 2) >= points_1er
- maintien_assure: bool - L'équipe ne peut plus descendre?
  Calcul: points_actuels > points_releguable + (rondes_restantes * 2)
- urgence_level: Classification de l'urgence
  - 'critique': Dernier match pour sauver objectif
  - 'haute': 2-3 matchs pour atteindre objectif
  - 'normale': Situation contrôlée
  - 'aucune': Objectif assuré ou abandonné

Justification calculs:
- 2 pts max par match (victoire)
- Formule standard de mathématiques sportives

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features depuis données réelles, calculs documentés
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_urgency_features(
    standings: pd.DataFrame,
    ronde_actuelle: int,
    nb_rondes_total: int = 9,
) -> pd.DataFrame:
    """Calcule l'urgence mathématique par équipe.

    Args:
    ----
        standings: DataFrame classement avec colonnes:
            - equipe, position, points_cumules, nb_equipes
        ronde_actuelle: Numéro de la ronde actuelle
        nb_rondes_total: Nombre total de rondes (défaut: 9 pour interclubs)

    Returns:
    -------
        DataFrame avec colonnes:
        - equipe: nom équipe
        - saison: saison concernée
        - montee_possible: bool
        - maintien_assure: bool
        - rondes_restantes: int
        - urgence_level: 'critique', 'haute', 'normale', 'aucune'
        - points_max_possibles: int (points actuels + max restant)

    ISO 5259: Urgence calculée depuis mathématiques du classement.
    """
    logger.info(f"Calcul urgence mathématique (ronde {ronde_actuelle}/{nb_rondes_total})...")

    if standings.empty:
        return pd.DataFrame()

    rondes_restantes = max(0, nb_rondes_total - ronde_actuelle)
    points_max_restants = rondes_restantes * 2  # 2 pts par victoire

    # Calculer points du 1er et du relégable
    points_premier = standings["points_cumules"].max() if not standings.empty else 0

    # Relégable = avant-dernier ou dernier selon nb équipes
    standings_sorted = standings.sort_values("points_cumules", ascending=True)
    if len(standings_sorted) >= 2:
        points_releguable = standings_sorted.iloc[1]["points_cumules"]
    else:
        points_releguable = 0

    urgency_data = []

    for _, row in standings.iterrows():
        equipe = row["equipe"]
        saison = row.get("saison", 2025)
        points = row["points_cumules"]
        position = row["position"]
        nb_equipes = row["nb_equipes"]

        # Points max possibles
        points_max = points + points_max_restants

        # Montée possible?
        # Peut-on atteindre ou dépasser le 1er?
        montee_possible = points_max >= points_premier or position <= 2

        # Maintien assuré?
        # Est-on suffisamment au-dessus du relégable?
        if position <= nb_equipes - 2:
            # Pas en zone rouge
            marge = points - points_releguable
            maintien_assure = marge > points_max_restants
        else:
            # En zone rouge
            maintien_assure = False

        # Niveau d'urgence
        urgence = _classify_urgency(
            rondes_restantes=rondes_restantes,
            montee_possible=montee_possible,
            maintien_assure=maintien_assure,
            position=position,
            nb_equipes=nb_equipes,
            ecart_premier=row.get("ecart_premier", 0),
            ecart_dernier=row.get("ecart_dernier", 0),
        )

        urgency_data.append(
            {
                "equipe": equipe,
                "saison": saison,
                "ronde": ronde_actuelle,
                "position": position,
                "points_cumules": points,
                "montee_possible": montee_possible,
                "maintien_assure": maintien_assure,
                "rondes_restantes": rondes_restantes,
                "points_max_possibles": points_max,
                "urgence_level": urgence,
            }
        )

    result = pd.DataFrame(urgency_data)
    logger.info(f"  {len(result)} équipes avec urgence mathématique")
    return result


def _classify_urgency(
    rondes_restantes: int,
    montee_possible: bool,
    maintien_assure: bool,
    position: int,
    nb_equipes: int,
    ecart_premier: int,
    ecart_dernier: int,
) -> str:
    """Classifie le niveau d'urgence.

    Classification (DOCUMENTÉE ISO 5259):
    - 'critique': Dernière ronde + objectif en jeu (maintien ou montée)
    - 'haute': 2-3 rondes restantes + situation tendue
    - 'normale': Situation contrôlée
    - 'aucune': Objectif assuré ou hors course

    Returns
    -------
        Niveau d'urgence: 'critique', 'haute', 'normale', 'aucune'
    """
    # Aucune urgence si maintien assuré et pas en course
    if maintien_assure and position > 4:
        return "aucune"

    # Critique si dernière chance (1 ronde ou moins)
    if rondes_restantes <= 1:
        en_danger = not maintien_assure and (position > nb_equipes - 2 or ecart_dernier <= 2)
        en_course = (montee_possible and position <= 2) or (ecart_premier <= 2 and position <= 4)
        if en_danger or en_course:
            return "critique"

    # Haute si situation tendue (2-3 rondes)
    if rondes_restantes <= 3:
        danger_proche = ecart_dernier <= 2 and not maintien_assure
        course_proche = ecart_premier <= 2 and position <= 4
        if danger_proche or course_proche:
            return "haute"

    # Normale par défaut
    return "normale"
