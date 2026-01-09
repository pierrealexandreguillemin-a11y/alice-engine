"""Scénarios classement (CE) - ISO 5055/5259.

Ce module calcule les scénarios de classement pour le Composition Engine.
Détermine la stratégie de composition selon la position au classement.

Scénarios (DOCUMENTÉS ISO 5259):
- "course_titre": Position 1-2 ou écart <= 2 pts du 1er
  -> Composition maximale, titulaires systématiques
- "course_montee": Position 3-4 avec barrage possible
  -> Renforcement progressif, préparation playoffs
- "danger": Position relégable ou écart <= 2 pts du relégable
  -> Mobilisation totale, éviter descente
- "condamne": Dernier avec écart >= 6 pts (mathématiquement descendu)
  -> Démobilisation possible, faire tourner
- "mi_tableau": Ni course ni danger
  -> Rotation, repos titulaires, préparation prochaine saison

Justification seuils:
- 2 pts écart: 1 victoire peut changer la situation
- Position 1-2: Zone montée directe en N1/N2
- Position 3-4: Zone barrage possible
- 6 pts écart: 2 victoires + 1 nul = irrattrapage à 3 rondes

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Features depuis données réelles, seuils documentés
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_scenario_features(standings: pd.DataFrame) -> pd.DataFrame:
    """Calcule le scénario d'équipe depuis le classement.

    Args:
    ----
        standings: DataFrame classement avec colonnes:
            - equipe, saison, position, points_cumules
            - nb_equipes, ecart_premier, ecart_dernier

    Returns:
    -------
        DataFrame avec colonnes:
        - equipe: nom équipe
        - saison: saison concernée
        - ronde: ronde du classement
        - scenario: 'course_titre', 'course_montee', 'danger', 'condamne', 'mi_tableau'
        - urgence_score: float [0, 1] - niveau d'urgence de la situation

    ISO 5259: Scénario calculé depuis classement réel.
    """
    logger.info("Calcul scénarios classement...")

    if standings.empty:
        return pd.DataFrame()

    required_cols = ["equipe", "position", "points_cumules", "nb_equipes"]
    if not all(col in standings.columns for col in required_cols):
        logger.warning("Colonnes requises manquantes pour scénarios")
        return pd.DataFrame()

    scenario_data = []

    for _, row in standings.iterrows():
        equipe = row["equipe"]
        saison = row.get("saison", 2025)
        ronde = row.get("ronde", 1)
        position = row["position"]
        points = row["points_cumules"]
        nb_equipes = row["nb_equipes"]
        ecart_premier = row.get("ecart_premier", 0)
        ecart_dernier = row.get("ecart_dernier", 0)

        # Déterminer le scénario
        scenario, urgence = _classify_scenario(
            position=position,
            nb_equipes=nb_equipes,
            ecart_premier=ecart_premier,
            ecart_dernier=ecart_dernier,
            points=points,
        )

        scenario_data.append(
            {
                "equipe": equipe,
                "saison": saison,
                "ronde": ronde,
                "position": position,
                "scenario": scenario,
                "urgence_score": urgence,
                "ecart_premier": ecart_premier,
                "ecart_dernier": ecart_dernier,
            }
        )

    result = pd.DataFrame(scenario_data)
    logger.info(f"  {len(result)} équipes avec scénario")
    return result


def _classify_scenario(
    position: int,
    nb_equipes: int,
    ecart_premier: int,
    ecart_dernier: int,
    points: int,
) -> tuple[str, float]:
    """Classifie le scénario d'une équipe.

    Returns
    -------
        Tuple (scenario, urgence_score)
    """
    # Zone relégation (2 derniers)
    zone_relegation = position > nb_equipes - 2

    # Course au titre (positions 1-2 ou proche)
    if position <= 2 or ecart_premier <= 2:
        return "course_titre", 0.9

    # Course montée (positions 3-4, barrage possible)
    if position <= 4 and ecart_premier <= 4:
        return "course_montee", 0.7

    # Condamné (dernier avec gros écart)
    if position == nb_equipes and ecart_dernier == 0 and ecart_premier >= 6:
        return "condamne", 0.3

    # Danger (zone rouge ou proche)
    if zone_relegation or ecart_dernier <= 2:
        return "danger", 0.85

    # Mi-tableau (confort)
    return "mi_tableau", 0.4
