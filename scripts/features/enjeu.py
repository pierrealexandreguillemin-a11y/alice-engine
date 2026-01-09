"""Features zones d'enjeu équipe - ISO 5055/5259.

Ce module calcule les zones d'enjeu (promotion/maintien) par équipe.

Conformité:
- ISO 5055: Module <300 lignes, responsabilité unique
- ISO 5259: Zone basée sur position réelle
"""

from __future__ import annotations

import logging

import pandas as pd

from scripts.ffe_rules_features import calculer_zone_enjeu, get_niveau_equipe

logger = logging.getLogger(__name__)


def extract_team_enjeu_features(
    df: pd.DataFrame,
    standings: pd.DataFrame,
) -> pd.DataFrame:
    """Extrait les features de zone d'enjeu par équipe et saison.

    CORRIGÉ: Utilise position réelle calculée depuis les scores.

    Args:
    ----
        df: DataFrame échiquiers avec colonnes ronde, saison
        standings: DataFrame classement calculé

    Returns:
    -------
        DataFrame avec zone_enjeu par équipe/saison/ronde

    ISO 5259: Zone d'enjeu basée sur position réelle, pas estimée.
    """
    logger.info("Extraction features zones d'enjeu (position réelle)...")

    if df.empty:
        return pd.DataFrame()

    if "ronde" not in df.columns or "saison" not in df.columns:
        logger.warning("  Colonnes ronde/saison manquantes, skip zones enjeu")
        return pd.DataFrame()

    if standings.empty:
        logger.warning("  Classement vide, fallback estimation")
        return extract_team_enjeu_fallback(df)

    # Enrichir avec zone d'enjeu
    features_data = []

    for _, row in standings.iterrows():
        division = str(row["division"]) if row["division"] else "N4"
        zone = calculer_zone_enjeu(row["position"], row["nb_equipes"], division)

        features_data.append(
            {
                "equipe": row["equipe"],
                "saison": row["saison"],
                "competition": row["competition"],
                "division": row["division"],
                "groupe": row["groupe"],
                "ronde": row["ronde"],
                "position": row["position"],
                "points_cumules": row["points_cumules"],
                "nb_equipes": row["nb_equipes"],
                "ecart_premier": row["ecart_premier"],
                "ecart_dernier": row["ecart_dernier"],
                "zone_enjeu": zone,
                "niveau_hierarchique": get_niveau_equipe(str(row["equipe"])),
            }
        )

    result = pd.DataFrame(features_data)
    logger.info(f"  {len(result)} équipes/rondes avec zones enjeu réelles")

    return result


def extract_team_enjeu_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback si calcul classement impossible (données incomplètes).

    ISO 5259: Marque explicitement comme estimation (is_fallback=True).
    """
    logger.warning("  Utilisation fallback zone enjeu (estimation)")

    features_data = []

    for equipe_col in ["equipe_dom", "equipe_ext"]:
        if equipe_col not in df.columns:
            continue

        for (equipe, saison), group in df.groupby([equipe_col, "saison"]):
            division = str(equipe).split()[0] if equipe else "N4"
            niveau = get_niveau_equipe(str(equipe))
            nb_equipes = 10 if niveau <= 4 else 8

            # Fallback: position estimée mi-tableau
            position_estimee = nb_equipes // 2
            zone = calculer_zone_enjeu(position_estimee, nb_equipes, division)

            features_data.append(
                {
                    "equipe": equipe,
                    "saison": saison,
                    "zone_enjeu": zone,
                    "niveau_hierarchique": niveau,
                    "nb_rondes": group["ronde"].nunique() if "ronde" in group.columns else 0,
                    "position": position_estimee,
                    "nb_equipes": nb_equipes,
                    "is_fallback": True,  # ISO 5259: marqué comme estimation
                }
            )

    result = pd.DataFrame(features_data)
    if len(result) > 0:
        result = result.drop_duplicates(subset=["equipe", "saison"])

    return result
