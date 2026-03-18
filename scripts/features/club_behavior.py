"""Features comportement club — stabilite effectif, noyau, rotation.

Features specifiees dans FEATURE_SPECIFICATION.md §10.

Conformite ISO/IEC:
- 5055: Code maintenable (<300 lignes, SRP)
- 5259: Qualite donnees ML
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def extract_club_behavior(df_history: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features de comportement du club par saison.

    Returns
    -------
        DataFrame avec colonnes: equipe, saison,
        nb_joueurs_utilises, rotation_effectif, noyau_stable, profondeur_effectif
    """
    rows = []
    for (equipe, saison), group in df_history.groupby(["equipe_dom", "saison"]):
        _process_club(group, str(equipe), int(saison), "blanc", rows)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.groupby(["equipe", "saison"]).first().reset_index()
    logger.info("  %d clubs avec features comportement", len(result))
    return result


def _process_club(
    group: pd.DataFrame,
    equipe: str,
    saison: int,
    color: str,
    rows: list,
) -> None:
    """Traite un club pour une saison."""
    nom_col = f"{color}_nom"
    if nom_col not in group.columns:
        return

    nb_rondes = group["ronde"].nunique()
    if nb_rondes == 0:
        return

    joueurs = group[nom_col].value_counts()
    nb_joueurs = len(joueurs)

    # Noyau stable = joueurs presents > 80% des rondes
    noyau = int((joueurs >= nb_rondes * 0.8).sum())

    # Rotation = joueurs differents par ronde en moyenne
    rotation = group.groupby("ronde")[nom_col].nunique().mean()

    rows.append(
        {
            "equipe": equipe,
            "saison": saison,
            "nb_joueurs_utilises": nb_joueurs,
            "rotation_effectif": round(float(rotation), 2),
            "noyau_stable": noyau,
            "profondeur_effectif": nb_joueurs,
        }
    )
