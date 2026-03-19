"""Noyau equipe - features reglementaires FFE (A02 Art. 3.7.f) - ISO 5055.

Features:
- est_dans_noyau_blanc/noir (bool→int): joueur a-t-il deja joue pour cette
  equipe cette saison avant cette ronde ?
- pct_noyau_equipe_dom/ext (float 0-1): % de la composition actuelle qui
  est dans le noyau de l'equipe.

Conformite ISO/IEC:
- 5055: Module <300 lignes, SRP, fonctions <50 lignes
- 5259: Qualite donnees ML

Document ID: ALICE-FEA-NOYAU-001
Version: 1.0.0
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def extract_noyau_features(df_history_played: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features noyau par partie.

    Pour chaque ligne (blanc_nom, noir_nom, equipe, ronde, saison),
    indique si le joueur a deja joue pour cette equipe durant les rondes
    precedentes de la meme saison.

    Args:
    ----
        df_history_played: DataFrame parties jouees (sans forfaits)

    Returns:
    -------
        DataFrame avec colonnes: joueur_nom, equipe, saison, ronde,
        est_dans_noyau, pct_noyau_match.
        Chaque joueur peut apparaitre plusieurs fois (une par partie).
    """
    if df_history_played.empty:
        return pd.DataFrame()

    rows = _collect_noyau_rows(df_history_played)
    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    logger.info("  Noyau: %d entrees joueur/match calculees", len(result))
    return result


def _collect_noyau_rows(df: pd.DataFrame) -> list[dict]:
    """Collecte les donnees noyau pour chaque joueur par match."""
    rows: list[dict] = []

    for (saison, equipe, ronde), group in df.groupby(["saison", "equipe_dom", "ronde"]):
        _process_match_noyau(df, group, saison, str(equipe), int(ronde), "blanc", rows)

    for (saison, equipe, ronde), group in df.groupby(["saison", "equipe_ext", "ronde"]):
        _process_match_noyau(df, group, saison, str(equipe), int(ronde), "noir", rows)

    return rows


def _process_match_noyau(
    df_all: pd.DataFrame,
    group: pd.DataFrame,
    saison: int,
    equipe: str,
    ronde: int,
    color: str,
    rows: list[dict],
) -> None:
    """Traite un match pour extraire les features noyau."""
    nom_col = f"{color}_nom"
    if nom_col not in group.columns:
        return

    equipe_col = "equipe_dom" if color == "blanc" else "equipe_ext"

    # Joueurs ayant joue pour cette equipe dans les rondes PRECEDENTES
    prior = df_all[
        (df_all["saison"] == saison)
        & (df_all[equipe_col] == equipe)
        & (df_all["ronde"] < ronde)
        & (df_all[nom_col].notna())
    ]
    noyau_set = set(prior[nom_col].unique())

    # Joueurs dans la composition actuelle
    joueurs_match = group[nom_col].dropna().tolist()
    nb_match = len(joueurs_match)
    if nb_match == 0:
        return

    nb_noyau = sum(1 for j in joueurs_match if j in noyau_set)
    pct = nb_noyau / nb_match

    for joueur in joueurs_match:
        rows.append(
            {
                "joueur_nom": joueur,
                "equipe": equipe,
                "saison": saison,
                "ronde": ronde,
                "est_dans_noyau": int(joueur in noyau_set),
                "pct_noyau_match": round(pct, 3),
            }
        )
