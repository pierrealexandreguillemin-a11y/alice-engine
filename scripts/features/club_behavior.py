"""Features comportement club — stabilite effectif, noyau, rotation.

Features specifiees dans FEATURE_SPECIFICATION.md §10.
Ajout features §14 (renforce_fin_saison, avantage_dom_club,
club_utilise_marge_100).

Conformite ISO/IEC:
- 5055: Code maintenable (<300 lignes, SRP)
- 5259: Qualite donnees ML
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_PLAYED_RESULTS = frozenset(
    {
        "victoire_blanc",
        "victoire_noir",
        "nulle",
        "victoire_blanc_ajournement",
        "victoire_noir_ajournement",
    }
)


def extract_club_behavior(df_history: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features de comportement du club par saison.

    Returns
    -------
        DataFrame avec colonnes: equipe, saison,
        nb_joueurs_utilises, rotation_effectif, noyau_stable,
        profondeur_effectif, renforce_fin_saison,
        avantage_dom_club, club_utilise_marge_100
    """
    if df_history.empty or "equipe_dom" not in df_history.columns:
        return pd.DataFrame()

    rows = []
    for (equipe, saison), group in df_history.groupby(["equipe_dom", "saison"]):
        _process_club(group, str(equipe), int(saison), rows)

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
    rows: list,
) -> None:
    """Traite un club (equipe_dom) pour une saison."""
    nom_col = "blanc_nom"
    if nom_col not in group.columns:
        return

    nb_rondes = group["ronde"].nunique()
    if nb_rondes == 0:
        return

    joueurs = group[nom_col].value_counts()
    nb_joueurs = len(joueurs)
    noyau = int((joueurs >= nb_rondes * 0.8).sum())
    rotation = group.groupby("ronde")[nom_col].nunique().mean()

    renforce = _calc_renforce_fin_saison(group)
    avantage_dom = _calc_avantage_dom(group)
    marge_100 = _calc_marge_100(group)

    rows.append(
        {
            "equipe": equipe,
            "saison": saison,
            "nb_joueurs_utilises": nb_joueurs,
            "rotation_effectif": round(float(rotation), 2),
            "noyau_stable": noyau,
            "profondeur_effectif": nb_joueurs,
            "renforce_fin_saison": renforce,
            "avantage_dom_club": avantage_dom,
            "club_utilise_marge_100": marge_100,
        }
    )


def _calc_renforce_fin_saison(group: pd.DataFrame) -> int:
    """Bool: le club joue un ELO moyen plus eleve en fin saison (R7+) qu'en debut (R1-R3)."""
    elo_col = "blanc_elo"
    if elo_col not in group.columns or "ronde" not in group.columns:
        return 0

    debut = group[group["ronde"] <= 3][elo_col].mean()
    fin = group[group["ronde"] >= 7][elo_col].mean()

    if pd.isna(debut) or pd.isna(fin):
        return 0
    return int(fin > debut)


def _calc_avantage_dom(group: pd.DataFrame) -> float:
    """Taux de victoire a domicile (parties jouees uniquement)."""
    if "type_resultat" not in group.columns:
        return 0.0

    played = group[group["type_resultat"].isin(_PLAYED_RESULTS)]
    if len(played) == 0:
        return 0.0

    victoires_dom = (
        played["type_resultat"].isin({"victoire_blanc", "victoire_blanc_ajournement"})
    ).sum()
    return round(float(victoires_dom) / len(played), 3)


def _calc_marge_100(group: pd.DataFrame) -> float:
    """Ratio de compositions ou le club utilise la marge 100 pts (decalages)."""
    if "blanc_elo" not in group.columns or "echiquier" not in group.columns:
        return 0.0

    rondes = group["ronde"].unique()
    if len(rondes) == 0:
        return 0.0

    nb_utilise = 0
    for ronde in rondes:
        match = group[group["ronde"] == ronde].sort_values("echiquier")
        if len(match) < 2:
            continue
        if _has_inversion(match):
            nb_utilise += 1

    return round(nb_utilise / len(rondes), 3)


def _has_inversion(match: pd.DataFrame) -> bool:
    """Verifie si le classement Elo n'est pas respecte (inversion < 100 pts)."""
    elos = match["blanc_elo"].fillna(0).tolist()
    for i in range(len(elos) - 1):
        if elos[i] < elos[i + 1] and (elos[i + 1] - elos[i]) < 100:
            return True
    return False
