"""Features comportement club — stabilite effectif, noyau, rotation.

Features specifiees dans FEATURE_SPECIFICATION.md §10.
Ajout features §14 (renforce_fin_saison, win_rate_home, draw_rate_home,
club_utilise_marge_100).

Conformite ISO/IEC:
- 5055: Code maintenable (<300 lignes, SRP)
- 5259: Qualite donnees ML — forfaits exclus via filter_played_games()
"""

from __future__ import annotations

import logging

import pandas as pd

from scripts.features.helpers import filter_played_games

logger = logging.getLogger(__name__)


def extract_club_behavior(df_history: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features de comportement du club par saison.

    Returns
    -------
        DataFrame avec colonnes: equipe, saison,
        nb_joueurs_utilises, rotation_effectif, noyau_stable,
        profondeur_effectif, renforce_fin_saison,
        win_rate_home, draw_rate_home, club_utilise_marge_100
    """
    if df_history.empty or "equipe_dom" not in df_history.columns:
        return pd.DataFrame()

    unified = _build_unified_view(df_history)
    if unified.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (equipe, saison), group in unified.groupby(["equipe", "saison"]):
        _process_club(group, str(equipe), int(saison), rows)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.groupby(["equipe", "saison"]).first().reset_index()
    logger.info("  %d clubs avec features comportement", len(result))
    return result


def _build_unified_view(df: pd.DataFrame) -> pd.DataFrame:
    """Build unified view with both home and away perspectives."""
    parts = []
    # Home: equipe_dom, blanc_nom, blanc_elo
    if "equipe_dom" in df.columns and "blanc_nom" in df.columns:
        home = df[["equipe_dom", "saison", "ronde", "blanc_nom", "echiquier"]].copy()
        home.columns = ["equipe", "saison", "ronde", "joueur_nom", "echiquier"]
        if "blanc_elo" in df.columns:
            home["elo"] = df["blanc_elo"].values
        if "type_resultat" in df.columns:
            home["type_resultat"] = df["type_resultat"].values
            home["resultat_blanc"] = df["resultat_blanc"].values
            home["is_home"] = True
        parts.append(home)
    # Away: equipe_ext, noir_nom, noir_elo
    if "equipe_ext" in df.columns and "noir_nom" in df.columns:
        away = df[["equipe_ext", "saison", "ronde", "noir_nom", "echiquier"]].copy()
        away.columns = ["equipe", "saison", "ronde", "joueur_nom", "echiquier"]
        if "noir_elo" in df.columns:
            away["elo"] = df["noir_elo"].values
        if "type_resultat" in df.columns:
            away["type_resultat"] = df["type_resultat"].values
            away["resultat_blanc"] = df["resultat_blanc"].values
            away["is_home"] = False
        parts.append(away)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _process_club(
    group: pd.DataFrame,
    equipe: str,
    saison: int,
    rows: list,
) -> None:
    """Traite un club pour une saison (unified view)."""
    if "joueur_nom" not in group.columns:
        return

    nb_rondes = group["ronde"].nunique()
    if nb_rondes == 0:
        return

    joueurs = group["joueur_nom"].value_counts()
    nb_joueurs = len(joueurs)
    noyau = int((joueurs >= nb_rondes * 0.8).sum())
    rotation = group.groupby("ronde")["joueur_nom"].nunique().mean()

    renforce = _calc_renforce_fin_saison(group)
    win_rate_home, draw_rate_home = _calc_home_rates(group)
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
            "win_rate_home": win_rate_home,
            "draw_rate_home": draw_rate_home,
            "club_utilise_marge_100": marge_100,
        }
    )


def _calc_renforce_fin_saison(group: pd.DataFrame) -> int:
    """Bool: le club joue un ELO moyen plus eleve en fin saison (R7+) qu'en debut (R1-R3)."""
    elo_col = "elo"
    if elo_col not in group.columns or "ronde" not in group.columns:
        return 0

    debut = group[group["ronde"] <= 3][elo_col].mean()
    fin = group[group["ronde"] >= 7][elo_col].mean()

    if pd.isna(debut) or pd.isna(fin):
        return 0
    return int(fin > debut)


def _calc_home_rates(group: pd.DataFrame) -> tuple[float, float]:
    """W/D rates a domicile (parties jouees uniquement — forfaits exclus ISO 5259)."""
    if (
        "type_resultat" not in group.columns
        or "is_home" not in group.columns
        or "resultat_blanc" not in group.columns
    ):
        return 0.0, 0.0

    home_all = group[group["is_home"]].copy()
    if home_all.empty:
        return 0.0, 0.0

    # Exclude forfeits using filter_played_games (needs type_resultat + resultat_blanc)
    home = filter_played_games(home_all)
    n = len(home)
    if n == 0:
        return 0.0, 0.0

    # Home team is always blanc (equipe_dom plays blanc)
    wins = (home["resultat_blanc"] == 1.0).sum()
    draws = (home["resultat_blanc"] == 0.5).sum()
    return round(float(wins) / n, 3), round(float(draws) / n, 3)


def _calc_marge_100(group: pd.DataFrame) -> float:
    """Ratio de compositions ou le club utilise la marge 100 pts (decalages)."""
    if "elo" not in group.columns or "echiquier" not in group.columns:
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
    elos = match["elo"].fillna(0).tolist()
    for i in range(len(elos) - 1):
        if elos[i] < elos[i + 1] and (elos[i + 1] - elos[i]) < 100:
            return True
    return False
