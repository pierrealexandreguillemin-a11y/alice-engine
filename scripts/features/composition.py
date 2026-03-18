"""Features de stratégie de composition et contexte match.

- Stratégie composition 100 pts (A02 Art. 3.6.e)
- Avantage domicile
- Titre FIDE numérique

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (<300 lignes, SRP)
- ISO/IEC 5259:2024 - Data Quality for ML
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TITLE_MAP = {"GM": 5, "IM": 4, "FM": 3, "CM": 2, "WGM": 4, "WIM": 3, "WFM": 2, "": 0}


def extract_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """Titre FIDE numérique + différentiel titre.

    GM=5, IM=4, FM=3, CM=2, WGM=4, WIM=3, WFM=2, ""=0.
    """
    result = pd.DataFrame(index=df.index)
    result["blanc_titre_num"] = df["blanc_titre"].map(TITLE_MAP).fillna(0).astype(int)
    result["noir_titre_num"] = df["noir_titre"].map(TITLE_MAP).fillna(0).astype(int)
    result["diff_titre"] = result["blanc_titre_num"] - result["noir_titre_num"]
    logger.info("  %d lignes avec features titre", len(result))
    return result


def extract_home_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Bool domicile pour le joueur blanc."""
    result = pd.DataFrame(index=df.index)
    result["est_domicile_blanc"] = (df["blanc_equipe"] == df["equipe_dom"]).astype(int)
    logger.info("  %d lignes avec feature domicile", len(result))
    return result


def extract_composition_strategy(df_played: pd.DataFrame) -> pd.DataFrame:
    """Décalage de position par rapport à l'ordre Elo strict.

    Pour chaque joueur dans un match, calcule la différence entre
    sa position réelle (échiquier) et sa position naturelle
    (rang par Elo décroissant dans la composition de son équipe).

    A02 Art. 3.6.e: inversion autorisée si écart < 100 pts.
    """
    # Construire les compositions dom + ext séparément
    dom = _composition_shift(df_played, "equipe_dom", "blanc")
    ext = _composition_shift(df_played, "equipe_ext", "noir")
    result = pd.concat([dom, ext], ignore_index=True)
    logger.info("  %d joueurs avec features composition", len(result))
    return result


def _composition_shift(
    df: pd.DataFrame,
    equipe_col: str,
    color: str,
) -> pd.DataFrame:
    """Calcule le décalage position pour une couleur/équipe."""
    elo_col = f"{color}_elo"
    nom_col = f"{color}_nom"

    # Filtrer les lignes de cette équipe/couleur avec Elo > 0
    mask = df[elo_col] > 0
    sub = df.loc[
        mask,
        [
            "saison",
            "competition",
            "division",
            "groupe",
            "ronde",
            equipe_col,
            "echiquier",
            elo_col,
            nom_col,
        ],
    ].copy()

    if sub.empty:
        return pd.DataFrame(
            columns=["nom", "decalage_position", "joueur_decale_haut", "joueur_decale_bas"]
        )

    # Rang naturel = rang par Elo décroissant DANS chaque composition
    group_cols = ["saison", "competition", "division", "groupe", "ronde", equipe_col]
    sub["rang_naturel"] = (
        sub.groupby(group_cols)[elo_col]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )
    sub["rang_reel"] = (
        sub.groupby(group_cols)["echiquier"]
        .rank(
            ascending=True,
            method="min",
        )
        .astype(int)
    )
    sub["decalage_position"] = sub["rang_reel"] - sub["rang_naturel"]

    return pd.DataFrame(
        {
            "nom": sub[nom_col].values,
            "decalage_position": sub["decalage_position"].values,
            "joueur_decale_haut": (sub["decalage_position"] < 0).astype(int).values,
            "joueur_decale_bas": (sub["decalage_position"] > 0).astype(int).values,
        }
    )
