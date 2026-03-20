"""Pipeline extended features — ALI, CE, temporelles (ISO 5055).

Branche les modules ALI (presence, patterns) et CE (scenarios) dans
le pipeline de feature engineering, plus les features temporelles
et contextuelles manquantes.

Nouvelles features (2026-03):
- rondes_manquees_consecutives_blanc/noir (absence patterns)
- taux_presence_global (3 dernieres saisons)
- match_important (top 4 ou zone relegation)

Conformite ISO/IEC:
- 5055: Code maintenable (<300 lignes, SRP)
- 5259: Qualite donnees ML (features documentees)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def extract_ali_features(df_history_played: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Extrait les features ALI (presence + patterns) depuis l'historique."""
    from scripts.features.ali.patterns import calculate_selection_patterns
    from scripts.features.ali.presence import calculate_presence_features

    features = {}

    # Presence: taux_presence_saison, derniere_presence, regularite
    presence = calculate_presence_features(df_history_played)
    if not presence.empty:
        # Agreger par joueur (derniere saison = plus pertinente)
        presence = presence.sort_values("saison").groupby("joueur_nom").last().reset_index()
        features["ali_presence"] = presence

    # Patterns: role_type, echiquier_prefere, flexibilite_echiquier
    patterns = calculate_selection_patterns(df_history_played)
    if not patterns.empty:
        patterns = patterns.sort_values("saison").groupby("joueur_nom").last().reset_index()
        features["ali_patterns"] = patterns

    # Absence patterns + taux_presence_global
    absence = extract_absence_features(df_history_played)
    if not absence.empty:
        features["ali_absence"] = absence

    return features


def extract_absence_features(df_history_played: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features d'absence par joueur (vectorise).

    Features:
    - rondes_manquees_consecutives: max rondes consecutives manquees
    - taux_presence_global: taux sur les 3 dernieres saisons

    Vectorized v2: groupby au lieu de boucle par joueur.
    """
    if df_history_played.empty:
        return pd.DataFrame()

    saisons_all = sorted(df_history_played["saison"].unique())
    saisons_3 = saisons_all[-3:] if len(saisons_all) >= 3 else saisons_all
    df3 = df_history_played[df_history_played["saison"].isin(saisons_3)]

    # Collect all player×saison×ronde appearances vectorized
    parts_all: list[pd.DataFrame] = []
    parts_3: list[pd.DataFrame] = []
    for col in ("blanc_nom", "noir_nom"):
        if col in df_history_played.columns:
            sub = df_history_played[[col, "saison", "ronde"]].dropna(subset=[col])
            parts_all.append(sub.rename(columns={col: "joueur_nom"}))
        if col in df3.columns:
            sub3 = df3[[col, "saison", "ronde"]].dropna(subset=[col])
            parts_3.append(sub3.rename(columns={col: "joueur_nom"}))

    if not parts_all:
        return pd.DataFrame()

    all_app = pd.concat(parts_all, ignore_index=True)

    # taux_presence_global (3 dernieres saisons)
    # Denominator = rondes per saison (not global) to handle varying group formats
    if parts_3:
        app_3 = pd.concat(parts_3, ignore_index=True)
        rondes_per_saison = app_3.groupby("saison")["ronde"].nunique().reset_index()
        rondes_per_saison.columns = ["saison", "nb_rondes_total"]
        # Per joueur×saison: rondes played
        joueur_saison = app_3.groupby(["joueur_nom", "saison"])["ronde"].nunique().reset_index()
        joueur_saison.columns = ["joueur_nom", "saison", "nb_jouees"]
        joueur_saison = joueur_saison.merge(rondes_per_saison, on="saison", how="left")
        joueur_saison["taux_saison"] = joueur_saison["nb_jouees"] / joueur_saison["nb_rondes_total"]
        # Mean across 3 seasons
        taux = joueur_saison.groupby("joueur_nom")["taux_saison"].mean().reset_index()
        taux.columns = ["joueur_nom", "taux_presence_global"]
        taux["taux_presence_global"] = taux["taux_presence_global"].round(3)
    else:
        taux = pd.DataFrame(columns=["joueur_nom", "taux_presence_global"])

    # rondes_manquees_consecutives: per-player universe based on own max ronde
    app_cons = pd.concat(parts_3, ignore_index=True) if parts_3 else all_app
    played_sets = app_cons.groupby("joueur_nom")["ronde"].agg(["max", set]).reset_index()
    played_sets.columns = ["joueur_nom", "max_ronde", "rondes_jouees"]
    played_sets["rondes_manquees_consecutives"] = played_sets.apply(
        lambda row: _max_consecutive_absences(
            list(range(1, int(row["max_ronde"]) + 1)), row["rondes_jouees"]
        ),
        axis=1,
    )

    result = played_sets[["joueur_nom", "rondes_manquees_consecutives"]].merge(
        taux[["joueur_nom", "taux_presence_global"]],
        on="joueur_nom",
        how="left",
    )
    result["taux_presence_global"] = result["taux_presence_global"].fillna(0.0)

    logger.info("  Absence features: %d joueurs calcules", len(result))
    return result


def _max_consecutive_absences(
    rondes_all: list[int],
    rondes_jouees: set[int],
) -> int:
    """Calcule la serie maximale d'absences consecutives."""
    if not rondes_all:
        return 0

    max_cons = 0
    current = 0
    for r in rondes_all:
        if r not in rondes_jouees:
            current += 1
            max_cons = max(max_cons, current)
        else:
            current = 0
    return max_cons


def extract_temporal_features(df_split: pd.DataFrame) -> None:
    """Ajoute les features temporelles directement sur le split (in-place).

    Features ajoutees:
    - phase_saison: debut (R1-R3), milieu (R4-R6), fin (R7+)
    - ronde_normalisee: ronde / max_ronde du groupe
    """
    if "ronde" not in df_split.columns:
        return

    df_split["phase_saison"] = pd.cut(
        df_split["ronde"],
        bins=[0, 3, 6, 99],
        labels=["debut", "milieu", "fin"],
    ).astype(str)

    # Ronde normalisee par groupe
    max_ronde = df_split.groupby(["saison", "competition", "division", "groupe"])[
        "ronde"
    ].transform("max")
    df_split["ronde_normalisee"] = (df_split["ronde"] / max_ronde).clip(0, 1)

    logger.info("  Features temporelles ajoutees: phase_saison, ronde_normalisee")


def extract_match_important(df_split: pd.DataFrame) -> None:
    """Ajoute match_important (bool→int) directement sur le split (in-place).

    Un match est important si l'une des deux equipes est en top 4 ou en
    zone de relegation (zone_enjeu_dom ou zone_enjeu_ext = 'montee' ou 'danger').

    Requiert que zone_enjeu_dom / zone_enjeu_ext soient deja merges.
    """
    zone_dom = "zone_enjeu_dom"
    zone_ext = "zone_enjeu_ext"

    if zone_dom not in df_split.columns and zone_ext not in df_split.columns:
        logger.warning("  match_important: zone_enjeu non disponible — skip")
        return

    enjeu_values = frozenset({"montee", "danger"})

    dom_flag = pd.Series(False, index=df_split.index)
    ext_flag = pd.Series(False, index=df_split.index)

    if zone_dom in df_split.columns:
        dom_flag = df_split[zone_dom].isin(enjeu_values)
    if zone_ext in df_split.columns:
        ext_flag = df_split[zone_ext].isin(enjeu_values)

    df_split["match_important"] = (dom_flag | ext_flag).astype(int)
    logger.info("  Feature match_important ajoutee")


def extract_adversaire_niveau(
    df_split: pd.DataFrame,
    standings: pd.DataFrame,
) -> None:
    """Ajoute la position actuelle de l'adversaire au classement (in-place)."""
    if standings.empty or "equipe_ext" not in df_split.columns:
        return

    # Position de l'equipe ext dans le classement
    if "equipe" in standings.columns and "position" in standings.columns:
        latest = standings.sort_values("ronde").groupby("equipe").last().reset_index()
        mapping = dict(zip(latest["equipe"], latest["position"], strict=False))
        df_split["adversaire_niveau_dom"] = df_split["equipe_ext"].map(mapping)
        df_split["adversaire_niveau_ext"] = df_split["equipe_dom"].map(mapping)
        logger.info("  Feature adversaire_niveau ajoutee")


def merge_ali_features(
    result: pd.DataFrame,
    features: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge les features ALI sur le DataFrame cible."""
    from scripts.features.merge_helpers import merge_player_features

    if "ali_presence" in features:
        result = merge_player_features(
            result,
            features["ali_presence"],
            ["taux_presence_saison", "derniere_presence", "regularite"],
        )

    if "ali_patterns" in features:
        result = merge_player_features(
            result,
            features["ali_patterns"],
            ["role_type", "echiquier_prefere", "flexibilite_echiquier"],
        )

    if "ali_absence" in features:
        result = merge_player_features(
            result,
            features["ali_absence"],
            ["rondes_manquees_consecutives", "taux_presence_global"],
        )

    return result
