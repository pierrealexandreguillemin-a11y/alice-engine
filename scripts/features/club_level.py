"""Features vases communiquants — niveaux equipes, mouvements joueurs.

Document ID: ALICE-FEAT-CLUB-LEVEL  Version: 1.0.0

A player descending from N1 to N3 is a strong reinforcement signal.
Public API: extract_club_level_features, extract_player_team_context.

ISO: 5055 (SRP <300 lines), 5259 (forfait exclusion), 42001 (traceability).
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from scripts.features.helpers import filter_played_games
from scripts.ffe_rules_features import get_niveau_equipe

logger = logging.getLogger(__name__)

# -- Private helpers ---------------------------------------------------------

_DIVISION_PATTERN = re.compile(
    r"\s*[-–]?\s*(top\s*\d+f?|nationale?\s*\d*f?|n\d+f?|r[123]|"
    r"regionale?|departemental|groupe\s+\w+)\b.*$",
    re.IGNORECASE,
)


def _extract_club_name(team_name: str) -> str:
    """Extract club name — strips division suffix and trailing number.

    'Marseille Echecs 1 N1' -> 'Marseille Echecs', 'Paris - N2 groupe A' -> 'Paris'.
    """
    clean = _DIVISION_PATTERN.sub("", str(team_name))
    clean = re.sub(r"\s+\d+$", "", clean)
    return clean.strip()


def _build_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten home/away into unified (equipe, saison, ronde, joueur_nom, elo) rows."""
    parts: list[pd.DataFrame] = []
    if "equipe_dom" in df.columns and "blanc_nom" in df.columns:
        home = df[["equipe_dom", "saison", "ronde", "blanc_nom"]].copy()
        home.columns = ["equipe", "saison", "ronde", "joueur_nom"]
        if "blanc_elo" in df.columns:
            home["elo"] = df["blanc_elo"].values
        parts.append(home)
    if "equipe_ext" in df.columns and "noir_nom" in df.columns:
        away = df[["equipe_ext", "saison", "ronde", "noir_nom"]].copy()
        away.columns = ["equipe", "saison", "ronde", "joueur_nom"]
        if "noir_elo" in df.columns:
            away["elo"] = df["noir_elo"].values
        parts.append(away)
    if not parts:
        return pd.DataFrame()

    unified = pd.concat(parts, ignore_index=True)
    unified = unified[unified["joueur_nom"].notna()].copy()
    unified["club"] = unified["equipe"].apply(_extract_club_name)
    return unified


def _rank_teams_in_club(teams: list[str], avg_elo: dict[str, float]) -> dict[str, int]:
    """Rank teams by get_niveau_equipe, breaking ties by avg Elo (higher = better)."""

    def sort_key(t: str) -> tuple[int, float]:
        return (get_niveau_equipe(t), -avg_elo.get(t, 0.0))

    sorted_teams = sorted(teams, key=sort_key)
    return {team: rank + 1 for rank, team in enumerate(sorted_teams)}


def _primary_team(player_df: pd.DataFrame) -> str:
    """Player's primary team = mode of equipe (most games played)."""
    mode_val = player_df["equipe"].mode()
    return str(mode_val.iloc[0]) if len(mode_val) > 0 else ""


# -- Public: extract_club_level_features ------------------------------------


def extract_club_level_features(df_history: pd.DataFrame) -> pd.DataFrame:
    """Compute per (equipe, saison) club hierarchy features.

    Args:
    ----
        df_history: raw echiquiers DataFrame

    Returns:
    -------
        DataFrame with columns:
        - equipe, saison
        - team_rank_in_club: 1=fanion (strongest), ascending
        - club_nb_teams: total teams from this club this season
        - reinforcement_rate: fraction of rounds with an outside-club player
        - stabilite_effectif: fraction of players who also played last season
        - elo_moyen_evolution: avg Elo delta (last round minus first round)
    """
    if df_history.empty:
        return pd.DataFrame()

    df = filter_played_games(df_history)
    unified = _build_unified(df)
    if unified.empty:
        return pd.DataFrame()

    # Pre-index by (club, saison) for O(1) prev-season lookup
    club_saison_groups = unified.groupby(["club", "saison"])
    prev_season_players: dict[tuple, set] = {}
    for (club, saison), grp in club_saison_groups:
        prev_season_players[(club, saison)] = set(grp["joueur_nom"].dropna())

    rows: list[dict] = []
    for (club, saison), club_df in club_saison_groups:
        teams = club_df["equipe"].unique().tolist()
        avg_elo_by_team = (
            club_df.groupby("equipe")["elo"].mean().to_dict() if "elo" in club_df.columns else {}
        )
        ranks = _rank_teams_in_club(teams, avg_elo_by_team)
        nb_teams = len(teams)

        prev_players = prev_season_players.get((club, saison - 1), set())
        prev_season_df = (
            pd.DataFrame({"joueur_nom": list(prev_players)}) if prev_players else pd.DataFrame()
        )

        for team in teams:
            team_df = club_df[club_df["equipe"] == team]
            row = _compute_team_row(
                team=str(team),
                saison=int(saison),
                team_df=team_df,
                club_df=club_df,
                prev_season_df=prev_season_df,
                rank=ranks[str(team)],
                nb_teams=nb_teams,
            )
            rows.append(row)

    result = pd.DataFrame(rows)
    logger.info("  %d (equipe, saison) avec club_level features", len(result))
    return result


def _compute_team_row(
    team: str,
    saison: int,
    team_df: pd.DataFrame,
    club_df: pd.DataFrame,
    prev_season_df: pd.DataFrame,
    rank: int,
    nb_teams: int,
) -> dict:
    """Compute feature dict for one (team, saison) group."""
    reinforcement_rate = _calc_reinforcement_rate(team, team_df, club_df)
    stabilite = _calc_stabilite(team_df, prev_season_df)
    elo_evolution = _calc_elo_evolution(team_df)

    return {
        "equipe": team,
        "saison": saison,
        "team_rank_in_club": rank,
        "club_nb_teams": nb_teams,
        "reinforcement_rate": reinforcement_rate,
        "stabilite_effectif": stabilite,
        "elo_moyen_evolution": elo_evolution,
    }


def _calc_reinforcement_rate(team: str, team_df: pd.DataFrame, club_df: pd.DataFrame) -> float:
    """Fraction of rounds where >= 1 player from another club team appeared."""
    if "ronde" not in team_df.columns:
        return 0.0

    rondes = team_df["ronde"].unique()
    if len(rondes) == 0:
        return 0.0

    # Pre-compute once (invariant across rondes)
    other_team_players = set(club_df.loc[club_df["equipe"] != team, "joueur_nom"].dropna())
    if not other_team_players:
        return 0.0

    # Vectorized: group players by ronde and check intersection
    ronde_groups = team_df.groupby("ronde")["joueur_nom"].apply(
        lambda x: bool(set(x.dropna()) & other_team_players)
    )
    return round(ronde_groups.sum() / len(rondes), 3)


def _calc_stabilite(team_df: pd.DataFrame, prev_df: pd.DataFrame) -> float:
    """Fraction of this season's players who also played last season."""
    current_players = set(team_df["joueur_nom"].dropna())
    if not current_players or prev_df.empty:
        return 0.0

    prev_players = set(prev_df["joueur_nom"].dropna())
    return round(len(current_players & prev_players) / len(current_players), 3)


def _calc_elo_evolution(team_df: pd.DataFrame) -> float:
    """Avg Elo delta: last round - first round (positive = reinforcement)."""
    if "elo" not in team_df.columns or "ronde" not in team_df.columns:
        return 0.0

    rondes = sorted(team_df["ronde"].dropna().unique())
    if len(rondes) < 2:
        return 0.0

    elo_first = team_df[team_df["ronde"] == rondes[0]]["elo"].mean()
    elo_last = team_df[team_df["ronde"] == rondes[-1]]["elo"].mean()

    if pd.isna(elo_first) or pd.isna(elo_last):
        return 0.0

    return round(float(elo_last - elo_first), 1)


# -- Public: extract_player_team_context ------------------------------------


def extract_player_team_context(df_history: pd.DataFrame) -> pd.DataFrame:
    """Compute per (joueur_nom, equipe, saison, ronde) movement flags.

    Fully vectorized — no iterrows(). Args/Returns unchanged.
    """
    if df_history.empty:
        return pd.DataFrame()

    df = filter_played_games(df_history)
    unified = _build_unified(df)
    if unified.empty:
        return pd.DataFrame()

    # 1. Team avg Elo per (equipe, saison)
    team_avg_elo = (
        unified.groupby(["equipe", "saison"])["elo"].mean()
        if "elo" in unified.columns
        else pd.Series(dtype=float)
    )

    # 2. Primary team per (joueur_nom, saison) = mode of equipe
    primary_teams = (
        unified.groupby(["joueur_nom", "saison"])["equipe"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "")
        .rename("primary_team")
    )

    # 3. Niveau for each equipe (vectorized via map)
    unique_teams = unified["equipe"].unique()
    niveau_map = {t: get_niveau_equipe(t) for t in unique_teams}

    # 4. Join primary team onto unified
    result = unified[["joueur_nom", "equipe", "saison", "ronde"]].copy()
    if "elo" in unified.columns:
        result["elo"] = unified["elo"].values

    result = result.merge(primary_teams, on=["joueur_nom", "saison"], how="left")

    # 5. Map niveaux
    result["current_level"] = result["equipe"].map(niveau_map)
    result["primary_level"] = result["primary_team"].map(niveau_map)

    # 6. Promu/relegue (lower level number = stronger)
    result["joueur_promu"] = result["current_level"] < result["primary_level"]
    result["joueur_relegue"] = result["current_level"] > result["primary_level"]

    # 7. Player-team Elo gap (merge is safer than MultiIndex.map)
    if "elo" in result.columns and not team_avg_elo.empty:
        team_avg_df = team_avg_elo.reset_index(name="team_avg")
        result = result.merge(team_avg_df, on=["equipe", "saison"], how="left")
        result["player_team_elo_gap"] = (result["elo"] - result["team_avg"]).round(1)
        result = result.drop(columns=["elo", "team_avg"])
    else:
        result["player_team_elo_gap"] = float("nan")

    result = result.drop(columns=["primary_team", "current_level", "primary_level"])
    logger.info("  %d player-context rows computed", len(result))
    return result
