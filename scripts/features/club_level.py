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

from scripts.features.helpers import exclude_forfeits
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

    df = exclude_forfeits(df_history)
    unified = _build_unified(df)
    if unified.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (club, saison), club_df in unified.groupby(["club", "saison"]):
        teams = club_df["equipe"].unique().tolist()
        avg_elo_by_team = (
            club_df.groupby("equipe")["elo"].mean().to_dict() if "elo" in club_df.columns else {}
        )
        ranks = _rank_teams_in_club(teams, avg_elo_by_team)
        nb_teams = len(teams)

        prev_season_df = unified[(unified["club"] == club) & (unified["saison"] == saison - 1)]

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

    nb_reinforced = 0
    for ronde in rondes:
        players_this_round = set(team_df[team_df["ronde"] == ronde]["joueur_nom"].dropna())
        # players who played for OTHER teams of same club THIS season
        other_team_players = set(club_df[club_df["equipe"] != team]["joueur_nom"].dropna())
        if players_this_round & other_team_players:
            nb_reinforced += 1

    return round(nb_reinforced / len(rondes), 3)


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

    Args:
    ----
        df_history: raw echiquiers DataFrame

    Returns:
    -------
        DataFrame with columns:
        - joueur_nom, equipe, saison, ronde
        - joueur_promu: playing for a stronger team than primary team
        - joueur_relegue: playing for a weaker team (reinforcement)
        - player_team_elo_gap: player Elo minus team avg Elo
    """
    if df_history.empty:
        return pd.DataFrame()

    df = exclude_forfeits(df_history)
    unified = _build_unified(df)
    if unified.empty:
        return pd.DataFrame()

    team_avg_elo = (
        unified.groupby(["equipe", "saison"])["elo"].mean().to_dict()
        if "elo" in unified.columns
        else {}
    )

    rows: list[dict] = []
    for (joueur, saison), player_df in unified.groupby(["joueur_nom", "saison"]):
        primary = _primary_team(player_df)
        primary_level = get_niveau_equipe(primary)

        for _, game_row in player_df.iterrows():
            current_team = str(game_row["equipe"])
            current_level = get_niveau_equipe(current_team)

            joueur_promu = current_level < primary_level
            joueur_relegue = current_level > primary_level

            player_elo = float(game_row.get("elo", float("nan")))
            team_avg = team_avg_elo.get((current_team, saison), float("nan"))
            gap = (
                round(player_elo - team_avg, 1)
                if not (pd.isna(player_elo) or pd.isna(team_avg))
                else float("nan")
            )

            rows.append(
                {
                    "joueur_nom": joueur,
                    "equipe": current_team,
                    "saison": saison,
                    "ronde": game_row.get("ronde"),
                    "joueur_promu": joueur_promu,
                    "joueur_relegue": joueur_relegue,
                    "player_team_elo_gap": gap,
                }
            )

    result = pd.DataFrame(rows)
    logger.info("  %d player-context rows computed", len(result))
    return result
