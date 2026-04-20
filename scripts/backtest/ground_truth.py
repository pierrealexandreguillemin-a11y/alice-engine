"""Ground truth extraction : observed adversary lineup from echiquiers.parquet.

Plan 3 V2 T2. ISO 5259 (data lineage, no leakage).

For a (club_name, saison, ronde), extract the K joueurs who actually played
+ their echiquier assignment. Ground truth for backtest metrics T13-T17.

Uses `blanc_equipe` / `noir_equipe` columns to identify which player belongs
to the target club on each board (robust — no reliance on FFE board color
parity convention). Each board is played by exactly one player of each team,
so per match we emit one ObservedPlayer per echiquier (the club's player).

Document ID: ALICE-BACKTEST-GROUND-TRUTH
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.ali.cache import ALIDataCache


@dataclass(frozen=True)
class ObservedPlayer:
    """One observed player at a specific board for a match."""

    joueur_nom: str
    echiquier: int
    elo: int


@dataclass(frozen=True)
class ObservedLineup:
    """Complete observed lineup for a match (sorted by echiquier ascending)."""

    club_name: str
    saison: int
    ronde: int
    players: tuple[ObservedPlayer, ...]

    def player_names(self) -> frozenset[str]:
        """Return the set of observed player names (dedup)."""
        return frozenset(p.joueur_nom for p in self.players)


def extract_observed_lineup(
    cache: ALIDataCache,
    club_name: str,
    saison: int,
    ronde: int,
    *,
    as_domicile: bool | None = None,
) -> ObservedLineup:
    """Extract real lineup from echiquiers.parquet for (club, saison, ronde).

    For each board of the match, the club's player is the one whose team
    (`blanc_equipe` or `noir_equipe`) equals ``club_name``.

    @param cache: ALIDataCache exposing ``echiquiers_total``.
    @param club_name: team name as stored in ``equipe_dom`` / ``equipe_ext``.
    @param saison: FFE season (year).
    @param ronde: round number (1..nb_rondes).
    @param as_domicile: if True, filter matches where club is home;
        if False, filter matches where club is away; if None, try home first
        then away.
    @raises KeyError: no match found for the given club/saison/ronde.
    """
    df = cache.echiquiers_total
    sub = df[(df["saison"] == saison) & (df["ronde"] == ronde)]
    if sub.empty:
        msg = f"No match found for saison={saison} ronde={ronde}"
        raise KeyError(msg)

    match_rows, is_home = _select_match_rows(sub, club_name, as_domicile)
    if match_rows.empty:
        msg = (
            f"Club '{club_name}' not found in saison={saison} ronde={ronde} "
            f"(as_domicile={as_domicile})"
        )
        raise KeyError(msg)

    players = _extract_players(match_rows, is_home=is_home)
    if not players:
        msg = (
            f"No player rows matched club='{club_name}' via blanc_equipe/noir_equipe "
            f"for saison={saison} ronde={ronde}"
        )
        raise KeyError(msg)

    players.sort(key=lambda p: p.echiquier)
    return ObservedLineup(
        club_name=club_name,
        saison=saison,
        ronde=ronde,
        players=tuple(players),
    )


def _select_match_rows(
    sub: object,  # pd.DataFrame, kept as object to avoid runtime import
    club_name: str,
    as_domicile: bool | None,
) -> tuple[object, bool]:
    """Filter ``sub`` to a single match (home or away). Returns (rows, is_home)."""
    empty = sub.iloc[0:0]  # type: ignore[attr-defined]
    if as_domicile is True:
        home = sub[sub["equipe_dom"] == club_name]  # type: ignore[index]
        return home, True
    if as_domicile is False:
        away = sub[sub["equipe_ext"] == club_name]  # type: ignore[index]
        return away, False
    # None → try home first, then away
    home = sub[sub["equipe_dom"] == club_name]  # type: ignore[index]
    if not home.empty:
        return home, True
    away = sub[sub["equipe_ext"] == club_name]  # type: ignore[index]
    if not away.empty:
        return away, False
    return empty, True


def _extract_players(match_rows: object, *, is_home: bool) -> list[ObservedPlayer]:
    """For every board row, pick the club's player using ``blanc_equipe`` / ``noir_equipe``."""
    target_col_values = (
        match_rows["equipe_dom"] if is_home else match_rows["equipe_ext"]  # type: ignore[index]
    )
    target_club = str(target_col_values.iloc[0])  # type: ignore[attr-defined]
    players: list[ObservedPlayer] = []
    for _, row in match_rows.iterrows():  # type: ignore[attr-defined]
        if str(row["blanc_equipe"]) == target_club:
            name_raw, elo_raw = row["blanc_nom"], row["blanc_elo"]
        elif str(row["noir_equipe"]) == target_club:
            name_raw, elo_raw = row["noir_nom"], row["noir_elo"]
        else:
            continue
        name = str(name_raw).strip()
        if not name:
            continue
        try:
            elo = int(elo_raw)
        except (TypeError, ValueError):
            elo = 0
        players.append(
            ObservedPlayer(joueur_nom=name, echiquier=int(row["echiquier"]), elo=elo),
        )
    return players
