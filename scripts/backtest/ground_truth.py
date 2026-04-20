"""Ground truth extraction : observed adversary lineup from echiquiers.parquet.

Plan 3 V2 T2. ISO 5259 (data lineage, no leakage), ISO 24029 (robustness).

For a (club_name, saison, ronde), extract the K joueurs who actually played
+ their echiquier assignment. Ground truth for backtest metrics T13-T17.

**Defense in depth (D-P3-14)** :
1. Filter via `blanc_equipe` / `noir_equipe` columns (data-driven, robust)
2. Validate FFE A02 §3.6 invariant : within a match, color alternates
   deterministically by echiquier parity (exploits domain knowledge as
   data-quality cross-check, ISO 24029)
Both layers combined : robust filter + fail-fast on data corruption.

Document ID: ALICE-BACKTEST-GROUND-TRUTH
Version: 1.1.0 (D-P3-14 : FFE invariant check)
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

    # D-P3-14 : FFE A02 §3.6 invariant cross-check (ISO 24029 data quality)
    _validate_ffe_color_invariant(match_rows, club_name, saison, ronde)

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


class FFEDataQualityError(ValueError):
    """Raised when echiquiers.parquet violates FFE A02 §3.6 color alternance invariant.

    ISO 24029 : fail-fast on data corruption to avoid silent propagation into metrics.
    """


def _validate_ffe_color_invariant(
    match_rows: object, club_name: str, saison: int, ronde: int
) -> None:
    """Check FFE A02 §3.6 invariant (color alternance by echiquier parity).

    Within one match, blanc_equipe must be consistent per echiquier parity
    (odd boards one team, even boards the other). Validates data integrity :
    if parquet scrape mis-attributed blanc_equipe, we catch it here rather
    than propagating silently into ground truth.

    @raises FFEDataQualityError on invariant violation.
    """
    odd_blancs = set()
    even_blancs = set()
    for _, row in match_rows.iterrows():  # type: ignore[attr-defined]
        ech = int(row["echiquier"])
        blanc_eq = str(row["blanc_equipe"])
        noir_eq = str(row["noir_equipe"])
        # Sanity : blanc_equipe ≠ noir_equipe (both teams on the board)
        if blanc_eq == noir_eq:
            msg = (
                f"FFE invariant violated [{club_name} saison={saison} ronde={ronde} "
                f"ech={ech}] : blanc_equipe == noir_equipe == '{blanc_eq}'"
            )
            raise FFEDataQualityError(msg)
        # Collect by parity
        if ech % 2 == 1:
            odd_blancs.add(blanc_eq)
        else:
            even_blancs.add(blanc_eq)
    # Alternance : tous odd boards ont le même blanc_equipe ; idem even
    if len(odd_blancs) > 1:
        msg = (
            f"FFE invariant violated [{club_name} saison={saison} ronde={ronde}] : "
            f"odd echiquiers have inconsistent blanc_equipe : {odd_blancs} "
            f"(A02 §3.6 requires strict alternance)"
        )
        raise FFEDataQualityError(msg)
    if len(even_blancs) > 1:
        msg = (
            f"FFE invariant violated [{club_name} saison={saison} ronde={ronde}] : "
            f"even echiquiers have inconsistent blanc_equipe : {even_blancs} "
            f"(A02 §3.6 requires strict alternance)"
        )
        raise FFEDataQualityError(msg)
    # Finally : odd_blancs ∩ even_blancs should be empty (colors must swap)
    if odd_blancs and even_blancs and odd_blancs == even_blancs:
        msg = (
            f"FFE invariant violated [{club_name} saison={saison} ronde={ronde}] : "
            f"odd and even echiquiers have same blanc_equipe {odd_blancs} "
            f"(A02 §3.6 requires color alternance)"
        )
        raise FFEDataQualityError(msg)


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
