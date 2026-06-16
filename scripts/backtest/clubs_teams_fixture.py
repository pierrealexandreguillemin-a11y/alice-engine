"""Team-name-keyed loader for config/clubs_teams_2024.json (Phase 4a T9).

Maps an echiquiers team_name (e.g. "Mulhouse 3") + ronde + match_date -> the
ordered list[TeamSpec] of the SIMULTANEOUS teams its club fields on that date,
top-down by division force (A02 §3.7.b) so joint_conditional.superior_teams()
correctly identifies the teams ranked above the target.

WHY team-name keying (controller diagnostic 2026-06-16): the fixture club KEY is
derived from the team name by stripping the trailing number token ("Clichy 2" ->
"Clichy"), whereas ALIDataCache.team_to_club maps team -> joueurs.club
("Clichy" -> "Clichy-Echecs-92"). Keying by joueurs.club matches ~8% of clubs;
keying by team_name (the fixture's stored team label) resolves 77% of N3 teams.
So we build a team_name -> club_key reverse index from the payload and resolve
via the opponent's exact team name.

CRITICAL (debt D-2026-06-10): filter by `date`, NOT `ronde` alone
(date_coherence_rate ~= 0.40 across divisions).

Document ID: ALICE-BACKTEST-CLUBS-TEAMS-FIXTURE
Version: 2.0.0
"""

from __future__ import annotations

from typing import Any

from services.ali.types import TeamSpec

# Descending force rank (A02 §3.7.b). Lower index = stronger. Jeunes/régionale/
# départemental/unknown -> _FORCE_DEFAULT (weakest); correct for an N3 target
# since only Top16/N1/N2 (rank < 3) are then treated as superior.
_FORCE_ORDER: dict[str, int] = {
    "Top 16": 0,
    "Top16": 0,
    "Nationale 1": 1,
    "Nationale 2": 2,
    "Nationale 3": 3,
    "Nationale 4": 4,
}
_FORCE_DEFAULT = 99


def _force_rank(division: str) -> int:
    return _FORCE_ORDER.get(division, _FORCE_DEFAULT)


def build_team_to_club_index(payload: dict[str, Any]) -> dict[str, str]:
    """Reverse index team_name -> club_key (one pass over the payload)."""
    index: dict[str, str] = {}
    for club, data in payload.get("clubs", {}).items():
        for entries in data.get("rondes", {}).values():
            for entry in entries:
                index[str(entry[0])] = club
    return index


def load_simultaneous_teams(
    payload: dict[str, Any],
    *,
    team_name: str,
    ronde: int,
    match_date: str,
    team_index: dict[str, str] | None = None,
) -> list[TeamSpec]:
    """Return the club (that fields `team_name`)'s TeamSpec list for `match_date`.

    Ordered top-down by division force. Empty list if `team_name` is absent
    from the fixture, the ronde is absent, or no entry matches the date. Pass a
    precomputed `team_index` (from `build_team_to_club_index`) to avoid rebuilding
    it per call in a loop.
    """
    index = team_index if team_index is not None else build_team_to_club_index(payload)
    club = index.get(team_name)
    if club is None:
        return []
    entries = payload["clubs"][club].get("rondes", {}).get(str(ronde), [])
    matched = [e for e in entries if str(e[3])[:10] == match_date]
    matched.sort(key=lambda e: (_force_rank(str(e[1])), str(e[0])))
    return [
        TeamSpec(team_name=str(e[0]), division=str(e[1]), board_count=int(e[2])) for e in matched
    ]
