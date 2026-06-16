"""Tests for scripts/backtest/clubs_teams_fixture.py (Phase 4a T9.2).

Document ID: ALICE-BACKTEST-CLUBS-TEAMS-FIXTURE-TEST
Version: 1.0.0
Count: 7 unit tests — pure-function, no I/O, inline payloads only.
"""

from __future__ import annotations

from scripts.backtest.clubs_teams_fixture import (
    build_team_to_club_index,
    load_simultaneous_teams,
)

_PAYLOAD = {
    "clubs": {
        "Mulhouse": {
            "rondes": {
                "3": [
                    ["Mulhouse 3", "Nationale 3", 8, "2024-11-17"],
                    ["Mulhouse 1", "Nationale 1", 8, "2024-11-17"],
                    ["Mulhouse 2", "Nationale 2", 8, "2024-11-17"],
                    ["Mulhouse 4", "Nationale 4", 8, "2024-11-10"],  # different date -> excluded
                ]
            }
        }
    }
}


def test_resolves_club_by_team_name_and_orders_top_down() -> None:
    teams = load_simultaneous_teams(
        _PAYLOAD, team_name="Mulhouse 3", ronde=3, match_date="2024-11-17"
    )
    # club resolved from "Mulhouse 3"; date filter drops "Mulhouse 4"; force N1 > N2 > N3
    assert [t.team_name for t in teams] == ["Mulhouse 1", "Mulhouse 2", "Mulhouse 3"]
    assert teams[0].division == "Nationale 1"


def test_target_team_present_in_result() -> None:
    # superior_teams() requires the target to be in the list (else ValueError).
    teams = load_simultaneous_teams(
        _PAYLOAD, team_name="Mulhouse 3", ronde=3, match_date="2024-11-17"
    )
    assert "Mulhouse 3" in [t.team_name for t in teams]


def test_empty_when_team_absent() -> None:
    assert (
        load_simultaneous_teams(_PAYLOAD, team_name="Ghost 1", ronde=3, match_date="2024-11-17")
        == []
    )


def test_empty_when_no_entry_matches_date() -> None:
    assert (
        load_simultaneous_teams(_PAYLOAD, team_name="Mulhouse 3", ronde=3, match_date="2024-12-01")
        == []
    )


def test_reverse_index_maps_every_team_to_its_club() -> None:
    idx = build_team_to_club_index(_PAYLOAD)
    assert idx["Mulhouse 1"] == "Mulhouse"
    assert idx["Mulhouse 4"] == "Mulhouse"


def test_empty_when_ronde_absent() -> None:
    # docstring contract: empty list when the ronde key is absent
    assert (
        load_simultaneous_teams(_PAYLOAD, team_name="Mulhouse 3", ronde=99, match_date="2024-11-17")
        == []
    )


def test_precomputed_index_produces_same_result() -> None:
    # the team_index fast-path (used in the pilot loop) must match the auto-build path
    idx = build_team_to_club_index(_PAYLOAD)
    teams_auto = load_simultaneous_teams(
        _PAYLOAD, team_name="Mulhouse 3", ronde=3, match_date="2024-11-17"
    )
    teams_precomputed = load_simultaneous_teams(
        _PAYLOAD, team_name="Mulhouse 3", ronde=3, match_date="2024-11-17", team_index=idx
    )
    assert teams_auto == teams_precomputed
