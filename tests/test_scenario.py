"""Tests for ALI scenario frozen dataclasses (Plan 2 / P2-Task 3).

ISO 29119 : immutable dataclasses, value-equality, structural validation.
ISO 5055 : SRP (1 test = 1 invariant).
ISO 5259 : lineage_hash propagation verified via ScenarioSet contract.

Document ID: ALICE-TEST-ALI-SCENARIO
Version: 1.0.0
Test count: 6
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from services.ali.scenario import (
    BoardAssignment,
    Lineup,
    Scenario,
    ScenarioSet,
)
from services.ali.types import PlayerCandidate


def _player(nr: str, elo: int) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=nr,
        nom=f"P{nr}",
        prenom="X",
        elo=elo,
        club="C1",
        mute=False,
        genre="M",
        categorie="SE",
        licence_active=True,
    )


def _make_lineup(team_size: int = 3) -> Lineup:
    assignments = tuple(
        BoardAssignment(
            board=i + 1,
            player=_player(f"X{i}", 2000 - i * 50),
            p_assignment=0.7,
        )
        for i in range(team_size)
    )
    return Lineup(team_size=team_size, assignments=assignments)


def test_board_assignment_frozen() -> None:
    ba = BoardAssignment(board=1, player=_player("A1", 2000), p_assignment=0.8)
    assert ba.board == 1
    with pytest.raises(FrozenInstanceError):
        ba.board = 2  # type: ignore[misc]


def test_lineup_immutable_assignments() -> None:
    lineup = _make_lineup(3)
    assert lineup.team_size == 3
    assert len(lineup.assignments) == 3
    assert isinstance(lineup.assignments, tuple)


def test_scenario_creation() -> None:
    lineup = _make_lineup(3)
    s = Scenario(lineup=lineup, joint_prob=0.5, weight=0.05, source="topk")
    assert s.source == "topk"
    assert s.weight == 0.05


def test_scenario_set_validate_ok() -> None:
    lineup = _make_lineup(3)
    scenarios = tuple(
        Scenario(lineup=lineup, joint_prob=0.5, weight=0.05, source="topk") for _ in range(20)
    )
    ss = ScenarioSet(
        scenarios=scenarios,
        opponent_club_id="C99",
        round_date="2024-11-15",
        generated_at="2026-04-19T10:00:00Z",
        lineage_hash="a" * 64,
    )
    ss.validate()  # should not raise


def test_scenario_set_validate_wrong_count() -> None:
    lineup = _make_lineup(3)
    scenarios = tuple(
        Scenario(lineup=lineup, joint_prob=0.5, weight=0.1, source="topk") for _ in range(10)
    )
    ss = ScenarioSet(
        scenarios=scenarios,
        opponent_club_id="C99",
        round_date="2024-11-15",
        generated_at="2026-04-19T10:00:00Z",
        lineage_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="20"):
        ss.validate()


def test_scenario_set_validate_weights_sum_off() -> None:
    lineup = _make_lineup(3)
    scenarios = tuple(
        Scenario(lineup=lineup, joint_prob=0.5, weight=0.5, source="topk") for _ in range(20)
    )
    ss = ScenarioSet(
        scenarios=scenarios,
        opponent_club_id="C99",
        round_date="2024-11-15",
        generated_at="2026-04-19T10:00:00Z",
        lineage_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="weights"):
        ss.validate()
