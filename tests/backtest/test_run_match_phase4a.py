"""Tests for Phase 4a backward-compatible threading of run_backtest_match (T9.3).

Thread round_date + simultaneous_teams + target_team through run_backtest_match.

Verifies that:
- Phase 3 default path (no extra args) still uses the "{saison}-09-01" fake date
  and does NOT forward simultaneous_teams / target_team.
- Phase 4a path forwards the real round_date, simultaneous_teams, and target_team
  verbatim to ScenarioGenerator.generate(...).

Document ID: ALICE-BACKTEST-T9-PHASE4A
Version: 1.0.0
Count: 2 test cases
"""

from __future__ import annotations

from typing import Any

from scripts.backtest.run_match import run_backtest_match
from services.ali.types import TeamSpec


class _SpyGenerator:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return _FakeScenarioSet()


class _FakeScenarioSet:
    lineage_hash = "deadbeef"
    scenarios: list[Any] = []


class _Noop:
    def __getattr__(self, _name: str) -> Any:  # aggregate_from_scenarios no-ops on empty set
        return lambda *a, **k: []


def _run(**extra: Any) -> _SpyGenerator:
    gen = _SpyGenerator()
    run_backtest_match(
        user_club_id="A",
        opponent_club_id="B",
        saison=2024,
        ronde=3,
        nb_rondes_total=11,
        division="N3",
        team_size=8,
        user_lineup=[],
        scenario_generator=gen,
        inference=_Noop(),
        feature_store=_Noop(),
        strict=False,
        **extra,
    )
    return gen


def test_phase3_default_uses_fake_date_and_no_sim_teams() -> None:
    gen = _run()
    assert gen.kwargs["round_date"] == "2024-09-01"
    assert gen.kwargs.get("simultaneous_teams") is None
    assert gen.kwargs.get("target_team") is None


def test_phase4a_forwards_real_date_and_sim_teams() -> None:
    sim = [
        TeamSpec(team_name="B 1", division="Nationale 1", board_count=8),
        TeamSpec(team_name="B 3", division="Nationale 3", board_count=8),
    ]
    gen = _run(round_date="2024-11-17", simultaneous_teams=sim, target_team="B 3")
    assert gen.kwargs["round_date"] == "2024-11-17"
    assert gen.kwargs["simultaneous_teams"] == sim
    assert gen.kwargs["target_team"] == "B 3"
