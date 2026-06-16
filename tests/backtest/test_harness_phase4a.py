"""Tests Phase 4a forwarding of round_date/simultaneous_teams/target_team.

Document ID: ALICE-TEST-HARNESS-PHASE4A
Version: 1.0.0
Count: 2
ISO 29119: Unit tests for BacktestHarness.run_match Phase 4a keyword arg forwarding.
Verifies that the thin harness wrapper correctly propagates Phase 4a context
args to run_backtest_match → scenario_generator.generate without changing
Phase 3 default behaviour (defaults None → run_backtest_match fills saison-09-01).
"""

from __future__ import annotations

from typing import Any

from scripts.backtest.harness import BacktestHarness
from services.ali.types import TeamSpec


class _SpyGenerator:
    """Spy stub: captures kwargs forwarded to generate()."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return _FakeScenarioSet()


class _FakeScenarioSet:
    """Minimal ScenarioSet stub with lineage_hash + empty scenarios."""

    lineage_hash = "deadbeef"

    def __init__(self) -> None:
        self.scenarios: list[Any] = []


class _Noop:
    """Catch-all stub: any attribute access returns a no-op callable returning []."""

    def __getattr__(self, _name: str) -> Any:
        return lambda *a, **k: []


def _harness_with_fakes() -> tuple[BacktestHarness, _SpyGenerator]:
    """Build harness with fakes injected directly (bypasses setup() / IO)."""
    h = BacktestHarness()
    gen = _SpyGenerator()
    h.scenario_generator = gen  # type: ignore[assignment]
    h.inference = _Noop()  # type: ignore[assignment]
    h.feature_store = _Noop()  # type: ignore[assignment]
    h.cache = _Noop()  # type: ignore[assignment]
    return h, gen


def _run(h: BacktestHarness, **extra: Any) -> None:
    """Thin helper: call run_match with mandatory Phase 3 args + optional extras."""
    h.run_match(
        user_club_id="A",
        opponent_club_id="B",
        saison=2024,
        ronde=3,
        nb_rondes_total=11,
        division="N3",
        team_size=8,
        user_lineup=[],
        strict=False,
        **extra,
    )


def test_harness_phase3_defaults_forward_none() -> None:
    """Phase 3 call (no Phase 4a args): run_backtest_match fills round_date from saison."""
    h, gen = _harness_with_fakes()
    _run(h)
    # run_backtest_match resolves None → "{saison}-09-01" before calling generate
    assert gen.kwargs["round_date"] == "2024-09-01"
    assert gen.kwargs.get("simultaneous_teams") is None
    assert gen.kwargs.get("target_team") is None


def test_harness_forwards_phase4a_args() -> None:
    """Phase 4a call: harness must forward round_date, simultaneous_teams, target_team."""
    h, gen = _harness_with_fakes()
    sim = [
        TeamSpec(team_name="B 1", division="Nationale 1", board_count=8),
        TeamSpec(team_name="B 3", division="Nationale 3", board_count=8),
    ]
    _run(h, round_date="2024-11-17", simultaneous_teams=sim, target_team="B 3")
    assert gen.kwargs["round_date"] == "2024-11-17"
    assert gen.kwargs["simultaneous_teams"] == sim
    assert gen.kwargs["target_team"] == "B 3"
