"""Tests for ScenarioGenerator Phase 4a dispatch (T5b).

Uses a fake pool loader + a stubbed Phase 3 pipeline so the dispatch is tested
fast (no 92 s parquet load). Asserts: (a) simultaneous_teams=None -> Phase 3
path with no exclusion, (b) non-None -> joint_conditional exclusion applied to
the target pool, (c) target=team_1 excludes nothing.

Document ID: ALICE-TEST-GENERATOR-PHASE4A
Version: 1.0.0
"""

from __future__ import annotations

from typing import Any

import pytest

from services.ali.generator import ScenarioGenerator
from services.ali.types import CompetitionContext, PlayerCandidate, TeamSpec


def _player(nr: str, elo: int) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=nr,
        nom=nr,
        prenom="X",
        elo=elo,
        club="CLUB",
        mute=False,
        genre="M",
        categorie="SE",
        licence_active=True,
    )


class _SpyPoolLoader:
    """Records the exclude_players passed to load_pool, returns a fixed pool."""

    def __init__(self, pool: list[PlayerCandidate]) -> None:
        self._pool = pool
        self.seen_exclude: set[str] | None = None

    def load_pool(
        self,
        club_id: str,  # noqa: ARG002
        round_date: str,  # noqa: ARG002
        overrides: Any = None,  # noqa: ARG002
        exclude_players: set[str] | None = None,
    ) -> list[PlayerCandidate]:
        self.seen_exclude = exclude_players
        excl = exclude_players or set()
        return [p for p in self._pool if p.nr_ffe not in excl]


def _ctx() -> CompetitionContext:
    return CompetitionContext(
        competition_code="A02",
        niveau="N3",
        ronde=5,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
    )


def _teams() -> list[TeamSpec]:
    return [
        TeamSpec(team_name="CLUB 1", division="N1", board_count=8),
        TeamSpec(team_name="CLUB 2", division="N3", board_count=8, target_team=True),
    ]


def _gen_with_spy(
    monkeypatch: pytest.MonkeyPatch, pool: list[PlayerCandidate]
) -> tuple[ScenarioGenerator, _SpyPoolLoader]:
    """Build a generator whose heavy collaborators are stubbed, with a spy pool."""
    spy = _SpyPoolLoader(pool)
    gen = ScenarioGenerator.__new__(ScenarioGenerator)  # bypass __init__ I/O
    gen._engine = None  # type: ignore[attr-defined]
    gen._classifier = None  # type: ignore[attr-defined]
    gen._cache = None  # type: ignore[attr-defined]
    gen._pool_loader = spy  # type: ignore[attr-defined]
    gen._history_enricher = None  # type: ignore[attr-defined]
    gen._decay_lambda = 0.9  # type: ignore[attr-defined]

    from services.ali import generator as gmod

    def _fake_phase3(self: Any, pool_arg: list[PlayerCandidate], **kw: Any) -> Any:
        return {"pool_size": len(pool_arg)}

    monkeypatch.setattr(gmod.ScenarioGenerator, "_run_phase3", _fake_phase3, raising=True)
    return gen, spy


def test_bc_none_passes_empty_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = [_player(f"P{i:03d}", 2400 - i * 20) for i in range(20)]
    gen, spy = _gen_with_spy(monkeypatch, pool)
    out = gen.generate(
        opponent_club_id="CLUB",
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
        simultaneous_teams=None,
    )
    assert spy.seen_exclude in (None, set())
    assert out["pool_size"] == 20


def test_phase4a_excludes_superior_team_players(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = [_player(f"P{i:03d}", 2400 - i * 20) for i in range(24)]
    gen, spy = _gen_with_spy(monkeypatch, pool)
    out = gen.generate(
        opponent_club_id="CLUB",
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
        simultaneous_teams=_teams(),
        target_team="CLUB 2",
    )
    assert spy.seen_exclude is not None
    assert len(spy.seen_exclude) == 8  # CLUB 1 consumed 8 players
    assert out["pool_size"] == 16  # 24 - 8


def test_phase4a_target_first_excludes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = [_player(f"P{i:03d}", 2400 - i * 20) for i in range(24)]
    gen, spy = _gen_with_spy(monkeypatch, pool)
    out = gen.generate(
        opponent_club_id="CLUB",
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
        simultaneous_teams=_teams(),
        target_team="CLUB 1",
    )
    assert spy.seen_exclude == set()
    assert out["pool_size"] == 24


def test_phase4a_missing_target_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = [_player(f"P{i:03d}", 2400 - i * 20) for i in range(24)]
    gen, _spy = _gen_with_spy(monkeypatch, pool)
    with pytest.raises(ValueError, match="target_team required"):
        gen.generate(
            opponent_club_id="CLUB",
            round_date="2024-11-15",
            context=_ctx(),
            saison=2024,
            current_round=5,
            nb_rondes_total=11,
            simultaneous_teams=_teams(),
            target_team=None,
        )
