"""Tests for services.ali.monte_carlo — MonteCarloSampler (F5 SOTA).

ISO 29119 : fixtures, reproductibilite (seed), coverage.
ISO 5055 : SRP, per-assertion clarity.
ISO 42001 : SOTA documented (McKay 1979, Hammersley & Morton 1956).

Document ID: ALICE-TEST-ALI-MONTE-CARLO
Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.ali.joint_sampler import CopulaJointSampler
from services.ali.monte_carlo import MonteCarloSampler
from services.ali.types import CompetitionContext, PlayerCandidate
from services.ffe.rule_engine import RuleEngine

REAL_A02 = Path("config/ffe_rules/a02.json")


def _player(nr: str, elo: int, taux: float = 0.7) -> PlayerCandidate:
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
        taux_presence_effectif=taux,
    )


def _ctx(team_size: int = 8) -> CompetitionContext:
    return CompetitionContext(
        competition_code="A02",
        niveau="N3",
        ronde=3,
        team_size=team_size,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
    )


def _build_sampler(
    pool_size: int = 12,
) -> tuple[MonteCarloSampler, list[PlayerCandidate], CopulaJointSampler]:
    """Build helpers : pool, fitted copula, MC sampler."""
    pool = [_player(f"P{i}", 2200 - i * 30, taux=0.8 - i * 0.03) for i in range(pool_size)]
    # Build a fake history to fit the copula
    rows = []
    for i in range(pool_size):
        for r in range(1, 6):
            if (i + r) % 3 != 0:
                rows.append((f"P{i} X", 2024, r, (i % 8) + 1))
    history = pd.DataFrame(rows, columns=["joueur_nom", "saison", "ronde", "echiquier"])
    copula = CopulaJointSampler(decay_lambda=0.9)
    copula.fit(
        history=history,
        player_names=[f"P{i} X" for i in range(pool_size)],
        saison=2024,
        nb_rondes_total=5,
        current_round=6,
    )
    engine = RuleEngine.from_json_file(REAL_A02)
    mc = MonteCarloSampler(engine=engine, copula=copula)
    return mc, pool, copula


def test_mc_returns_n_scenarios():
    mc, pool, _ = _build_sampler(pool_size=14)
    rng = np.random.default_rng(42)
    scenarios = mc.sample(pool, _ctx(team_size=8), n_pairs=5, rng=rng)
    assert len(scenarios) == 10  # 5 pairs * 2


def test_mc_reproducibility_same_seed():
    mc, pool, _ = _build_sampler(pool_size=14)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    s1 = mc.sample(pool, _ctx(team_size=8), n_pairs=3, rng=rng1)
    s2 = mc.sample(pool, _ctx(team_size=8), n_pairs=3, rng=rng2)
    sigs1 = [tuple(a.player.nr_ffe for a in s.lineup.assignments) for s in s1]
    sigs2 = [tuple(a.player.nr_ffe for a in s.lineup.assignments) for s in s2]
    assert sigs1 == sigs2


def test_mc_lineup_size_matches_team_size():
    mc, pool, _ = _build_sampler(pool_size=14)
    rng = np.random.default_rng(0)
    scenarios = mc.sample(pool, _ctx(team_size=8), n_pairs=3, rng=rng)
    for s in scenarios:
        assert s.lineup.team_size == 8
        assert len(s.lineup.assignments) == 8


def test_mc_source_is_monte_carlo():
    mc, pool, _ = _build_sampler(pool_size=14)
    rng = np.random.default_rng(0)
    scenarios = mc.sample(pool, _ctx(team_size=8), n_pairs=2, rng=rng)
    for s in scenarios:
        assert s.source == "monte_carlo"


def test_mc_rejection_rate_observable():
    mc, pool, _ = _build_sampler(pool_size=14)
    rng = np.random.default_rng(0)
    mc.sample(pool, _ctx(team_size=8), n_pairs=3, rng=rng)
    rate = mc.last_rejection_rate
    assert 0.0 <= rate <= 1.0


def test_mc_weights_normalized():
    mc, pool, _ = _build_sampler(pool_size=14)
    rng = np.random.default_rng(0)
    scenarios = mc.sample(pool, _ctx(team_size=8), n_pairs=5, rng=rng)
    weights_sum = sum(s.weight for s in scenarios)
    assert abs(weights_sum - 1.0) < 1e-4


def test_mc_pool_too_small_raises():
    mc, pool, _ = _build_sampler(pool_size=5)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="too small"):
        mc.sample(pool, _ctx(team_size=8), n_pairs=3, rng=rng)


def test_mc_antithetic_pairs_property():
    """Pairs antithetic : un scenario sur deux est constructed from antithetic uniform."""
    mc, pool, _ = _build_sampler(pool_size=14)
    rng = np.random.default_rng(42)
    scenarios = mc.sample(pool, _ctx(team_size=8), n_pairs=5, rng=rng)
    assert len(scenarios) == 10  # toujours pair


def test_mc_n_pairs_zero_returns_empty():
    mc, pool, _ = _build_sampler(pool_size=14)
    rng = np.random.default_rng(0)
    scenarios = mc.sample(pool, _ctx(team_size=8), n_pairs=0, rng=rng)
    assert scenarios == []


def test_mc_unfit_copula_raises_runtime_error() -> None:
    """D-P3-12 fix : copula non-fittee -> RuntimeError fail-fast (ISO 24029).

    Avant 2026-04-28, _u_to_presence avait un fallback indep silencieux qui
    masquait un wiring incorrect (generator devant toujours fit la copula
    avant sample). Maintenant raise pour eviter biais conservateur cache.
    """
    pool = [_player(f"P{i}", 2200 - i * 30, taux=0.7) for i in range(12)]
    engine = RuleEngine.from_json_file(REAL_A02)
    unfit_copula = CopulaJointSampler()  # no fit() call
    mc = MonteCarloSampler(engine=engine, copula=unfit_copula)
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="copula not fit"):
        mc.sample(pool, _ctx(team_size=8), n_pairs=3, rng=rng)


def test_mc_copula_size_mismatch_raises_runtime_error() -> None:
    """D-P3-12 fix : copula fit on different pool size -> RuntimeError."""
    pool_12 = [_player(f"P{i}", 2200 - i * 30, taux=0.7) for i in range(12)]
    pool_14 = [_player(f"P{i}", 2200 - i * 30, taux=0.7) for i in range(14)]
    engine = RuleEngine.from_json_file(REAL_A02)
    # Fit copula on pool_14 then use with pool_12
    rows = [
        {"saison": 2024, "ronde": r, "joueur_nom": f"PP{i} X", "present": 1}
        for r in range(1, 11)
        for i in range(14)
    ]
    history = pd.DataFrame(rows)
    copula = CopulaJointSampler()
    copula.fit(
        history=history,
        player_names=[f"PP{i} X" for i in range(14)],
        saison=2024,
        nb_rondes_total=11,
        current_round=10,
    )
    mc = MonteCarloSampler(engine=engine, copula=copula)
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="n_players"):
        mc.sample(pool_12, _ctx(team_size=8), n_pairs=3, rng=rng)
