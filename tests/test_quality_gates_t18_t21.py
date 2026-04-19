"""Quality gates T18-T21 explicites pour Phase 3 Plan 2 generator.

ISO 25059 : T18 sum(weights)=1, T19 len=20, T20 distincts,
T21 MC rejection_rate <= 30%.

Document ID: ALICE-TEST-QUALITY-GATES-T18-T21
Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from services.ali.generator import ScenarioGenerator
from services.ali.history import HistoryEnricher
from services.ali.joint_sampler import CopulaJointSampler
from services.ali.monte_carlo import MonteCarloSampler
from services.ali.pool_loader import PlayerPoolLoader
from services.ali.types import CompetitionContext
from services.ali.verifiability import VerifiabilityClassifier
from services.ffe.rule_engine import RuleEngine

if TYPE_CHECKING:
    from services.ali.cache import ALIDataCache


REAL_A02 = Path("config/ffe_rules/a02.json")
CLASSIF = Path("config/ffe_rules/alice_verifiability.json")


def _ctx(team_size: int = 8) -> CompetitionContext:
    return CompetitionContext(
        competition_code="A02",
        niveau="N2",
        ronde=5,
        team_size=team_size,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
    )


def _find_viable_club(cache: ALIDataCache, min_size: int = 20) -> str:
    for club_id, df in cache.joueurs_by_club.items():
        if len(df) >= min_size:
            return club_id
    pytest.skip(f"no club with >= {min_size} players")
    return ""  # unreachable


def _build_generator(cache: ALIDataCache) -> ScenarioGenerator:
    return ScenarioGenerator(
        engine=RuleEngine.from_json_file(REAL_A02),
        classifier=VerifiabilityClassifier.from_json_file(CLASSIF),
        cache=cache,
        pool_loader=PlayerPoolLoader(cache),
        history_enricher=HistoryEnricher(cache, decay_lambda=0.9),
    )


def test_t18_weights_sum_equals_one(ali_data_cache: ALIDataCache) -> None:
    """T18 : sum(scenario.weight) == 1.0 +/- 1e-4."""
    gen = _build_generator(ali_data_cache)
    club = _find_viable_club(ali_data_cache)
    result = gen.generate(
        opponent_club_id=club,
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
    )
    weights_sum = sum(s.weight for s in result.scenarios)
    assert abs(weights_sum - 1.0) < 1e-4, f"T18 fail: sum={weights_sum}"


def test_t19_scenarios_count_exactly_20(ali_data_cache: ALIDataCache) -> None:
    """T19 : len(ScenarioSet.scenarios) == 20."""
    gen = _build_generator(ali_data_cache)
    club = _find_viable_club(ali_data_cache)
    result = gen.generate(
        opponent_club_id=club,
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
    )
    assert len(result.scenarios) == 20, f"T19 fail: count={len(result.scenarios)}"


def test_t20_scenarios_all_distinct(ali_data_cache: ALIDataCache) -> None:
    """T20 : tous les 20 scenarios distincts par signature (player x board)."""
    gen = _build_generator(ali_data_cache)
    club = _find_viable_club(ali_data_cache)
    result = gen.generate(
        opponent_club_id=club,
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
    )
    sigs = {
        tuple((a.player.nr_ffe, a.board) for a in s.lineup.assignments) for s in result.scenarios
    }
    assert len(sigs) == 20, f"T20 fail: {len(sigs)} distinct out of 20"


def test_t21_mc_rejection_rate_within_threshold(ali_data_cache: ALIDataCache) -> None:
    """T21 : MonteCarloSampler.last_rejection_rate <= 0.30."""
    cache = ali_data_cache
    club = _find_viable_club(cache)
    pool_loader = PlayerPoolLoader(cache)
    enricher = HistoryEnricher(cache, decay_lambda=0.9)
    pool = pool_loader.load_pool(club, "2024-11-15")
    enriched = enricher.enrich(
        pool,
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
    )

    # Fit copula on history for enriched pool
    copula = CopulaJointSampler(decay_lambda=0.9)
    names = [f"{c.nom} {c.prenom}".strip() for c in enriched]
    history = cache.lookup_history(names)
    parts: list[pd.DataFrame] = []
    for col in ("blanc_nom", "noir_nom"):
        if col in history.columns:
            sub = history[[col, "saison", "ronde", "echiquier"]].copy()
            sub = sub.rename(columns={col: "joueur_nom"})
            parts.append(sub)
    history_norm = (
        pd.concat(parts, ignore_index=True).drop_duplicates(
            subset=["joueur_nom", "saison", "ronde"],
        )
        if parts
        else pd.DataFrame(columns=["joueur_nom", "saison", "ronde", "echiquier"])
    )
    copula.fit(
        history=history_norm,
        player_names=names,
        saison=2024,
        nb_rondes_total=11,
        current_round=5,
    )

    # Sample with MC
    engine = RuleEngine.from_json_file(REAL_A02)
    mc = MonteCarloSampler(engine=engine, copula=copula)
    rng = np.random.default_rng(42)
    mc.sample(enriched, _ctx(), n_pairs=10, rng=rng)

    rate = mc.last_rejection_rate
    assert 0.0 <= rate <= 0.30, f"T21 fail: rejection_rate={rate}"
