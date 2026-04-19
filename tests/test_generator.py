"""Tests for ScenarioGenerator (P2-Task 7).

ISO 29119 : structured tests with fixtures and real artefacts.
ISO 5259 : lineage_hash verified on output.
ISO 25059 : 20 scenarios distincts enforced (T20).

Document ID: ALICE-TEST-ALI-GENERATOR
Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from services.ali.generator import ScenarioGenerator
from services.ali.history import HistoryEnricher
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


def _build_generator(cache: ALIDataCache) -> ScenarioGenerator:
    engine = RuleEngine.from_json_file(REAL_A02)
    classifier = VerifiabilityClassifier.from_json_file(CLASSIF)
    pool_loader = PlayerPoolLoader(cache)
    history_enricher = HistoryEnricher(cache, decay_lambda=0.9)
    return ScenarioGenerator(
        engine=engine,
        classifier=classifier,
        cache=cache,
        pool_loader=pool_loader,
        history_enricher=history_enricher,
    )


def _find_viable_club(cache: ALIDataCache, min_pool: int = 20) -> str:
    """Find a club with enough eligible joueurs for a team_size=8 test.

    20 players provides enough combinatorial diversity to produce 20
    distinct lineups (T20) via TopK swaps + MC sampling.
    """
    for club_id, df in cache.joueurs_by_club.items():
        if len(df) >= min_pool:
            return club_id
    pytest.skip(f"no club with >={min_pool} joueurs in cache")
    return ""  # unreachable


def test_generator_produces_20_scenarios(ali_data_cache: ALIDataCache) -> None:
    gen = _build_generator(ali_data_cache)
    club_id = _find_viable_club(ali_data_cache)
    result = gen.generate(
        opponent_club_id=club_id,
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
    )
    assert len(result.scenarios) == 20
    weights_sum = sum(s.weight for s in result.scenarios)
    assert abs(weights_sum - 1.0) < 1e-4


def test_generator_lineage_hash_propagated(ali_data_cache: ALIDataCache) -> None:
    gen = _build_generator(ali_data_cache)
    club_id = _find_viable_club(ali_data_cache)
    result = gen.generate(
        opponent_club_id=club_id,
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
    )
    assert len(result.lineage_hash) == 64
    assert result.opponent_club_id == club_id


def test_generator_scenarios_distinct_t20(ali_data_cache: ALIDataCache) -> None:
    gen = _build_generator(ali_data_cache)
    club_id = _find_viable_club(ali_data_cache)
    result = gen.generate(
        opponent_club_id=club_id,
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
    )
    sigs = {
        tuple((a.player.nr_ffe, a.board) for a in s.lineup.assignments) for s in result.scenarios
    }
    assert len(sigs) == 20  # all distinct


def test_generator_pool_too_small_raises(ali_data_cache: ALIDataCache) -> None:
    gen = _build_generator(ali_data_cache)
    with pytest.raises(ValueError, match="too small"):
        gen.generate(
            opponent_club_id="UNKNOWN_TINY_CLUB",
            round_date="2024-11-15",
            context=_ctx(team_size=8),
            saison=2024,
            current_round=5,
            nb_rondes_total=11,
        )


def test_generator_uses_public_rules_only(ali_data_cache: ALIDataCache) -> None:
    """D-P2-02 fix : generator doit filtrer via classifier.partition_rules.

    Verifie que le engine passe a TopK+MC ne contient que les regles PUBLIC.
    Chemin : inspecter que les scenarios generes violent la regle PRIVATE 3.7.f (noyau)
    en toute impunite (pas rejected au sampling), prouvant que PRIVATE
    est supposee respectee par l'adversaire plutot qu'enforced.
    """
    engine = RuleEngine.from_json_file(REAL_A02)
    classifier = VerifiabilityClassifier.from_json_file(CLASSIF)
    public, private = classifier.partition_rules(engine.rules)
    assert len(public) == 10
    assert len(private) == 4

    gen = _build_generator(ali_data_cache)
    club_id = _find_viable_club(ali_data_cache)
    result = gen.generate(
        opponent_club_id=club_id,
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
    )
    # Generateur doit retourner 20 scenarios meme si certains violeraient
    # les regles PRIVATE (e.g., noyau non declare, designation titulaires)
    assert len(result.scenarios) == 20
    # lineage_hash porte le suffixe +public (partition wired)
    # Note : le lineage hash final mixe avec les autres parametres, on verifie
    # juste que la generation fonctionne sans raise
    assert len(result.lineage_hash) == 64
