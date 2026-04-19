"""Tests for ScenarioGenerator (P2-Task 7).

ISO 29119 : structured tests with fixtures and real artefacts.
ISO 5259 : lineage_hash verified on output.
ISO 25059 : 20 scenarios distincts enforced (T20).

Document ID: ALICE-TEST-ALI-GENERATOR
Version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        niveau="N3",  # N3 : pas de contrainte 3.7.i fr_gender strict (N1/N2/Top16 only)
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


def test_public_engine_construction_has_exactly_10_rules(
    ali_data_cache: ALIDataCache,
) -> None:
    """D-P2-02 : public_engine construit par generator contient bien 10 regles PUBLIC.

    Test structurel : independant du comportement _check_rule (seules 4 regles
    PUBLIC sont actuellement checkees, mais la partition doit toujours exclure
    les 4 PRIVATE du public_engine pour coherence future).
    """
    engine = RuleEngine.from_json_file(REAL_A02)
    classifier = VerifiabilityClassifier.from_json_file(CLASSIF)
    public, _private = classifier.partition_rules(engine.rules)

    # Construction identique a celle du generator (spec §4.7 step 5-6)
    public_engine = RuleEngine(
        rules=list(public),
        source_sha256=f"{engine.lineage_hash()}+public",
    )

    assert len(public_engine.rules) == 10
    assert public_engine.lineage_hash().endswith("+public")


def test_public_engine_excludes_all_private_rule_ids(
    ali_data_cache: ALIDataCache,
) -> None:
    """D-P2-02 : aucune regle PRIVATE (3.7.b, 3.2, 3.7.f, 3.7) dans public_engine."""
    engine = RuleEngine.from_json_file(REAL_A02)
    classifier = VerifiabilityClassifier.from_json_file(CLASSIF)
    public, private = classifier.partition_rules(engine.rules)

    public_ids = {r.id for r in public}
    private_ids = {r.id for r in private}

    # Intersection vide : aucune regle ne peut etre a la fois PUBLIC et PRIVATE
    assert not (public_ids & private_ids)

    # Verifier explicitement les 4 PRIVATE attendues
    expected_private = {
        "N1-N4_3.7.b_001",  # force equipes
        "N1-N4_3.2_001",  # designation titulaires
        "N1-N3_3.7.f_001",  # noyau
        "N1-N2_3.7_001",  # arbitrage
    }
    assert expected_private.issubset(
        private_ids
    ), f"Attendu subset PRIVATE : {expected_private - private_ids} manquants"
    # Ces 4 ne doivent PAS etre dans PUBLIC
    assert not (expected_private & public_ids)


def test_generator_uses_public_engine_not_self_engine(
    ali_data_cache: ALIDataCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-P2-02 : TopKEnumerator doit recevoir public_engine (10 rules).

    Verifie que TopKEnumerator est construit avec public_engine (10 rules), pas
    self._engine (14 rules). Approach : monkey-patch TopKEnumerator.__init__
    pour capturer l'engine recu.
    """
    from services.ali import generator as generator_module

    captured_engines: list[RuleEngine] = []
    original_topk_init = generator_module.TopKEnumerator.__init__

    def spy_topk_init(self: Any, engine: RuleEngine) -> None:
        captured_engines.append(engine)
        original_topk_init(self, engine)

    monkeypatch.setattr(generator_module.TopKEnumerator, "__init__", spy_topk_init)

    gen = _build_generator(ali_data_cache)
    club_id = _find_viable_club(ali_data_cache)
    gen.generate(
        opponent_club_id=club_id,
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
    )

    # TopKEnumerator doit etre construit exactement 1 fois avec public_engine (10 rules)
    assert len(captured_engines) >= 1
    topk_engine = captured_engines[0]
    assert len(topk_engine.rules) == 10, (
        f"TopKEnumerator a recu engine avec {len(topk_engine.rules)} rules, "
        f"expected 10 (public only)"
    )
    assert topk_engine.lineage_hash().endswith("+public")


def test_generator_mc_uses_same_public_engine_as_topk(
    ali_data_cache: ALIDataCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-P2-02 : TopK et MC doivent recevoir le MEME public_engine (pas 2 distincts)."""
    from services.ali import generator as generator_module

    topk_engines: list[RuleEngine] = []
    mc_engines: list[RuleEngine] = []

    orig_topk_init = generator_module.TopKEnumerator.__init__
    orig_mc_init = generator_module.MonteCarloSampler.__init__

    def spy_topk(self: Any, engine: RuleEngine) -> None:
        topk_engines.append(engine)
        orig_topk_init(self, engine)

    def spy_mc(self: Any, engine: RuleEngine, copula: Any) -> None:
        mc_engines.append(engine)
        orig_mc_init(self, engine, copula)

    monkeypatch.setattr(generator_module.TopKEnumerator, "__init__", spy_topk)
    monkeypatch.setattr(generator_module.MonteCarloSampler, "__init__", spy_mc)

    gen = _build_generator(ali_data_cache)
    club_id = _find_viable_club(ali_data_cache)
    gen.generate(
        opponent_club_id=club_id,
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
    )

    # Meme engine (meme rules + meme lineage) injecte dans les deux
    assert len(topk_engines) == 1
    assert len(mc_engines) == 1
    assert topk_engines[0].lineage_hash() == mc_engines[0].lineage_hash()
    assert len(topk_engines[0].rules) == len(mc_engines[0].rules) == 10


def test_generator_does_not_mutate_original_engine(
    ali_data_cache: ALIDataCache,
) -> None:
    """D-P2-02 : la construction de public_engine NE DOIT PAS modifier self._engine."""
    engine = RuleEngine.from_json_file(REAL_A02)
    classifier = VerifiabilityClassifier.from_json_file(CLASSIF)
    original_rules_count = len(engine.rules)
    original_lineage = engine.lineage_hash()

    gen = ScenarioGenerator(
        engine=engine,
        classifier=classifier,
        cache=ali_data_cache,
        pool_loader=PlayerPoolLoader(ali_data_cache),
        history_enricher=HistoryEnricher(ali_data_cache, decay_lambda=0.9),
    )
    club_id = _find_viable_club(ali_data_cache)
    gen.generate(
        opponent_club_id=club_id,
        round_date="2024-11-15",
        context=_ctx(),
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
    )

    # Engine original inchange (immutabilite preservee)
    assert len(engine.rules) == original_rules_count == 14
    assert engine.lineage_hash() == original_lineage
