"""Smoke E2E Plan 1 : load rules + classify + load pool + enrich.

ISO 29119 : integration smoke test.
Valide que les 5 composants Plan 1 s'enchaînent sans erreur.
"""

from pathlib import Path

from services.ali.history import HistoryEnricher
from services.ali.pool_loader import PlayerPoolLoader
from services.ali.verifiability import VerifiabilityClassifier
from services.ffe.rule_engine import RuleEngine

REAL_A02 = Path("config/ffe_rules/a02.json")
CLASSIF = Path("config/ffe_rules/alice_verifiability.json")


def test_plan1_smoke_pipeline_complete(ali_data_cache):
    # 1. RuleEngine
    engine = RuleEngine.from_json_file(REAL_A02)
    assert len(engine.rules) == 14

    # 2. Verifiability
    classifier = VerifiabilityClassifier.from_json_file(CLASSIF)
    public, private = classifier.partition_rules(engine.rules)
    # 10 PUBLIC + 3 PRIVATE + 1 OUT_OF_SCOPE (D-P3-09 reclassement, ADR
    # session 2026-04-28 : article 3.7 arbitrage out-of-scope car non-éligibilité
    # joueur, pas composition équipe)
    assert len(public) == 10 and len(private) == 3

    # 3. Cache (session-scoped fixture)
    assert ali_data_cache.lineage_ok()

    # 4. Pool loader (filter clubs >= 40 joueurs pour smoke ALI viable
    # — dedup TopK+MC échoue R-ALI-02 si pool trop petit)
    eligible_clubs = [c for c, df in ali_data_cache.joueurs_by_club.items() if len(df) >= 40]
    if not eligible_clubs:
        import pytest as _pytest

        _pytest.skip("aucun club avec >=40 joueurs pour smoke ALI viable")
    first_club = eligible_clubs[0]
    loader = PlayerPoolLoader(ali_data_cache)
    pool = loader.load_pool(first_club, "2024-11-15")
    assert len(pool) > 0

    # 5. Enricher
    enricher = HistoryEnricher(ali_data_cache, decay_lambda=0.9)
    enriched = enricher.enrich(pool[:5], saison=2024, current_round=5, nb_rondes_total=11)
    assert all(e.taux_presence_effectif is not None for e in enriched)
    assert all(e.played_lag1 is not None for e in enriched)
