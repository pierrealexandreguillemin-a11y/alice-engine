"""T18 Determinism tests (Plan 3 V2 §T18).

DoD : 2 runs seed=42 identiques -> memes metrics (bit-identiques).
Hash check ScenarioSet.

Sources SOTA
------------
- ISO 24029 §6.5 reproducibility : same seed -> same output, deterministic
  pipelines testable.
- ISO 29119 testability : determinisme verifiable par re-execution.
- Henderson et al. 2018 "Deep Reinforcement Learning that Matters" (AAAI)
  — la determinism via seed est une exigence minimale de reproductibilite.

4 tests :
    1. ScenarioGenerator : 2 calls memes params -> lineage_hash + scenarios
       tuple bit-identical (champ ``generated_at`` exclu du hash).
    2. ScenarioGenerator : seed different -> lineage_hash different (sanity
       counterpart : interdit qu'un seed soit silencieusement ignore).
    3. BacktestRunner : 2 runs config identique -> MatchStats listes
       bit-identical + CI bornes (recall, accuracy, jaccard, brier, ece, mae).
    4. BacktestRunner : seed different -> au moins 1 metric divergente
       (sanity counterpart bootstrap).

Document ID: ALICE-BACKTEST-DETERMINISM
Version: 1.0.0
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts.backtest.harness import BacktestHarness
from scripts.backtest.runner import BacktestRunner
from scripts.backtest.runner_types import RunnerConfig
from services.ali.types import CompetitionContext

if TYPE_CHECKING:
    from scripts.backtest.runner_types import MatchStats
    from services.ali.scenario import ScenarioSet

J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")

pytestmark = pytest.mark.skipif(
    not (J.exists() and E.exists()),
    reason="data parquets absent",
)


@pytest.fixture(scope="module")
def harness() -> BacktestHarness:
    """Module-scoped harness (setup once for all 4 determinism tests)."""
    h = BacktestHarness()
    h.setup()
    return h


def _generate_set(harness: BacktestHarness, *, seed: int) -> ScenarioSet:
    """Generate a ScenarioSet for a fixed (club, ronde, season) probe."""
    assert harness.scenario_generator is not None
    assert harness.cache is not None
    # Pick first club whose pool >= 12 (deterministic selection)
    clubs = sorted(c for c, df in harness.cache.joueurs_by_club.items() if len(df) >= 12)
    if len(clubs) < 1:
        pytest.skip("aucun club de >=12 joueurs disponible")
    opp_club = clubs[0]
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N3",
        ronde=5,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
    )
    return harness.scenario_generator.generate(
        opponent_club_id=opp_club,
        round_date="2024-09-01",
        context=ctx,
        saison=2024,
        current_round=5,
        nb_rondes_total=11,
        seed=seed,
    )


def _scenarios_signature(ss: ScenarioSet) -> tuple[object, ...]:
    """Compute a comparable signature : (lineage_hash, scenarios tuple).

    Excludes ``generated_at`` (UTC timestamp, varies between calls by design).
    """
    sigs = tuple(
        (
            tuple((a.board, a.player.nr_ffe) for a in s.lineup.assignments),
            round(s.weight, 12),
            round(s.joint_prob, 12),
            s.source,
        )
        for s in ss.scenarios
    )
    return (ss.lineage_hash, sigs)


# ---------------------------------------------------------------------------
# T18.1 : ScenarioSet bit-identical for same seed
# ---------------------------------------------------------------------------


def test_scenario_set_bit_identical_same_seed(harness: BacktestHarness) -> None:
    """2 calls memes params + seed=42 -> ScenarioSet identique (hash + tuple)."""
    ss1 = _generate_set(harness, seed=42)
    ss2 = _generate_set(harness, seed=42)
    assert ss1.lineage_hash == ss2.lineage_hash, (
        "lineage_hash divergent : seed propagation cassee OU " "generated_at fuit dans le hash"
    )
    assert _scenarios_signature(ss1) == _scenarios_signature(
        ss2
    ), "scenarios tuple divergent malgre seed=42 identique"


# ---------------------------------------------------------------------------
# T18.2 : seed different -> lineage diverge (sanity counterpart)
# ---------------------------------------------------------------------------


def test_scenario_set_diverges_different_seed(harness: BacktestHarness) -> None:
    """Sanity : seed=42 vs seed=123 produisent un lineage_hash different.

    Sans ce check, un bug "seed silencieusement ignore" passerait T18.1
    inapercu (les 2 runs seraient identiques pour la mauvaise raison).
    """
    ss1 = _generate_set(harness, seed=42)
    ss2 = _generate_set(harness, seed=123)
    assert ss1.lineage_hash != ss2.lineage_hash, (
        "seed=42 et seed=123 produisent le meme lineage_hash : "
        "le seed est ignore quelque part dans la pipeline"
    )


# ---------------------------------------------------------------------------
# T18.3 : BacktestRunner bit-identical for same seed
# ---------------------------------------------------------------------------


def _per_match_signature(stats: list[MatchStats]) -> tuple[object, ...]:
    """Stable signature for per-match metrics (rounded to 12 decimals)."""
    return tuple(
        (
            s.saison,
            s.ronde,
            s.user_team,
            s.opponent_team,
            round(s.recall_ali, 12),
            round(s.accuracy_ali, 12),
            round(s.jaccard_ali, 12),
            round(s.brier_ali, 12),
            round(s.ece_ali, 12),
            round(s.e_score_mae, 12),
            s.ali_correct,
            s.baseline_correct,
        )
        for s in stats
    )


def test_backtest_runner_bit_identical_same_seed(harness: BacktestHarness) -> None:
    """2 runs RunnerConfig(seed=42) -> MatchStats + CI bornes identiques.

    Couvre la chaine complete : harness -> ScenarioGenerator (RNG seed)
    -> ML inference (deterministe modele fige) -> metrics -> bootstrap CI
    (seed bootstrap=42).
    """
    cfg = RunnerConfig(max_matches=5, rondes=(5,), n_bootstrap=200, seed=42)
    runner1 = BacktestRunner(harness=harness, config=cfg)
    runner2 = BacktestRunner(harness=harness, config=dataclasses.replace(cfg))
    r1 = runner1.run()
    r2 = runner2.run()

    assert r1.n_matches == r2.n_matches
    assert _per_match_signature(r1.per_match) == _per_match_signature(
        r2.per_match
    ), "MatchStats divergent entre 2 runs seed=42 identiques"
    # CI bornes bit-identical (bootstrap seed propage)
    for name, c1, c2 in [
        ("recall", r1.ci_recall, r2.ci_recall),
        ("accuracy", r1.ci_accuracy, r2.ci_accuracy),
        ("jaccard", r1.ci_jaccard, r2.ci_jaccard),
        ("brier", r1.ci_brier, r2.ci_brier),
        ("ece", r1.ci_ece, r2.ci_ece),
        ("mae", r1.ci_mae, r2.ci_mae),
    ]:
        assert (
            round(c1.lower, 12) == round(c2.lower, 12)
            and round(c1.point, 12) == round(c2.point, 12)
            and round(c1.upper, 12) == round(c2.upper, 12)
        ), f"CI {name} divergent : seed bootstrap non propage"


# ---------------------------------------------------------------------------
# T18.4 : seed different -> au moins une metric diverge (sanity counterpart)
# ---------------------------------------------------------------------------


def test_backtest_runner_diverges_different_seed(harness: BacktestHarness) -> None:
    """Sanity : seed=42 vs seed=123 produisent au moins 1 metric differente.

    Sans ce check, un bug "seed runner ignore" rendrait T18.3 vacuously true.
    """
    cfg42 = RunnerConfig(max_matches=5, rondes=(5,), n_bootstrap=200, seed=42)
    cfg123 = RunnerConfig(max_matches=5, rondes=(5,), n_bootstrap=200, seed=123)
    r42 = BacktestRunner(harness=harness, config=cfg42).run()
    r123 = BacktestRunner(harness=harness, config=cfg123).run()

    sig42 = _per_match_signature(r42.per_match)
    sig123 = _per_match_signature(r123.per_match)
    # Difference attendue sur metrics ALI (recall/accuracy/jaccard/brier/ece)
    # OU sur CI bootstrap (seed propage). Au moins 1 axe.
    matches_diverge = sig42 != sig123
    ci_diverge = round(r42.ci_recall.lower, 12) != round(r123.ci_recall.lower, 12)
    assert matches_diverge or ci_diverge, (
        "seed=42 et seed=123 produisent metrics + CI strictement identiques : "
        "le seed est ignore par toute la pipeline backtest"
    )
