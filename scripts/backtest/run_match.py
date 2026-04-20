"""Direct wrapper run_backtest_match : generator + aggregation, no FastAPI overhead.

Plan 3 P3-Task 1. Réutilise ScenarioGenerator + StackingInferenceService Plan 2.
strict=True → fail-fast ISO 42001.

Document ID: ALICE-BACKTEST-RUN-MATCH
Version: 1.0.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from services.ali.aggregation import (
    AggregatedBoard,
    ScenarioAggregationCtx,
    aggregate_from_scenarios,
)
from services.ali.types import CompetitionContext

if TYPE_CHECKING:
    from services.ali.generator import ScenarioGenerator
    from services.ali.scenario import ScenarioSet
    from services.inference import StackingInferenceService


@dataclass
class BacktestMatchResult:
    """Resultat d'un match backtest (ALI generation + ML inference + aggregation)."""

    saison: int
    ronde: int
    user_club: str
    opponent_club: str
    division: str
    scenario_set: ScenarioSet
    aggregated_boards: list[AggregatedBoard]
    elapsed_ms: float
    lineage_hash: str


def run_backtest_match(  # noqa: PLR0913
    *,
    user_club_id: str,
    opponent_club_id: str,
    saison: int,
    ronde: int,
    nb_rondes_total: int,
    division: str,
    team_size: int,
    user_lineup: list[dict[str, Any]],
    scenario_generator: ScenarioGenerator,
    inference: StackingInferenceService,
    feature_store: Any,
    seed: int = 42,
    strict: bool = True,
) -> BacktestMatchResult:
    """Run un match backtest. Strict=True (ISO 42001 fail-fast).

    Pas de TestClient overhead. Réutilise helpers Plan 2.
    """
    start = time.perf_counter()

    context = CompetitionContext(
        competition_code="A02",
        niveau=division,
        ronde=ronde,
        team_size=team_size,
        noyau_min=50,
        max_mutes=3,
        elo_max=2400 if division == "N4" else None,
    )

    scenario_set = scenario_generator.generate(
        opponent_club_id=opponent_club_id,
        round_date=f"{saison}-09-01",
        context=context,
        saison=saison,
        current_round=ronde,
        nb_rondes_total=nb_rondes_total,
        seed=seed,
    )

    agg_ctx = ScenarioAggregationCtx(
        scenario_set=scenario_set,
        user_lineup=user_lineup,
        team_size=team_size,
        ronde=ronde,
        division=division,
        strict=strict,
    )
    boards = aggregate_from_scenarios(inference, feature_store, agg_ctx)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return BacktestMatchResult(
        saison=saison,
        ronde=ronde,
        user_club=user_club_id,
        opponent_club=opponent_club_id,
        division=division,
        scenario_set=scenario_set,
        aggregated_boards=boards,
        elapsed_ms=elapsed_ms,
        lineage_hash=scenario_set.lineage_hash,
    )
