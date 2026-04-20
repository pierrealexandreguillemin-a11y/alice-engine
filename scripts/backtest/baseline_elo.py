"""Baseline Elo legitimate : 1 scenario tri Elo pour comparaison T22.

Plan 3 V2 T10. ISO 25059 (baseline comparison), 29119.

Baseline method
---------------
Pour chaque match à backtest, construire UN scenario unique par tri Elo
descendant des joueurs observés disponibles au pool. Weight = 1.0.
Représente la stratégie naïve "composer les meilleurs Elo".

ALI SOTA doit battre ce baseline (T22 McNemar p<0.05) sur ≥ 3 metrics T13-T17.

Helpers exposés
---------------
- `baseline_elo_scenario_set(pool, team_size)` : ScenarioSet 1 scenario
- `baseline_elo_brier(observed, pool, team_size)` : Brier score du baseline,
  utilisé comme reference pour Brier skill score (T6).

Document ID: ALICE-BACKTEST-BASELINE-ELO
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.ali.scenario import BoardAssignment, Lineup, Scenario, ScenarioSet

if TYPE_CHECKING:
    from scripts.backtest.ground_truth import ObservedLineup
    from services.ali.types import PlayerCandidate


def baseline_elo_scenario_set(
    pool: list[PlayerCandidate],
    team_size: int,
    opponent_club_id: str = "baseline",
    round_date: str = "2024-01-01",
    generated_at: str = "2024-01-01T00:00:00Z",
    lineage_hash: str = "baseline_elo" + "0" * 52,
) -> ScenarioSet:
    """Build a 1-scenario ScenarioSet : top team_size joueurs tri Elo desc.

    @param pool: liste des joueurs disponibles (typiquement observed pool)
    @param team_size: taille équipe (K échiquiers)

    @raises ValueError: pool plus petit que team_size
    """
    if len(pool) < team_size:
        msg = f"baseline: pool too small {len(pool)} < team_size {team_size}"
        raise ValueError(msg)
    sorted_pool = sorted(pool, key=lambda p: -p.elo)[:team_size]
    assignments = tuple(
        BoardAssignment(
            board=i + 1,
            player=p,
            p_assignment=1.0,
        )
        for i, p in enumerate(sorted_pool)
    )
    lineup = Lineup(team_size=team_size, assignments=assignments)
    scenario = Scenario(
        lineup=lineup,
        joint_prob=1.0,
        weight=1.0,
        source="monte_carlo",  # baseline uses valid source literal
    )
    return ScenarioSet(
        scenarios=(scenario,),
        opponent_club_id=opponent_club_id,
        round_date=round_date,
        generated_at=generated_at,
        lineage_hash=lineage_hash,
    )


def baseline_elo_brier(
    observed: ObservedLineup,
    pool: list[PlayerCandidate],
    team_size: int,
) -> float:
    """Compute Brier score of baseline Elo scenario (helper for T6 skill score).

    Concern #4 résolu : au lieu de passer baseline_brier param numeric arbitraire,
    on calcule dynamiquement depuis pool + observed.

    @param observed: lineup réel du club adverse
    @param pool: liste joueurs disponibles
    @param team_size: taille équipe

    @returns Brier score baseline ∈ [0, 1]
    """
    # Avoid circular import
    from scripts.backtest.metrics import brier_presence  # noqa: PLC0415

    ss = baseline_elo_scenario_set(pool, team_size)
    return brier_presence(observed, ss)
