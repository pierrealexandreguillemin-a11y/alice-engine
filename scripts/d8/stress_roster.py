"""D8 stress roster - turnover perturbation (Tran 2022, ISO 24029 §6.5).

Drop-out aleatoire d'une fraction du pool joueurs pour mesurer la
robustesse du backtest face a l'absence de joueurs (turnover hebdo
realiste 5-20%).

Sources SOTA :
- Tran et al. 2022 "On Distribution Shift in ML for Sports" (drop-out
  joueur 10-20% comme stress test compositions equipes).
- ISO/IEC TR 24029:2021 §6.5 distribution shift robustness.
- Goodfellow et al. 2015 (input dropout adversarial baseline).

Document ID: ALICE-D8-STRESS-ROSTER
Version: 1.0.0
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")

TURNOVER_RATES: list[float] = [0.05, 0.10, 0.20]


@dataclass(frozen=True)
class RosterStressOutcome:
    """Aggregated outcome of a roster stress run for a single turnover.

    @attr turnover_pct: fraction dropped (0.10 = 10%).
    @attr baseline_recall: mean recall across matches without drop.
    @attr perturbed_recall_mean: mean recall under turnover (None entries skipped).
    @attr recall_drop: max(0, baseline - perturbed) absolute drop.
    """

    turnover_pct: float
    baseline_recall: float
    perturbed_recall_mean: float
    recall_drop: float


def drop_random_players(
    pool: list[T],
    turnover_pct: float,
    seed: int = 42,
) -> list[T]:
    """Drop turnover_pct fraction (floor) of pool uniformly at random.

    @param pool: input list (any T).
    @param turnover_pct: fraction in [0, 1). 0 = noop, returns pool copy.
    @param seed: rng seed (ISO 29119 reproducibility).
    @raises ValueError: turnover_pct < 0 or >= 1 (ISO 27034 input guard).
    @returns new list with kept items in original order.
    """
    if not 0 <= turnover_pct < 1:
        msg = f"turnover_pct must be in [0, 1), got {turnover_pct}"
        raise ValueError(msg)
    if not pool:
        return []
    n = len(pool)
    n_drop = int(n * turnover_pct)
    if n_drop == 0:
        return list(pool)
    rng = np.random.default_rng(seed)
    drop_idx = set(rng.choice(n, size=n_drop, replace=False).tolist())
    return [item for i, item in enumerate(pool) if i not in drop_idx]


def compute_stress_roster_for_match(
    opp_pool: list[T],
    backtest_run_fn: Callable[[list[T]], float],
    min_pool_size: int,
    turnover_rates: list[float] = TURNOVER_RATES,
    seed: int = 42,
) -> dict[float, float | None]:
    """Per-match : run backtest_fn on dropped roster for each turnover rate.

    @param opp_pool: opponent pool roster (players, ids, dicts...).
    @param backtest_run_fn: callable(perturbed_pool) -> recall.
    @param min_pool_size: skip turnover if remaining roster < min_pool_size.
    @param turnover_rates: list of turnover_pct in [0, 1).
    @param seed: rng seed propagated to drop_random_players.
    @returns {turnover_pct: recall or None if pool too small}.
    @raises ValueError: any turnover out of [0, 1) (delegated).
    """
    result: dict[float, float | None] = {}
    for rate in turnover_rates:
        perturbed = drop_random_players(opp_pool, rate, seed=seed)
        if len(perturbed) < min_pool_size:
            result[rate] = None
        else:
            result[rate] = backtest_run_fn(perturbed)
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_roster_outcomes(
    matches_baseline_recalls: list[float],
    perturbed_recalls_per_turnover: dict[float, list[float]],
) -> list[RosterStressOutcome]:
    """Aggregate baseline + perturbed recalls per turnover rate.

    @param matches_baseline_recalls: per-match baseline recall list.
    @param perturbed_recalls_per_turnover: {turnover_pct: per-match recall list}.
    @returns sorted by turnover_pct ascending. recall_drop clamped >= 0.
    """
    base = _mean(matches_baseline_recalls)
    return [
        RosterStressOutcome(
            turnover_pct=rate,
            baseline_recall=base,
            perturbed_recall_mean=_mean(perturbed),
            recall_drop=max(0.0, base - _mean(perturbed)),
        )
        for rate, perturbed in sorted(perturbed_recalls_per_turnover.items())
    ]
