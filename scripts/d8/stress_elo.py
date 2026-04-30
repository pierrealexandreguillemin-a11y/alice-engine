"""D8 stress Elo - multi-noise wrapper around perturb_elos (ISO 24029 §6.5).

Etend scripts/backtest/robustness.perturb_elos en :
- Iterant sur 5 niveaux de bruit gaussien standards [1%, 3%, 5%, 7%, 10%]
- Aggregeant baseline_recall + perturbed_recall_mean -> recall_drop par noise

Sources SOTA :
- ISO/IEC TR 24029:2021 §6.5 Robustness perturbations multi-niveaux
- Goodfellow et al. 2015 "Explaining and Harnessing Adversarial Examples" ICLR
- Madry et al. 2018 "Towards Deep Learning Models Resistant to Adversarial
  Attacks" ICLR (PGD multi-eps stress baseline)

Document ID: ALICE-D8-STRESS-ELO
Version: 1.0.0
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from scripts.backtest.robustness import perturb_elos

NOISE_LEVELS: list[float] = [0.01, 0.03, 0.05, 0.07, 0.10]


@dataclass(frozen=True)
class ElostressOutcome:
    """Aggregated outcome of an Elo stress run for a single noise level.

    @attr noise_pct: stddev fraction (0.05 = 5% of mean Elo).
    @attr baseline_recall: mean recall across matches without perturbation.
    @attr perturbed_recall_mean: mean recall under noise.
    @attr recall_drop: max(0, baseline - perturbed) absolute drop.
    """

    noise_pct: float
    baseline_recall: float
    perturbed_recall_mean: float
    recall_drop: float


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_multinoise(
    matches_baseline_recalls: list[float],
    perturbed_recalls_per_noise: dict[float, list[float]],
) -> list[ElostressOutcome]:
    """Aggregate baseline + perturbed recalls per noise into ElostressOutcome list.

    @param matches_baseline_recalls: per-match baseline recall list.
    @param perturbed_recalls_per_noise: {noise_pct: per-match recall list}.
    @returns sorted by noise_pct ascending. recall_drop clamped >= 0.
    """
    base = _mean(matches_baseline_recalls)
    return [
        ElostressOutcome(
            noise_pct=noise,
            baseline_recall=base,
            perturbed_recall_mean=_mean(perturbed),
            recall_drop=max(0.0, base - _mean(perturbed)),
        )
        for noise, perturbed in sorted(perturbed_recalls_per_noise.items())
    ]


def compute_stress_elo_for_match(
    opp_pool_elos: list[int],
    backtest_run_fn: Callable[[list[int]], float],
    noise_levels: list[float] = NOISE_LEVELS,
    seed: int = 42,
) -> dict[float, float]:
    """Per-match : run backtest_fn on perturbed Elos for each noise level.

    @param opp_pool_elos: opponent pool Elos.
    @param backtest_run_fn: callable(perturbed_elos) -> recall.
    @param noise_levels: list of noise_pct (each in [0, 1)).
    @param seed: rng seed propagated to perturb_elos for reproducibility.
    @returns {noise_pct: perturbed_recall}. Same seed -> identical perturbations.
    @raises ValueError: any noise_level out of [0, 1) (delegated to perturb_elos).
    """
    return {
        noise: backtest_run_fn(perturb_elos(opp_pool_elos, noise, seed=seed))
        for noise in noise_levels
    }
