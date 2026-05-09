"""D8 Distributionally Robust Optimization — Wasserstein-2 worst-case.

Source SOTA :
- Sinha, Namkoong, Duchi 2018 "Certifying Some Distributional Robustness with
  Principled Adversarial Training" ICLR §4 (Wasserstein-2 ε-ball)
- Duchi & Namkoong 2021 "Learning Models with Uniform Performance via
  Distributionally Robust Optimization" JMLR §6
- Goodfellow 2015 "Explaining and Harnessing Adversarial Examples" ε-bounded
- ISO/IEC TR 24029-2:2024 §6.5 robustness under distribution shift
- ISO/IEC 27034:2011 input validation requirement

Approach :
    Approximate the Wasserstein-2 ε-ball worst-case by sampling
    `n_perturbations` independent shift+scale perturbations of the input Elo
    distribution and reporting the minimal recall observed. Gradient-free,
    suitable for the non-differentiable ALI Monte-Carlo sampler.

Marginal guarantee (Sinha 2018 Eq. 4) :
    For ε small, the worst-case loss in the W2 ε-ball is approximated by
    the sup over translations + scalings of magnitude ≤ ε.

Document ID: ALICE-D8-DRO
Version: 1.0.0
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

ELO_FLOOR = 800
ELO_CEILING = 2900
DEFAULT_N_PERTURBATIONS = 50
DEFAULT_EPSILONS: tuple[float, ...] = (0.05, 0.10)


@dataclass(frozen=True)
class DROOutcome:
    """Worst-case recall under one Wasserstein-2 ε-ball (Sinha 2018 §4).

    @attr epsilon: ε bound on Wasserstein-2 perturbation magnitude
    @attr n_perturbations: number of independent samples drawn from the ε-ball
    @attr recall_worst_case: minimum recall observed over n_perturbations
    @attr worst_perturbation_finding: shift + scale producing the worst recall
    """

    epsilon: float
    n_perturbations: int
    recall_worst_case: float
    worst_perturbation_finding: str


def _validate_epsilon(epsilon: float) -> None:
    """ISO 27034 input validation on ε."""
    if not isinstance(epsilon, int | float) or math.isnan(epsilon) or math.isinf(epsilon):
        msg = f"epsilon must be a finite number, got {epsilon!r}"
        raise ValueError(msg)
    if not 0.0 <= epsilon < 1.0:
        msg = f"epsilon must be in [0, 1), got {epsilon}"
        raise ValueError(msg)


def _validate_n_perturbations(n_perturbations: int) -> None:
    """ISO 27034 input validation on n_perturbations."""
    if not isinstance(n_perturbations, int) or n_perturbations < 1:
        msg = f"n_perturbations must be a positive int, got {n_perturbations!r}"
        raise ValueError(msg)


def perturb_wasserstein(
    elos: list[int],
    epsilon: float,
    seed: int,
) -> tuple[list[int], dict[str, float]]:
    """Apply one shift+scale perturbation within the Wasserstein-2 ε-ball.

    Sinha 2018 §4 approximates the W2 ε-ball worst-case by translation +
    scaling. The Elo array is centered, scaled by a uniform factor in
    [1-ε, 1+ε], translated by a uniform shift in [-ε·μ, +ε·μ], then
    clipped to FFE-plausible bounds [800, 2900].

    Returns :
        (perturbed_elos, params) where params = {shift, scale} for diagnostics.
    """
    _validate_epsilon(epsilon)
    if not elos:
        return [], {"shift": 0.0, "scale": 1.0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(elos, dtype=float)
    if not np.all(np.isfinite(arr)):
        msg = "elos must contain only finite values"
        raise ValueError(msg)
    mean_elo = float(arr.mean())
    shift = float(rng.uniform(-epsilon * mean_elo, epsilon * mean_elo))
    scale = float(rng.uniform(1.0 - epsilon, 1.0 + epsilon))
    perturbed = ((arr - mean_elo) * scale + mean_elo + shift).clip(ELO_FLOOR, ELO_CEILING)
    return perturbed.astype(int).tolist(), {"shift": shift, "scale": scale}


def wasserstein_worst_case(
    elos: list[int],
    backtest_run_fn: Callable[[list[int]], float],
    epsilon: float,
    n_perturbations: int = DEFAULT_N_PERTURBATIONS,
    seed_base: int = 42,
) -> DROOutcome:
    """Find the worst-case (minimum) recall over a sample of W2 ε-perturbations.

    Sinha 2018 §4 grid-free approximation : draws ``n_perturbations`` independent
    samples of (shift, scale) from the ε-ball and evaluates the user-supplied
    ``backtest_run_fn`` on each perturbed Elo vector. Returns the minimum
    recall observed and the perturbation responsible for it.

    Args :
        elos: opponent pool Elo vector to perturb
        backtest_run_fn: takes a perturbed Elo list, returns a recall in [0, 1]
        epsilon: ε bound (0 ≤ ε < 1) on perturbation magnitude
        n_perturbations: number of samples drawn from the ε-ball (≥1)
        seed_base: PRNG seed offset for reproducibility

    Returns :
        DROOutcome with the worst recall and the (shift, scale) that produced it.
    """
    _validate_epsilon(epsilon)
    _validate_n_perturbations(n_perturbations)

    if not elos:
        return DROOutcome(
            epsilon=epsilon,
            n_perturbations=0,
            recall_worst_case=0.0,
            worst_perturbation_finding="empty pool",
        )

    worst_recall = math.inf
    worst_params: dict[str, float] = {"shift": 0.0, "scale": 1.0}
    for k in range(n_perturbations):
        perturbed, params = perturb_wasserstein(elos, epsilon, seed_base + k)
        r = float(backtest_run_fn(perturbed))
        if not math.isfinite(r):
            msg = f"backtest_run_fn returned non-finite recall {r!r} at k={k}"
            raise RuntimeError(msg)
        if not 0.0 <= r <= 1.0:
            msg = f"backtest_run_fn returned recall outside [0, 1]: {r} at k={k}"
            raise RuntimeError(msg)
        if r < worst_recall:
            worst_recall = r
            worst_params = params

    finding = f"shift={worst_params['shift']:+.1f}+scale={worst_params['scale']:.3f}"
    return DROOutcome(
        epsilon=epsilon,
        n_perturbations=n_perturbations,
        recall_worst_case=float(worst_recall),
        worst_perturbation_finding=finding,
    )


def compute_dro_for_match(
    opp_elos: list[int],
    backtest_run_fn: Callable[[list[int]], float],
    epsilons: Iterable[float] = DEFAULT_EPSILONS,
    n_perturbations: int = DEFAULT_N_PERTURBATIONS,
    seed_base: int = 42,
) -> dict[float, DROOutcome]:
    """Run wasserstein_worst_case for each epsilon and return a mapping.

    Each epsilon receives a deterministic seed offset (``seed_base + int(eps*1000)``)
    so results are reproducible per (match, epsilon).
    """
    epsilons_list = list(epsilons)
    if not epsilons_list:
        msg = "epsilons must be non-empty"
        raise ValueError(msg)
    return {
        eps: wasserstein_worst_case(
            opp_elos,
            backtest_run_fn,
            eps,
            n_perturbations,
            seed_base + int(eps * 1000),
        )
        for eps in epsilons_list
    }
