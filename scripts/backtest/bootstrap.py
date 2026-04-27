"""Bootstrap confidence intervals (BCa) for backtest metrics.

Plan 3 V2 T8. ISO 25059 (gates statistiquement valides), 29119.

Source : Efron 1987 "Better bootstrap confidence intervals".
BCa (Bias-Corrected and accelerated) est le standard SOTA pour IC non-paramétriques
sur metrics arbitraires (recall, Brier, Jaccard, etc.).

Usage
-----
>>> values = [0.85, 0.90, 0.87, 0.92, 0.88, ...]  # per-match metric
>>> ci = bootstrap_ci(values, confidence=0.95, n_resamples=1000, method="BCa")
>>> ci.lower, ci.point, ci.upper
(0.86, 0.88, 0.90)

ISO 25059 : un gate "T13 ≥ 0.90" PASS si ci.lower ≥ 0.90 (rigueur statistique),
pas si ci.point ≥ 0.90 (point estimate trompeur).

Document ID: ALICE-BACKTEST-BOOTSTRAP
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BootstrapCI:
    """Confidence interval with point estimate + metadata.

    @param lower: lower bound of CI
    @param point: point estimate (mean of original sample)
    @param upper: upper bound of CI
    @param confidence: confidence level (e.g. 0.95)
    @param n_resamples: number of bootstrap resamples used
    @param method: "BCa" (bias-corrected accelerated) or "percentile"
    @param n_samples: size of original sample
    """

    lower: float
    point: float
    upper: float
    confidence: float
    n_resamples: int
    method: str
    n_samples: int

    def passes_gate(self, threshold: float, *, direction: Literal["ge", "le"]) -> bool:
        """Check gate with statistical rigor.

        direction="ge" (gate ≥ threshold) : CI lower bound must ≥ threshold
        direction="le" (gate ≤ threshold) : CI upper bound must ≤ threshold
        """
        if direction == "ge":
            return self.lower >= threshold
        return self.upper <= threshold


def bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    method: Literal["BCa", "percentile"] = "BCa",
    seed: int | None = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for mean of values.

    @param values: per-match metric values (≥ 20 required for stability)
    @param confidence: confidence level in (0, 1)
    @param n_resamples: number of bootstrap resamples
    @param method: "BCa" (Efron 1987) ou "percentile"
    @param seed: RNG seed pour reproductibilité (None = non-déterministe)

    @raises ValueError: values empty or confidence out of range
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        msg = "bootstrap_ci: values empty"
        raise ValueError(msg)
    if not 0 < confidence < 1:
        msg = f"bootstrap_ci: confidence must be in (0,1), got {confidence}"
        raise ValueError(msg)

    point = float(np.mean(arr))

    # IC dégénéré : n<2 OU variance nulle (toutes valeurs identiques).
    # Sans ce guard, scipy.stats.bootstrap BCa retourne NaN sur var=0
    # (DegenerateDataWarning) — pollue downstream metrics.
    # Detected by Hypothesis property test (T19, falsifying values=[0.0, 0.0]).
    if n < 2 or float(np.var(arr)) == 0.0:
        return BootstrapCI(
            lower=point,
            point=point,
            upper=point,
            confidence=confidence,
            n_resamples=0,
            method=method,
            n_samples=n,
        )

    rng = np.random.default_rng(seed)
    # scipy.stats.bootstrap requires samples as tuple
    res = stats.bootstrap(
        data=(arr,),
        statistic=np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method=method,
        random_state=rng,
        vectorized=False,
    )
    return BootstrapCI(
        lower=float(res.confidence_interval.low),
        point=point,
        upper=float(res.confidence_interval.high),
        confidence=confidence,
        n_resamples=n_resamples,
        method=method,
        n_samples=n,
    )
