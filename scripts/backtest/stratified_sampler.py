"""T15 Stratified sampling protocol (Plan 3 V2 Phase 3, ISO 24027).

Balance sampling across strata pour eviter gate-passing masque par
repartition inegale entre groupes (ex. 80% N3 matches, 5% N1 -> N1
subset trop petit pour detecter disparite performance).

Protocole (Barocas/Hardt/Narayanan 2019 §3) :
- Definir strata (ronde, club_size quartile, etc.)
- Sample quota fixe par stratum (min_per_stratum) ou proportion
- Fairness gates P3G12 applicables per-stratum car sample sufficient

Document ID: ALICE-BACKTEST-STRATIFIED
Version: 1.0.0
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class StratifiedSamplerConfig:
    """Configuration for stratified sampling."""

    min_per_stratum: int = 5
    max_per_stratum: int = 20
    seed: int = 42


def stratified_sample(
    items: list[T],
    strata_fn: object,
    config: StratifiedSamplerConfig,
) -> list[T]:
    """Sample items with balanced representation per stratum.

    Algorithme :
    1. Group items by strata_fn(item) -> str
    2. Per-stratum : sample min(len(stratum), max_per_stratum)
    3. Drop strata with count < min_per_stratum (insufficient for stats)
    4. Shuffle deterministiquement (seed)
    5. Return flat list

    @param items: list to sample from.
    @param strata_fn: callable item -> str (stratum key).
    @param config: sampler config.

    @returns flat list balanced across strata, shuffled deterministically.
    """
    rng = random.Random(config.seed)  # noqa: S311 non-cryptographic reproducibility
    groups: dict[str, list[T]] = {}
    for item in items:
        k = str(strata_fn(item))  # type: ignore[operator]
        groups.setdefault(k, []).append(item)

    out: list[T] = []
    for stratum_items in groups.values():
        if len(stratum_items) < config.min_per_stratum:
            continue
        shuffled = stratum_items.copy()
        rng.shuffle(shuffled)
        out.extend(shuffled[: config.max_per_stratum])

    rng.shuffle(out)
    return out


def strata_coverage(
    items: list[T],
    strata_fn: object,
    min_per_stratum: int = 5,
) -> dict[str, int]:
    """Audit helper : count items per stratum (before sampling).

    @returns {stratum_key: count} pour diagnostic fairness protocol.
    """
    groups: dict[str, int] = {}
    for item in items:
        k = str(strata_fn(item))  # type: ignore[operator]
        groups[k] = groups.get(k, 0) + 1
    return groups


def sufficient_strata(
    coverage: dict[str, int],
    min_per_stratum: int = 5,
) -> dict[str, bool]:
    """Return {stratum: has >= min_per_stratum}."""
    return {k: v >= min_per_stratum for k, v in coverage.items()}
