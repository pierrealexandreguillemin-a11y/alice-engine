"""Pure scenario-set helpers for ScenarioGenerator (ISO 5055 SRP split).

Merge/dedup/pad/renormalize + history normalization + scenario signature,
extracted from generator.py to keep that module under the 300-line limit.
No I/O, no generator state: free functions over Scenario lists.

Document ID: ALICE-ALI-GENERATOR-HELPERS
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from services.ali.scenario import Scenario

if TYPE_CHECKING:
    import numpy as np

    from services.ali.monte_carlo import MonteCarloSampler
    from services.ali.types import CompetitionContext, PlayerCandidate

_EXPECTED_TOTAL = 20


def signature(s: Scenario) -> tuple[tuple[str, int], ...]:
    """Canonical signature (nr_ffe x board) for dedup."""
    return tuple((a.player.nr_ffe, a.board) for a in s.lineup.assignments)


def dedup_distinct(scenarios: list[Scenario]) -> list[Scenario]:
    """Garde scenarios distincts par signature (player nr_ffe x board)."""
    seen: set[tuple[tuple[str, int], ...]] = set()
    out: list[Scenario] = []
    for s in scenarios:
        sig = signature(s)
        if sig not in seen:
            seen.add(sig)
            out.append(s)
    return out


def pad_with_mc(
    mc: MonteCarloSampler,
    pool: list[PlayerCandidate],
    context: CompetitionContext,
    n_extra: int,
    rng: np.random.Generator,
) -> list[Scenario]:
    """Best-effort pad with extra MC samples if dedup left us short.

    Double the request to increase diversity (LHS + antithetic produces
    many duplicates when the dominant lineup has very high joint_prob).
    """
    if n_extra <= 0:
        return []
    n_pairs = max(2, n_extra)  # oversample to improve diversity
    extra = mc.sample(pool, context, n_pairs=n_pairs, rng=rng)
    return list(extra)


def merge_and_pad(  # noqa: PLR0913
    topk: list[Scenario],
    mc: list[Scenario],
    mc_sampler: MonteCarloSampler,
    pool: list[PlayerCandidate],
    context: CompetitionContext,
    rng: np.random.Generator,
) -> list[Scenario]:
    """Dedup + pad to _EXPECTED_TOTAL (retries up to 5 rounds best-effort)."""
    merged = dedup_distinct(topk + mc)
    existing_sigs = {signature(s) for s in merged}
    max_rounds = 5
    for _ in range(max_rounds):
        missing = _EXPECTED_TOTAL - len(merged)
        if missing <= 0:
            break
        extras = pad_with_mc(mc_sampler, pool, context, missing, rng)
        progressed = False
        for e in extras:
            sig = signature(e)
            if sig not in existing_sigs:
                merged.append(e)
                existing_sigs.add(sig)
                progressed = True
            if len(merged) >= _EXPECTED_TOTAL:
                break
        if not progressed:
            break  # pool trop petit pour generer 20 lineups distincts
    return merged[:_EXPECTED_TOTAL]


def renormalize(scenarios: list[Scenario]) -> list[Scenario]:
    """Normalize weights so sum = 1.0 across the final set."""
    total = sum(s.joint_prob for s in scenarios)
    if total <= 0:
        uniform = 1.0 / len(scenarios) if scenarios else 0.0
        return [
            Scenario(
                lineup=s.lineup,
                joint_prob=s.joint_prob,
                weight=uniform,
                source=s.source,
            )
            for s in scenarios
        ]
    return [
        Scenario(
            lineup=s.lineup,
            joint_prob=s.joint_prob,
            weight=s.joint_prob / total,
            source=s.source,
        )
        for s in scenarios
    ]


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Echiquiers raw -> joueur_nom long format (reuse pattern history.py)."""
    parts: list[pd.DataFrame] = []
    for col in ("blanc_nom", "noir_nom"):
        if col in df.columns:
            sub = df[[col, "saison", "ronde", "echiquier"]].copy()
            sub = sub.rename(columns={col: "joueur_nom"})
            parts.append(sub)
    if not parts:
        return pd.DataFrame(
            columns=["joueur_nom", "saison", "ronde", "echiquier"],
        )
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=["joueur_nom", "saison", "ronde"])
