"""D8 perturbation runner — cache-mutation infra for stress / DRO closures.

Wires real per-match perturbation re-runs by mutating the BacktestHarness cache
opponent pool Elo column temporarily, invoking runner.run_single under the
mutated cache, then restoring on exit. Resorbs dette D-2026-05-09-d8-perturbation-runs.

Sources :
- ISO/IEC TR 24029-2:2024 §6.5 (perturbation under bounded ε)
- Goodfellow 2015 (ε-bounded), Madry 2018 (PGD bounded)
- Tran 2022 §3.4 (roster turnover under distribution shift)
- Sinha 2018 §4 (Wasserstein-2 worst-case via gradient-free sampling)
- Efron 1993 (statistical power, stratified subsampling)

Stratified subsampling : per-saison stress over a STRATIFIED 30-match sample to
respect spec §11 compute budget (~12 min stress_elo + 8 min roster + 10 min DRO
per saison wallclock). 30 matches × 5 noise × 4 saisons = 600 backtests ~1h.

Document ID: ALICE-D8-PERTURB-RUNNER
Version: 1.0.0
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.backtest.robustness import perturb_elos
from scripts.d8.dro import compute_dro_for_match
from scripts.d8.stress_elo import ElostressOutcome, run_multinoise
from scripts.d8.stress_roster import RosterStressOutcome

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from scripts.d8.dro import DROOutcome

NOISE_LEVELS_ELO: tuple[float, ...] = (0.01, 0.03, 0.05, 0.07, 0.10)
TURNOVER_RATES_ROSTER: tuple[float, ...] = (0.05, 0.10, 0.20)
DRO_EPSILONS: tuple[float, ...] = (0.05, 0.10)
DRO_N_PERTURBATIONS = 50
ELO_FLOOR = 800
ELO_CEILING = 2900
SUBSET_PER_SAISON = 30  # spec §3 + Efron 1993 statistical power


@contextmanager
def perturbed_opponent_pool(
    cache: Any,
    opp_club: str,
    perturbed_elos: list[int],
) -> Iterator[Any]:
    """Mutate cache.joueurs_by_club[opp_club] elo column temporarily.

    Restores the original DataFrame on exit (try/finally guarantees lineage
    safety even on exception). ISO 27001 §A.14.2.5 secure system principles.
    """
    if opp_club not in cache.joueurs_by_club:
        msg = f"Club {opp_club!r} absent from cache.joueurs_by_club"
        raise KeyError(msg)
    original: pd.DataFrame = cache.joueurs_by_club[opp_club].copy(deep=True)
    perturbed_df = original.copy(deep=True)
    n = min(len(perturbed_df), len(perturbed_elos))
    perturbed_df = perturbed_df.reset_index(drop=True)
    perturbed_df.loc[: n - 1, "elo"] = perturbed_elos[:n]
    cache.joueurs_by_club[opp_club] = perturbed_df
    try:
        yield cache
    finally:
        cache.joueurs_by_club[opp_club] = original


def _run_match_with_perturbation(
    runner: Any,
    match_candidate: Any,
    opp_club: str,
    perturbed_elos: list[int],
) -> float:
    """Re-run match under perturbed opponent pool; return recall_ali."""
    with perturbed_opponent_pool(runner.harness.cache, opp_club, perturbed_elos):
        result = runner.run_single(
            saison=match_candidate.saison,
            ronde=match_candidate.ronde,
            user_team=match_candidate.user_team,
            opp_team=match_candidate.opp_team,
            opp_club=opp_club,
        )
    return float(result.recall_ali) if result is not None else 0.0


def stratified_subset(
    matches: list[Any],
    n: int = SUBSET_PER_SAISON,
    seed: int = 42,
) -> list[Any]:
    """Pick stratified subset over rondes for stress sub-sampling (Efron 1993)."""
    if len(matches) <= n:
        return list(matches)
    rng = np.random.default_rng(seed)
    rondes = sorted({m.ronde for m in matches})
    per_ronde_quota = max(1, n // max(len(rondes), 1))
    selected: list[Any] = []
    for r in rondes:
        bucket = [m for m in matches if m.ronde == r]
        idx = rng.choice(len(bucket), size=min(per_ronde_quota, len(bucket)), replace=False)
        selected.extend(bucket[i] for i in idx)
    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]
    return selected


def compute_stress_elo_real(
    runner: Any,
    match_candidates: list[Any],
    baseline_recalls: list[float],
    seed_base: int = 42,
) -> list[ElostressOutcome]:
    """Real per-match stress_elo : perturb opponent pool + re-run + aggregate.

    Returns one ElostressOutcome per noise level (drop = baseline_mean - perturbed_mean).
    """
    perturbed_recalls_by_noise: dict[float, list[float]] = {n: [] for n in NOISE_LEVELS_ELO}
    for cand_idx, cand in enumerate(match_candidates):
        opp_club = cand.opp_club
        opp_pool_df = runner.harness.cache.joueurs_by_club.get(opp_club)
        if opp_pool_df is None or len(opp_pool_df) == 0:
            continue
        opp_elos = [int(e or 1500) for e in opp_pool_df["elo"].tolist()]
        for noise_pct in NOISE_LEVELS_ELO:
            perturbed = perturb_elos(opp_elos, noise_pct, seed=seed_base + cand_idx)
            recall = _run_match_with_perturbation(runner, cand, opp_club, perturbed)
            perturbed_recalls_by_noise[noise_pct].append(recall)
    return run_multinoise(baseline_recalls, perturbed_recalls_by_noise)


def compute_stress_roster_real(
    runner: Any,
    match_candidates: list[Any],
    baseline_recalls: list[float],
    seed_base: int = 42,
) -> list[RosterStressOutcome]:
    """Real per-match stress_roster : drop fraction of opp pool + re-run."""
    outcomes: list[RosterStressOutcome] = []
    for turnover_pct in TURNOVER_RATES_ROSTER:
        perturbed_recalls = _run_roster_turnover_subset(
            runner, match_candidates, turnover_pct, seed_base
        )
        baseline_mean = float(np.mean(baseline_recalls)) if baseline_recalls else 0.0
        perturbed_mean = float(np.mean(perturbed_recalls)) if perturbed_recalls else 0.0
        outcomes.append(
            RosterStressOutcome(
                turnover_pct=turnover_pct,
                baseline_recall=baseline_mean,
                perturbed_recall_mean=perturbed_mean,
                recall_drop=max(0.0, baseline_mean - perturbed_mean),
            )
        )
    return outcomes


def _run_roster_turnover_subset(
    runner: Any,
    match_candidates: list[Any],
    turnover_pct: float,
    seed_base: int,
) -> list[float]:
    """Drop turnover_pct of opp pool per match + run + collect recalls."""
    perturbed_recalls: list[float] = []
    for cand_idx, cand in enumerate(match_candidates):
        opp_club = cand.opp_club
        opp_pool_df = runner.harness.cache.joueurs_by_club.get(opp_club)
        if opp_pool_df is None or len(opp_pool_df) == 0:
            continue
        n_drop = int(len(opp_pool_df) * turnover_pct)
        if len(opp_pool_df) - n_drop < 8:  # need ≥ team_size after drop  # noqa: PLR2004
            continue
        rng = np.random.default_rng(seed_base + cand_idx)
        drop_idx = set(rng.choice(len(opp_pool_df), size=n_drop, replace=False).tolist())
        keep_mask = [i not in drop_idx for i in range(len(opp_pool_df))]
        kept_df = opp_pool_df.iloc[keep_mask].reset_index(drop=True)
        original = runner.harness.cache.joueurs_by_club[opp_club].copy(deep=True)
        runner.harness.cache.joueurs_by_club[opp_club] = kept_df
        try:
            result = runner.run_single(
                saison=cand.saison,
                ronde=cand.ronde,
                user_team=cand.user_team,
                opp_team=cand.opp_team,
                opp_club=opp_club,
            )
            perturbed_recalls.append(float(result.recall_ali) if result else 0.0)
        finally:
            runner.harness.cache.joueurs_by_club[opp_club] = original
    return perturbed_recalls


def compute_dro_real(
    runner: Any,
    match_candidates: list[Any],
    seed_base: int = 42,
) -> dict[float, DROOutcome]:
    """Real DRO : per-match Wasserstein worst-case aggregated over subset.

    For each ε, runs `n_perturbations` shift+scale draws per match within ε-ball,
    finds the minimum recall observed across all match × perturbation combinations.
    """
    aggregated: dict[float, DROOutcome] = {}
    for eps in DRO_EPSILONS:
        worst_per_match: list[Any] = []
        for cand_idx, cand in enumerate(match_candidates):
            opp_club = cand.opp_club
            opp_pool_df = runner.harness.cache.joueurs_by_club.get(opp_club)
            if opp_pool_df is None or len(opp_pool_df) == 0:
                continue
            opp_elos = [int(e or 1500) for e in opp_pool_df["elo"].tolist()]

            def _backtest(perturbed: list[int], _cand: Any = cand, _club: str = opp_club) -> float:
                return _run_match_with_perturbation(runner, _cand, _club, perturbed)

            single = compute_dro_for_match(
                opp_elos,
                _backtest,
                epsilons=(eps,),
                n_perturbations=DRO_N_PERTURBATIONS,
                seed_base=seed_base + cand_idx,
            )
            worst_per_match.append(single[eps])
        if not worst_per_match:
            continue
        global_worst = min(o.recall_worst_case for o in worst_per_match)
        from scripts.d8.dro import DROOutcome as _DROOutcome

        aggregated[eps] = _DROOutcome(
            epsilon=eps,
            n_perturbations=DRO_N_PERTURBATIONS * len(worst_per_match),
            recall_worst_case=float(global_worst),
            worst_perturbation_finding=f"min over {len(worst_per_match)} matches",
        )
    return aggregated
