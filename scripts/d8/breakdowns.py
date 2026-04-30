"""D8 breakdowns — 7 dimensions stratification (ISO 24027 §6).

Source SOTA :
- Mehrabi et al. 2021 "Survey on Bias and Fairness in ML" ACM CSUR
- Holstein et al. 2024 "Industry fairness assessments" FAccT (breadth>depth)
- ISO/IEC TR 24027:2021 §6.4 protected vs service-level

Categorie age buckets : scripts/parse_dataset/constants.py CATEGORIES_AGE.

Document ID: ALICE-D8-BREAKDOWNS
Version: 1.0.0
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scripts.backtest.runner_types import MatchStats
from scripts.d8.types import D8GroupBreakdown, D8GroupStats

# ---- Bucket helpers ----

_CATEGORIE_AGE_BUCKETS: dict[str, str] = {
    # U12
    "PpoM": "U12",
    "PpoF": "U12",
    "PouM": "U12",
    "PouF": "U12",
    "PupM": "U12",
    "PupF": "U12",
    # U18
    "BenM": "U18",
    "BenF": "U18",
    "MinM": "U18",
    "MinF": "U18",
    "CadM": "U18",
    "CadF": "U18",
    # U20
    "JunM": "U20",
    "JunF": "U20",
    # Sen
    "SenM": "Sen",
    "SenF": "Sen",
    # S50+
    "SepM": "S50+",
    "SepF": "S50+",
    "VetM": "S50+",
    "VetF": "S50+",
}


def bucket_categorie_age(categorie_ffe: str) -> str:
    """Map 20 FFE age categories to 5 buckets (U12/U18/U20/Sen/S50+)."""
    return _CATEGORIE_AGE_BUCKETS.get(categorie_ffe, "unknown")


def bucket_pool_size_quartile(size: int, all_sizes: list[int]) -> str:
    """Quartile bucket : small/medium/large/xlarge based on dataset all_sizes.

    Uses strict less-than on q3 so the maximum value always lands in 'xlarge',
    keeping buckets meaningful for small datasets (n<=4).
    """
    if not all_sizes:
        msg = "all_sizes must be non-empty"
        raise ValueError(msg)
    sorted_sizes = sorted(all_sizes)
    n = len(sorted_sizes)
    q1 = sorted_sizes[n // 4]
    q2 = sorted_sizes[n // 2]
    q3 = sorted_sizes[3 * n // 4]
    if size <= q1:
        return "small"
    if size <= q2:
        return "medium"
    if size < q3:
        return "large"
    return "xlarge"


def bucket_elo_strata(team_mean_elo: int) -> str:
    """4-bucket Elo strata for team-level capability proxy."""
    if team_mean_elo < 1500:
        return "Q1_lt_1500"
    if team_mean_elo < 1700:
        return "Q2_1500_1700"
    if team_mean_elo < 1900:
        return "Q3_1700_1900"
    return "Q4_gte_1900"


# ---- Aggregator helper ----


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _build_group_stats(group_key: str, ms_list: list[MatchStats]) -> D8GroupStats:
    return D8GroupStats(
        group_key=group_key,
        n=len(ms_list),
        recall_mean=_mean([m.recall_ali for m in ms_list]),
        jaccard_mean=_mean([m.jaccard_ali for m in ms_list]),
        brier_mean=_mean([m.brier_ali for m in ms_list]),
        ece_mean=_mean([m.ece_ali for m in ms_list]),
        mae_mean=_mean([m.e_score_mae for m in ms_list]),
        bss_mean=_mean([m.bss for m in ms_list]),
    )


def _generic_breakdown(
    matches: list[MatchStats],
    dim_name: str,
    key_fn: Callable[[MatchStats], str],
) -> D8GroupBreakdown:
    groups_buckets: dict[str, list[MatchStats]] = {}
    for m in matches:
        k = key_fn(m)
        groups_buckets.setdefault(k, []).append(m)
    return D8GroupBreakdown(
        dim_name=dim_name,
        groups={k: _build_group_stats(k, v) for k, v in groups_buckets.items()},
    )


# ---- 7 dimensions ----


def breakdown_by_ronde(matches: list[MatchStats]) -> D8GroupBreakdown:
    """Group MatchStats by f'ronde_{m.ronde}'."""
    return _generic_breakdown(matches, "ronde", lambda m: f"ronde_{m.ronde}")


def breakdown_by_saison(matches: list[MatchStats]) -> D8GroupBreakdown:
    """Group MatchStats by str(m.saison)."""
    return _generic_breakdown(matches, "saison", lambda m: str(m.saison))


def breakdown_by_gender(
    matches: list[MatchStats],
    gender_fn: Callable[[MatchStats], str],
) -> D8GroupBreakdown:
    """Group by gender via external lookup (gender not in MatchStats)."""
    return _generic_breakdown(matches, "gender", gender_fn)


def breakdown_by_pool_size(
    matches: list[MatchStats],
    pool_size_fn: Callable[[MatchStats], int],
    all_pool_sizes: list[int] | None = None,
) -> D8GroupBreakdown:
    """Group by pool-size quartile (small/medium/large/xlarge)."""
    if all_pool_sizes is None:
        all_pool_sizes = [pool_size_fn(m) for m in matches] or [0]
    return _generic_breakdown(
        matches,
        "pool_size",
        lambda m: bucket_pool_size_quartile(pool_size_fn(m), all_pool_sizes),
    )


def breakdown_by_niveau(
    matches: list[MatchStats],
    niveau_fn: Callable[[MatchStats], str],
) -> D8GroupBreakdown:
    """Group by competition niveau via external lookup."""
    return _generic_breakdown(matches, "niveau", niveau_fn)


def breakdown_by_elo_strata(
    matches: list[MatchStats],
    team_elo_mean_fn: Callable[[MatchStats], int],
) -> D8GroupBreakdown:
    """Group by team-mean Elo strata (Q1..Q4)."""
    return _generic_breakdown(
        matches,
        "elo_strata_team",
        lambda m: bucket_elo_strata(team_elo_mean_fn(m)),
    )


def breakdown_by_categorie_age(
    matches: list[MatchStats],
    categorie_fn: Callable[[MatchStats], str],
) -> D8GroupBreakdown:
    """Group by FFE categorie age bucket (U12/U18/U20/Sen/S50+/unknown)."""
    return _generic_breakdown(
        matches,
        "categorie_age",
        lambda m: bucket_categorie_age(categorie_fn(m)),
    )


def compute_all_7(
    matches: list[MatchStats],
    gender_fn: Callable[[MatchStats], str],
    pool_size_fn: Callable[[MatchStats], int],
    all_pool_sizes: list[int],
    niveau_fn: Callable[[MatchStats], str],
    team_elo_mean_fn: Callable[[MatchStats], int],
    categorie_fn: Callable[[MatchStats], str],
) -> dict[str, D8GroupBreakdown]:
    """Compute all 7 dimensions breakdown in one call."""
    return {
        "by_gender": breakdown_by_gender(matches, gender_fn),
        "by_pool_size": breakdown_by_pool_size(matches, pool_size_fn, all_pool_sizes),
        "by_ronde": breakdown_by_ronde(matches),
        "by_saison": breakdown_by_saison(matches),
        "by_niveau": breakdown_by_niveau(matches, niveau_fn),
        "by_elo_strata": breakdown_by_elo_strata(matches, team_elo_mean_fn),
        "by_categorie_age": breakdown_by_categorie_age(matches, categorie_fn),
    }


# ---- Gap computation ----


def max_gap_recall(breakdown: D8GroupBreakdown) -> float:
    """Max - min recall across groups (Mehrabi 2021 fairness gap)."""
    if not breakdown.groups:
        return 0.0
    recalls = [g.recall_mean for g in breakdown.groups.values()]
    if any(math.isnan(r) for r in recalls):
        return float("nan")
    return max(recalls) - min(recalls)
