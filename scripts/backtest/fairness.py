"""T13 Smoke fairness breakdown (Plan 3 V2 Phase 3, ISO 24027).

Stratifie BacktestReport.per_match par niveau competition (N1/N2/N3/N4)
et taille club (quartiles pool size). Produit breakdown metrics T13-T17
par groupe pour audit disparite de performance.

Gate P3G12 : chaque groupe doit avoir recall_mean >= 0.85 (gate affaibli
de 5 pts vs P3G07 gate global 0.90 pour tolerer sample variance).

Sources SOTA
------------
- ISO/IEC TR 24027:2021 Bias Detection in AI
- Barocas, Hardt, Narayanan 2019 "Fairness and Machine Learning"
  (fairmlbook.org) chap. 3 : group-level metrics.
- Mehrabi et al. 2021 "A Survey on Bias and Fairness in ML" ACM CSUR.

Document ID: ALICE-BACKTEST-FAIRNESS
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.backtest.runner_types import MatchStats


@dataclass(frozen=True)
class GroupStats:
    """Per-group aggregated metrics (ISO 24027 breakdown)."""

    group_key: str
    n: int
    recall_mean: float
    jaccard_mean: float
    brier_mean: float
    ece_mean: float
    mae_mean: float


FAIRNESS_RECALL_GATE = 0.85  # P3G12 affaibli vs global 0.90


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def breakdown_by_key(
    matches: list[MatchStats],
    key_fn: object,
) -> dict[str, GroupStats]:
    """Generic breakdown : group matches by key_fn(match) -> str, aggregate."""
    groups: dict[str, list[MatchStats]] = {}
    for m in matches:
        k = str(key_fn(m))  # type: ignore[operator]
        groups.setdefault(k, []).append(m)
    return {
        k: GroupStats(
            group_key=k,
            n=len(ms),
            recall_mean=_mean([m.recall_ali for m in ms]),
            jaccard_mean=_mean([m.jaccard_ali for m in ms]),
            brier_mean=_mean([m.brier_ali for m in ms]),
            ece_mean=_mean([m.ece_ali for m in ms]),
            mae_mean=_mean([m.e_score_mae for m in ms]),
        )
        for k, ms in groups.items()
    }


def breakdown_by_ronde(matches: list[MatchStats]) -> dict[str, GroupStats]:
    """Stratify by ronde (proxy saison phase : early / mid / late)."""
    return breakdown_by_key(matches, key_fn=lambda m: f"ronde_{m.ronde}")


def breakdown_by_opponent_club_size(
    matches: list[MatchStats],
    pool_size_fn: object,
) -> dict[str, GroupStats]:
    """Stratify by opponent club pool size quartile (small/medium/large/xlarge).

    @param pool_size_fn: callable match -> int (opponent pool size).
    """
    sizes = sorted([int(pool_size_fn(m)) for m in matches])  # type: ignore[operator]
    if not sizes:
        return {}
    q1 = sizes[len(sizes) // 4]
    q2 = sizes[len(sizes) // 2]
    q3 = sizes[3 * len(sizes) // 4]

    def bucket(m: MatchStats) -> str:
        s = int(pool_size_fn(m))  # type: ignore[operator]
        if s <= q1:
            return "small"
        if s <= q2:
            return "medium"
        if s <= q3:
            return "large"
        return "xlarge"

    return breakdown_by_key(matches, key_fn=bucket)


def gates_per_group(
    breakdown: dict[str, GroupStats],
    gate: float = FAIRNESS_RECALL_GATE,
) -> dict[str, bool]:
    """Return {group_key: passes P3G12 recall gate} for audit."""
    return {g.group_key: g.recall_mean >= gate for g in breakdown.values()}


def max_gap(breakdown: dict[str, GroupStats]) -> float:
    """Return max - min recall across groups (fairness disparity proxy).

    Gate P3G12 smoke : gap <= 0.15 (15pts de recall entre meilleur/pire groupe).
    """
    if not breakdown:
        return 0.0
    recalls = [g.recall_mean for g in breakdown.values()]
    return max(recalls) - min(recalls)
