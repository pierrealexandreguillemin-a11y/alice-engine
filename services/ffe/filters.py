"""Per-article pre-composition filters for FFE RuleEngine.

Extracted from rule_engine.py for ISO 5055 compliance (module <= 300 lines).
Filters narrow a candidate pool before composition ; checkers validate the
final lineup. D-P3-11 Plan 2 Task 9 extension.

Document ID: ALICE-FFE-FILTERS
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.ali.types import CompetitionContext, PlayerCandidate


def filter_brule(pool: list[PlayerCandidate], context: CompetitionContext) -> list[PlayerCandidate]:
    """Article 3.7.c : exclude players burned for this team."""
    target_team = context.target_team_id
    target_rank = context.target_team_rank
    seuil = context.brule_seuil
    out = []
    for p in pool:
        entries = p.matchs_equipe_sup or ()
        blocked = any(
            cnt >= seuil and team != target_team and team_r < target_rank
            for team, cnt, team_r in entries
        )
        if not blocked:
            out.append(p)
    return out


def filter_match_count(
    pool: list[PlayerCandidate], context: CompetitionContext
) -> list[PlayerCandidate]:
    """Article 3.7.e : player must have played < ronde matchs."""
    return [p for p in pool if p.matchs_joues < context.ronde]


def filter_same_group(
    pool: list[PlayerCandidate], context: CompetitionContext
) -> list[PlayerCandidate]:
    """Article 3.7.d : player already in another group of same club blocked."""
    target = context.target_group
    return [p for p in pool if (p.group_history is None) or (p.group_history == target)]


def filter_elo_max(
    pool: list[PlayerCandidate], context: CompetitionContext
) -> list[PlayerCandidate]:
    """Article 3.7.j : drop players strictly above elo_max (if set)."""
    if context.elo_max is None:
        return list(pool)
    return [p for p in pool if p.elo <= context.elo_max]


def check_unique_assignment(teams: list[list[str]]) -> bool:
    """1 joueur = 1 equipe par match (cross-team helper)."""
    all_ids = [pid for team in teams for pid in team]
    return len(all_ids) == len(set(all_ids))
