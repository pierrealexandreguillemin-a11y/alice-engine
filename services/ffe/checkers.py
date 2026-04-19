"""Per-article lineup checkers for FFE RuleEngine.

Extracted from rule_engine.py for ISO 5055 compliance (module <= 300 lines).
Each checker returns a RuleViolation | None. Rank A cyclomatique.

D-P3-11 Plan 2 Task 9 extension : adds 6 articles missing from legacy
services/ffe_rules.py migration (3.7.c brule, 3.7.d same_group,
3.7.e match_count, 3.7.f noyau, 3.7.h foreign_quota, 3.7.i fr_gender).

Document ID: ALICE-FFE-CHECKERS
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.ali.types import RuleViolation

if TYPE_CHECKING:
    from services.ali.types import CompetitionContext, PlayerCandidate
    from services.ffe.rule_engine import Rule


def check_team_size(
    rule: Rule,
    lineup: list[PlayerCandidate],
    context: CompetitionContext,
) -> RuleViolation | None:
    """Article 3.7.a : lineup length matches expected team_size."""
    expected_raw = rule.conditions.get("team_size", context.team_size)
    expected = int(expected_raw) if expected_raw is not None else context.team_size
    if len(lineup) != expected:
        return RuleViolation(
            rule_uuid=rule.uuid,
            rule_article=rule.article,
            message=f"team_size: expected {expected}, got {len(lineup)}",
            severity="error",
        )
    return None


def check_elo_order(
    rule: Rule,
    lineup: list[PlayerCandidate],
    context: CompetitionContext,  # noqa: ARG001  (dispatcher signature parity)
) -> RuleViolation | None:
    """Article 3.6.e : Elo descending across boards within tolerance."""
    tolerance_raw = rule.conditions.get("elo_tolerance", 100)
    tolerance = int(tolerance_raw)
    elos = [p.elo for p in lineup]
    for i in range(len(elos) - 1):
        if elos[i + 1] - elos[i] > tolerance:
            return RuleViolation(
                rule_uuid=rule.uuid,
                rule_article=rule.article,
                message=(f"elo order: board {i + 1}={elos[i]} < board {i + 2}={elos[i + 1]} + tol"),
                severity="error",
            )
    return None


def check_mutes_limit(
    rule: Rule,
    lineup: list[PlayerCandidate],
    context: CompetitionContext,
) -> RuleViolation | None:
    """Article 3.7.g : number of muted players within allowed max."""
    max_m_raw = rule.conditions.get("max_mutes", context.max_mutes)
    max_m = int(max_m_raw) if max_m_raw is not None else context.max_mutes
    muted = sum(1 for p in lineup if p.mute)
    if muted > max_m:
        return RuleViolation(
            rule_uuid=rule.uuid,
            rule_article=rule.article,
            message=f"mutes: {muted} > max {max_m}",
            severity="error",
        )
    return None


def check_elo_max(
    rule: Rule,
    lineup: list[PlayerCandidate],
    context: CompetitionContext,
) -> RuleViolation | None:
    """Article 3.7.j : no player above optional elo_max cap."""
    if context.elo_max is None:
        return None
    over = [p for p in lineup if p.elo > context.elo_max]
    if over:
        return RuleViolation(
            rule_uuid=rule.uuid,
            rule_article=rule.article,
            message=f"elo_max: {len(over)} players above {context.elo_max}",
            severity="error",
        )
    return None


def check_brule(
    rule: Rule,
    lineup: list[PlayerCandidate],
    context: CompetitionContext,
) -> RuleViolation | None:
    """Article 3.7.c : no burned player aligned in weaker team.

    A player is burned if they played >= brule_seuil matches in a
    STRONGER team (lower rank than target_team_rank).
    """
    seuil = context.brule_seuil
    target = context.target_team_id
    target_rank = context.target_team_rank
    burned = [
        p
        for p in lineup
        if p.matchs_equipe_sup
        and any(
            cnt >= seuil and team != target and team_r < target_rank
            for team, cnt, team_r in p.matchs_equipe_sup
        )
    ]
    if burned:
        return RuleViolation(
            rule_uuid=rule.uuid,
            rule_article=rule.article,
            message=f"brule: {len(burned)} burned players aligned",
            severity="error",
        )
    return None


def check_match_count(
    rule: Rule,
    lineup: list[PlayerCandidate],
    context: CompetitionContext,
) -> RuleViolation | None:
    """Article 3.7.e : no player with matchs_joues >= ronde."""
    over = [p for p in lineup if p.matchs_joues >= context.ronde]
    if over:
        return RuleViolation(
            rule_uuid=rule.uuid,
            rule_article=rule.article,
            message=f"match_count: {len(over)} players at quota (ronde={context.ronde})",
            severity="error",
        )
    return None


def check_same_group(
    rule: Rule,
    lineup: list[PlayerCandidate],
    context: CompetitionContext,
) -> RuleViolation | None:
    """Article 3.7.d : player already in another group of same club blocked."""
    target = context.target_group
    wrong = [p for p in lineup if (p.group_history is not None) and (p.group_history != target)]
    if wrong:
        return RuleViolation(
            rule_uuid=rule.uuid,
            rule_article=rule.article,
            message=f"same_group: {len(wrong)} players from another group",
            severity="error",
        )
    return None


def check_noyau(
    rule: Rule,
    lineup: list[PlayerCandidate],
    context: CompetitionContext,
) -> RuleViolation | None:
    """Article 3.7.f : ronde > 1 requires >= 50% noyau players.

    Ronde 1 : pas encore de noyau forme -> toujours OK.
    Noyau vide en context : check short-circuit -> OK (backward-compat).
    """
    if context.ronde <= 1 or not context.noyau:
        return None
    core_count = sum(1 for p in lineup if p.nr_ffe in context.noyau)
    if core_count * 2 < len(lineup):
        return RuleViolation(
            rule_uuid=rule.uuid,
            rule_article=rule.article,
            message=f"noyau: {core_count}/{len(lineup)} players in noyau (min 50%)",
            severity="error",
        )
    return None


def check_foreign_quota(
    rule: Rule,
    lineup: list[PlayerCandidate],
    context: CompetitionContext,
) -> RuleViolation | None:
    """Article 3.7.h : at least min_fr_eu FR/UE players in lineup."""
    fr_eu = sum(1 for p in lineup if p.is_french_eu)
    if fr_eu < context.min_fr_eu:
        return RuleViolation(
            rule_uuid=rule.uuid,
            rule_article=rule.article,
            message=f"foreign_quota: {fr_eu}/{context.min_fr_eu} FR/UE players",
            severity="error",
        )
    return None


def check_fr_gender(
    rule: Rule,
    lineup: list[PlayerCandidate],
    context: CompetitionContext,
) -> RuleViolation | None:
    """Article 3.7.i : N1/N2/Top16 require at least 1 FR male + 1 FR female."""
    if context.niveau not in ("N1", "N2", "Top16"):
        return None
    has_male = any(p.is_french and p.sexe == "M" for p in lineup)
    has_female = any(p.is_french and p.sexe == "F" for p in lineup)
    if not (has_male and has_female):
        return RuleViolation(
            rule_uuid=rule.uuid,
            rule_article=rule.article,
            message="fr_gender: missing 1 FR male + 1 FR female for N1/N2/Top16",
            severity="error",
        )
    return None
