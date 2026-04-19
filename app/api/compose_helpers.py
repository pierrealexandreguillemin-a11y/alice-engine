"""FFE filter/validate helpers for /compose (ISO 5055 SRP, <300 lines).

D-P3-11 Plan 2 Task 9 : RuleEngine-backed helpers replace services/ffe_rules.py.
Separate from compose_scenarios.py for module size caps.

Document ID: ALICE-COMPOSE-HELPERS
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.ali.types import PlayerCandidate
from services.ffe.rule_engine import RuleEngine

if TYPE_CHECKING:
    from app.api.schemas import BoardResult, ComposeRequest
    from services.ali.types import CompetitionContext


def build_player_pool(body: ComposeRequest) -> list[PlayerCandidate]:
    """Build synthetic player pool (Phase 2: no MongoDB, all Elo 1500).

    D-P3-11 : list[PlayerCandidate] for direct RuleEngine consumption.
    """
    return [
        PlayerCandidate(
            nr_ffe=fid,
            nom=fid,
            prenom="",
            elo=1500,
            club=body.club_id,
            mute=False,
            genre="M",
            categorie="Sen",
            licence_active=True,
        )
        for fid in body.joueurs_disponibles
    ]


def apply_pre_filters(
    engine: RuleEngine | None,
    players: list[PlayerCandidate],
    ctx: CompetitionContext,
) -> list[PlayerCandidate]:
    """FFE pre-filter via RuleEngine.filter_by_article for 3.7.c / 3.7.e / 3.7.d.

    Falls back to no-op if engine unavailable (Phase 2 degraded path).
    """
    if engine is None:
        return list(players)
    eligible = engine.filter_by_article(players, "3.7.c", ctx)
    eligible = engine.filter_by_article(eligible, "3.7.e", ctx)
    return engine.filter_by_article(eligible, "3.7.d", ctx)


def _boards_to_lineup(
    boards: list[BoardResult], selected: list[PlayerCandidate]
) -> list[PlayerCandidate]:
    """Map BoardResults back to their PlayerCandidate (preserves metadata)."""
    by_ffe = {p.nr_ffe: p for p in selected}
    out: list[PlayerCandidate] = []
    for b in boards:
        base = by_ffe.get(b.joueur)
        if base is None:
            out.append(
                PlayerCandidate(
                    nr_ffe=b.joueur,
                    nom=b.joueur,
                    prenom="",
                    elo=b.elo,
                    club="",
                    mute=False,
                    genre="M",
                    categorie="Sen",
                    licence_active=True,
                )
            )
            continue
        out.append(PlayerCandidate(**{**base.__dict__, "elo": b.elo}))
    return out


def validate_ffe(
    engine: RuleEngine | None,
    boards: list[BoardResult],
    selected: list[PlayerCandidate],
    ctx: CompetitionContext,
) -> bool:
    """FFE post-check via RuleEngine.validate_lineup (D-P3-11 migration).

    True if no error-severity violation (or engine absent -> permissive).
    """
    if engine is None:
        return True
    lineup = _boards_to_lineup(boards, selected)
    if not RuleEngine.check_unique_assignment([[b.joueur for b in boards]]):
        return False
    violations = engine.validate_lineup(lineup, ctx)
    return not any(v.severity == "error" for v in violations)
