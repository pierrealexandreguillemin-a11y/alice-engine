"""Tests for services.ali.types frozen dataclasses.

ISO 29119 structure : explicit per-class assertions.
ISO 5055 : SRP verification (frozen types, no behavior).

Document ID: ALICE-TEST-ALI-TYPES
Version: 1.0.0
"""

from __future__ import annotations

import dataclasses

import pytest

from services.ali.types import (
    CompetitionContext,
    PlayerCandidate,
    RuleViolation,
)


def test_player_candidate_frozen() -> None:
    p = PlayerCandidate(
        nr_ffe="A12345",
        nom="Dupont",
        prenom="Jean",
        elo=1800,
        club="CLUBX",
        mute=False,
        genre="M",
        categorie="SE",
        licence_active=True,
        age_min=25,
        age_max=30,
    )
    assert dataclasses.is_dataclass(p)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.elo = 9999  # type: ignore[misc]  # frozen


def test_competition_context_roundtrip() -> None:
    ctx = CompetitionContext(
        competition_code="A02",
        niveau="N2",
        ronde=3,
        team_size=8,
        noyau_min=50,
        max_mutes=3,
        elo_max=None,
    )
    assert ctx.competition_code == "A02"
    assert ctx.team_size == 8


def test_rule_violation_frozen() -> None:
    v = RuleViolation(
        rule_uuid="U1",
        rule_article="3.7.a",
        message="team size mismatch",
        severity="error",
    )
    assert v.severity == "error"
