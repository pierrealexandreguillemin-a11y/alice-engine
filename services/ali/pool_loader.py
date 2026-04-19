"""PlayerPoolLoader — charge l'effectif eligible d'un club a une date.

F7 : filter `licence_active` (survivor bias protection).
Source: Brown, Goetzmann, Ross, Ibbotson 1992 (transpose sports depuis finance).

ISO 5055 : SRP strict (loading + filtering, pas d'enrichment).
ISO 24027 : F7 assumption documented in Model Card.

Document ID: ALICE-ALI-POOL-LOADER
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from services.ali.types import PlayerCandidate

if TYPE_CHECKING:
    from services.ali.cache import ALIDataCache


class PlayerPoolLoader:
    """Charge le pool joueurs eligibles pour un club x date donnes."""

    def __init__(self, cache: ALIDataCache) -> None:
        """Store cache reference (no I/O; cache preloaded at lifespan)."""
        self._cache = cache

    def load_pool(
        self,
        club_id: str,
        round_date: str,  # noqa: ARG002 (reserve usage futur J02)
        overrides: list[dict[str, Any]] | None = None,
    ) -> list[PlayerCandidate]:
        """Return eligible candidates. F7 survivor filter applied."""
        df = self._cache.lookup_club(club_id)
        if df.empty and not overrides:
            return []

        candidates: dict[str, PlayerCandidate] = {}
        for _, row in df.iterrows():
            c = _row_to_candidate(row)
            if not c.licence_active:
                continue  # F7 survivor filter
            candidates[c.nr_ffe] = c

        if overrides:
            for raw in overrides:
                c = _override_to_candidate(raw)
                candidates[c.nr_ffe] = c

        return list(candidates.values())


def _row_to_candidate(row: Any) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=str(row["nr_ffe"]),
        nom=str(row.get("nom", "")),
        prenom=str(row.get("prenom", "")),
        elo=int(row.get("elo") or 1500),
        club=str(row.get("club", "")),
        mute=bool(row.get("mute", False)),
        genre=str(row.get("genre", "M")),
        categorie=str(row.get("categorie", "SE")),
        licence_active=_row_licence_active(row),
        age_min=_to_int_or_none(row.get("age_min")),
        age_max=_to_int_or_none(row.get("age_max")),
    )


def _row_licence_active(row: Any) -> bool:
    """Deduce licence active. Conservative: True unless explicit flag."""
    elo_type = str(row.get("elo_type", "")).upper()
    if elo_type in {"ARCHIVE", "INACTIVE"}:
        return False
    return True


def _override_to_candidate(raw: dict[str, Any]) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=str(raw["nr_ffe"]),
        nom=str(raw.get("nom", "")),
        prenom=str(raw.get("prenom", "")),
        elo=int(raw.get("elo", 1500)),
        club=str(raw.get("club", "")),
        mute=bool(raw.get("mute", False)),
        genre=str(raw.get("genre", "M")),
        categorie=str(raw.get("categorie", "SE")),
        licence_active=bool(raw.get("licence_active", True)),
        age_min=_to_int_or_none(raw.get("age_min")),
        age_max=_to_int_or_none(raw.get("age_max")),
    )


def _to_int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None
