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
        exclude_players: set[str] | None = None,
    ) -> list[PlayerCandidate]:
        """Return eligible candidates. F7 survivor filter applied.

        `exclude_players` (Phase 4a D-P3-19): nr_ffe already consumed by the
        opponent club's superior teams (A02 §3.7.b top-down). Applied LAST,
        after overrides, so an injected override cannot re-introduce an
        excluded player.
        """
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

        exclude = exclude_players or set()
        return [c for c in candidates.values() if c.nr_ffe not in exclude]


def _row_to_candidate(row: Any) -> PlayerCandidate:
    genre = str(row.get("genre", "M"))
    return PlayerCandidate(
        nr_ffe=str(row["nr_ffe"]),
        nom=str(row.get("nom", "")),
        prenom=str(row.get("prenom", "")),
        elo=int(row.get("elo") or 1500),
        club=str(row.get("club", "")),
        mute=bool(row.get("mute", False)),
        genre=genre,
        categorie=str(row.get("categorie", "SE")),
        licence_active=_row_licence_active(row),
        age_min=_to_int_or_none(row.get("age_min")),
        age_max=_to_int_or_none(row.get("age_max")),
        sexe=genre,  # M1: derive sexe from genre (C7 §3.7.i reads .sexe)
    )


def _row_licence_active(row: Any) -> bool:  # noqa: ARG001
    """Return True : joueurs.parquet ne contient que des licences actives FFE.

    F7 (survivor bias) est enforce via la composition meme du parquet :
    joueurs.parquet est mis a jour regulierement par scraping FFE actif.
    Joueurs ayant quitte le club n'apparaissent plus dans
    joueurs_by_club[club_id] -> filtre implicite par membership.

    Source : analyse schema reel 2026-04-19 (D-P3-04), valeurs elo_type
    dans {E, F, N, ''}. Aucune valeur ARCHIVE/INACTIVE observee sur
    83K lignes. Override possible via overrides[*].licence_active = false
    (capitaine signale licence en cours de renouvellement, cas exceptionnel).
    """
    return True


def _override_to_candidate(raw: dict[str, Any]) -> PlayerCandidate:
    genre = str(raw.get("genre", "M"))
    return PlayerCandidate(
        nr_ffe=str(raw["nr_ffe"]),
        nom=str(raw.get("nom", "")),
        prenom=str(raw.get("prenom", "")),
        elo=int(raw.get("elo", 1500)),
        club=str(raw.get("club", "")),
        mute=bool(raw.get("mute", False)),
        genre=genre,
        categorie=str(raw.get("categorie", "SE")),
        licence_active=bool(raw.get("licence_active", True)),
        age_min=_to_int_or_none(raw.get("age_min")),
        age_max=_to_int_or_none(raw.get("age_max")),
        sexe=genre,  # M1: derive sexe from genre (C7 §3.7.i reads .sexe)
    )


def _to_int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None
