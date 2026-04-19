"""HistoryEnricher — enrichissement joueur avec features ALI.

F2 : recency decay exponentiel (Brown 1959 exponential smoothing, Silver 2012
methodology FiveThirtyEight). taux effectif pondere rondes recentes > anciennes.

F3 : autoregressive streak lag 1-3 (Box & Jenkins 1970, Pappalardo 2019)
— ajoute en Task 14.

ISO 5259 : lambda parametre trace dans lineage_hash (via `decay_lambda`).
ISO 42001 : explainability via features separees.

Document ID: ALICE-ALI-HISTORY-ENRICHER
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from services.ali.cache import ALIDataCache
    from services.ali.types import PlayerCandidate


def compute_recency_weighted_presence(  # noqa: PLR0913
    history: pd.DataFrame,
    player_name: str,
    round_date_saison: int,
    nb_rondes_total: int,
    current_round: int,
    decay_lambda: float = 0.9,
) -> float:
    """F2 : taux de presence pondere par exponential decay lambda.

    Formule :
        taux_effectif = Sigma_r lambda^(age_r) * 1[player plays r] / Sigma_r lambda^(age_r)
    ou age_r = max(current_round - r, 0).
    """
    if nb_rondes_total <= 0:
        return 0.0

    played_rounds = _played_rounds(history, player_name, round_date_saison)
    numerator = 0.0
    denominator = 0.0
    for r in range(1, nb_rondes_total + 1):
        age = max(current_round - r, 0)
        weight = decay_lambda**age
        denominator += weight
        if r in played_rounds:
            numerator += weight

    return numerator / denominator if denominator > 0 else 0.0


def _played_rounds(
    history: pd.DataFrame,
    player_name: str,
    round_date_saison: int,
) -> set[int]:
    """Return set of rounds played by player_name in round_date_saison."""
    sub = history[(history["joueur_nom"] == player_name) & (history["saison"] == round_date_saison)]
    return set(sub["ronde"].dropna().astype(int).tolist())


class HistoryEnricher:
    """Enrichit les PlayerCandidates avec features ALI calculees a inference time."""

    def __init__(
        self,
        cache: ALIDataCache,
        decay_lambda: float = 0.9,
        streak_lag: int = 3,
    ) -> None:
        """Store cache + F2/F3 hyperparams (lambda traced via ISO 5259 lineage)."""
        self._cache = cache
        self._lambda = decay_lambda
        self._streak_lag = streak_lag

    def enrich(
        self,
        candidates: list[PlayerCandidate],
        saison: int,
        current_round: int,
        nb_rondes_total: int,
    ) -> list[PlayerCandidate]:
        """Return enriched candidates with F2 taux_presence_effectif set."""
        names = [self._player_lookup_name(c) for c in candidates]
        history_raw = self._cache.lookup_history(names)
        history = _normalize_history(history_raw)

        enriched: list[PlayerCandidate] = []
        for c in candidates:
            lookup_name = self._player_lookup_name(c)
            taux = compute_recency_weighted_presence(
                history,
                lookup_name,
                saison,
                nb_rondes_total,
                current_round,
                self._lambda,
            )
            enriched.append(replace(c, taux_presence_effectif=taux))
        return enriched

    @staticmethod
    def _player_lookup_name(c: PlayerCandidate) -> str:
        """Reconstruct the name format stored in echiquiers.parquet."""
        return f"{c.nom} {c.prenom}".strip()


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Echiquiers has blanc_nom AND noir_nom. Union into joueur_nom."""
    parts = []
    for col in ("blanc_nom", "noir_nom"):
        if col in df.columns:
            sub = df[[col, "saison", "ronde", "echiquier"]].copy()
            sub = sub.rename(columns={col: "joueur_nom"})
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=["joueur_nom", "saison", "ronde", "echiquier"])
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=["joueur_nom", "saison", "ronde"])
