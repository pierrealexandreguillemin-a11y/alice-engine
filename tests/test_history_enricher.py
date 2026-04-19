"""Tests for services.ali.history HistoryEnricher F2 recency decay.

ISO 29119 structure : explicit per-case assertions.
ISO 5259 : lambda parameter traced via decay_lambda.
Source : Brown 1959 exponential smoothing, Silver 2012 FiveThirtyEight methodology.

Document ID: ALICE-TEST-ALI-HISTORY
Version: 1.0.0
"""

from __future__ import annotations

import pandas as pd

from services.ali.history import compute_recency_weighted_presence


def _hist(rows: list[tuple[str, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["joueur_nom", "saison", "ronde", "echiquier"])


def test_recency_all_rounds_played_returns_one() -> None:
    hist = _hist([("Jean", 2024, r, 1) for r in range(1, 12)])
    taux = compute_recency_weighted_presence(
        hist,
        player_name="Jean",
        round_date_saison=2024,
        nb_rondes_total=11,
        current_round=12,
        decay_lambda=0.9,
    )
    assert 0.999 < taux <= 1.0


def test_recency_no_rounds_played_returns_zero() -> None:
    hist = _hist([])
    taux = compute_recency_weighted_presence(
        hist,
        "Inconnu",
        2024,
        nb_rondes_total=11,
        current_round=12,
        decay_lambda=0.9,
    )
    assert taux == 0.0


def test_recency_weights_recent_more_than_old() -> None:
    hist = _hist(
        [
            ("A", 2024, 1, 1),
            ("B", 2024, 11, 1),
        ]
    )
    taux_a = compute_recency_weighted_presence(hist, "A", 2024, 11, 12, 0.9)
    taux_b = compute_recency_weighted_presence(hist, "B", 2024, 11, 12, 0.9)
    assert taux_b > taux_a


def test_recency_lambda_1_equals_flat_rate() -> None:
    hist = _hist([("X", 2024, r, 1) for r in [1, 3, 5, 7, 9]])
    taux = compute_recency_weighted_presence(hist, "X", 2024, 11, 12, 1.0)
    assert abs(taux - 5 / 11) < 1e-9
