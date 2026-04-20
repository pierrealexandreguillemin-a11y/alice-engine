"""Tests T9 McNemar paired — Plan 3 V2 Phase 3.

ISO 29119 : exact binomial vs chi2 branches, gate, edge cases, determinism.
"""

from __future__ import annotations

import pytest

from scripts.backtest.statistical import McNemarResult, mcnemar_paired


def test_mcnemar_identical_predictions_high_p():
    """Identiques → b == c == 0, pas de discordants → p_value = 1.0."""
    ali = [True, True, False, False]
    base = [True, True, False, False]
    res = mcnemar_paired(ali, base)
    assert res.b == 0
    assert res.c == 0
    assert res.n_discordant == 0
    assert res.p_value == pytest.approx(1.0)
    assert res.significant is False


def test_mcnemar_ali_dominates_small_sample_exact():
    """ALI domine sur 10 matches discordants (n_disc<25) → exact binomial."""
    # b = 10, c = 0 → exact binomial p = 2 * 0.5**10 = 0.00195
    ali = [True] * 10 + [False] * 5
    base = [False] * 10 + [False] * 5
    res = mcnemar_paired(ali, base)
    assert res.b == 10
    assert res.c == 0
    assert res.method == "exact_binomial"
    assert res.p_value < 0.01
    assert res.significant is True


def test_mcnemar_large_sample_uses_chi2():
    """n_disc >= 25 → chi2 Yates corrected. Forte asymetrie → p < 0.05."""
    # b = 25, c = 5 → Yates = (|25-5|-0.5)²/30 = 12.675 → p ~ 0.00037
    ali = [True] * 25 + [False] * 5 + [True] * 20
    base = [False] * 25 + [True] * 5 + [True] * 20
    res = mcnemar_paired(ali, base)
    assert res.b == 25
    assert res.c == 5
    assert res.n_discordant == 30
    assert res.method == "chi2_continuity"
    assert res.p_value < 0.05
    assert res.significant is True


def test_mcnemar_no_difference_large_sample():
    """Symetrique b==c sur grand echantillon → p_value eleve, non significatif.

    Yates continuity correction injects a small bias even when |b-c|=0
    (stat = 1/(b+c) > 0, p proche mais pas exactement 1.0).
    """
    ali = [True] * 15 + [False] * 15
    base = [False] * 15 + [True] * 15
    res = mcnemar_paired(ali, base)
    assert res.b == 15
    assert res.c == 15
    assert res.p_value > 0.5
    assert res.significant is False


def test_mcnemar_result_is_frozen():
    """McNemarResult immutable."""
    res = mcnemar_paired([True], [False])
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        res.p_value = 0.5  # type: ignore[misc]


def test_mcnemar_passes_gate_default():
    ali = [True] * 10
    base = [False] * 10
    res = mcnemar_paired(ali, base)
    assert res.passes_gate(alpha=0.05) is True
    assert res.passes_gate(alpha=0.001) is False  # stricter


def test_mcnemar_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        mcnemar_paired([True, True], [False])


def test_mcnemar_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mcnemar_paired([], [])


def test_mcnemar_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        mcnemar_paired([True], [False], alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        mcnemar_paired([True], [False], alpha=0.0)


def test_mcnemar_deterministic_same_input():
    """Même input → même résultat (ISO 29119)."""
    ali = [True, False, True, False]
    base = [False, True, True, False]
    res1 = mcnemar_paired(ali, base)
    res2 = mcnemar_paired(ali, base)
    assert res1.p_value == res2.p_value
    assert res1.statistic == res2.statistic


def test_mcnemar_ties_only_match_count():
    """ALI et baseline toujours d'accord (b=c=0) → returns pvalue=1, not crash."""
    ali = [True, False, True, False, True]
    base = [True, False, True, False, True]
    res = mcnemar_paired(ali, base)
    assert isinstance(res, McNemarResult)
    assert res.n_discordant == 0
    assert res.significant is False
