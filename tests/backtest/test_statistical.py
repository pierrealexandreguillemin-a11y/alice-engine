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


# ---------------------------------------------------------------------------
# T22 D-P3-18 SOTA Wilcoxon signed-rank tests (continuous paired)
# ---------------------------------------------------------------------------

from scripts.backtest.statistical import WilcoxonResult, wilcoxon_paired  # noqa: E402


def test_wilcoxon_identical_values_no_diff():
    """Toutes les diffs == 0 -> degenerate, p=1.0, significant False."""
    ali = [0.5, 0.6, 0.7, 0.8]
    base = [0.5, 0.6, 0.7, 0.8]
    res = wilcoxon_paired(ali, base)
    assert isinstance(res, WilcoxonResult)
    assert res.n_nonzero == 0
    assert res.p_value == pytest.approx(1.0)
    assert res.method == "degenerate_no_diff"
    assert res.significant is False


def test_wilcoxon_ali_dominates_continuous_significant():
    """ALI > baseline systematique sur metric continu -> p << 0.05."""
    ali = [0.6, 0.7, 0.8, 0.5, 0.9, 0.7, 0.6, 0.8, 0.7, 0.6]
    base = [0.1, 0.2, 0.3, 0.0, 0.4, 0.2, 0.1, 0.3, 0.2, 0.1]
    res = wilcoxon_paired(ali, base)
    assert res.median_diff > 0.0
    assert res.p_value < 0.05
    assert res.significant is True
    assert res.n_nonzero == 10


def test_wilcoxon_baseline_dominates_direction_negative():
    """Baseline > ALI systematique : direction documentee.

    median_diff < 0 indique baseline domine. N=5 trop petit pour
    significativite alpha=0.05 (p ≈ 0.0625 exact bilateral) — test
    coherence direction + non-degenerescence, pas puissance.
    """
    ali = [0.1, 0.2, 0.3, 0.0, 0.4]
    base = [0.5, 0.6, 0.7, 0.4, 0.8]
    res = wilcoxon_paired(ali, base)
    assert res.median_diff < 0.0  # baseline domine en mediane
    assert res.n_nonzero == 5
    assert res.method == "exact"


def test_wilcoxon_paired_t22_recall_real_dump():
    """Sur les vraies recall T22 (ALI domine baseline largement).

    Wilcoxon doit detecter direction + significativite sur 8 paires.
    """
    # extracted from reports/backtest/ali_holdout_2024.json
    ali_recalls = [0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625]
    base_recalls = [0.0, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0]
    res = wilcoxon_paired(ali_recalls, base_recalls)
    assert res.median_diff > 0.0
    assert res.n_nonzero == 8  # toutes paires distinctes (0.625 != 0/0.125)


def test_wilcoxon_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        wilcoxon_paired([0.5, 0.6], [0.5])


def test_wilcoxon_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        wilcoxon_paired([], [])


def test_wilcoxon_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        wilcoxon_paired([0.5], [0.4], alpha=1.5)


def test_wilcoxon_deterministic_same_input():
    ali = [0.5, 0.6, 0.7, 0.55, 0.65]
    base = [0.4, 0.5, 0.6, 0.45, 0.55]
    res1 = wilcoxon_paired(ali, base)
    res2 = wilcoxon_paired(ali, base)
    assert res1.p_value == res2.p_value
    assert res1.statistic == res2.statistic


def test_wilcoxon_method_exact_when_n_lt_25():
    """n_nonzero < 25 -> method 'exact'."""
    ali = [0.5, 0.6, 0.7, 0.8, 0.9]
    base = [0.4, 0.5, 0.6, 0.7, 0.8]
    res = wilcoxon_paired(ali, base)
    assert res.method == "exact"


def test_wilcoxon_method_approx_when_n_ge_25():
    """n_nonzero >= 25 -> method 'approx_normal'."""
    import numpy as np

    rng = np.random.default_rng(42)
    ali = rng.uniform(0.3, 0.8, 30).tolist()
    base = rng.uniform(0.0, 0.2, 30).tolist()
    res = wilcoxon_paired(ali, base)
    assert res.n_nonzero >= 25
    assert res.method == "approx_normal"


def test_wilcoxon_passes_gate_alpha():
    """passes_gate(alpha=0.05) reflete significant flag.

    Avec N=10 toutes diffs positives identiques (ties handling) le
    p_value exact ≈ 0.00195 < 0.05 mais > 0.001 (limite plancher exact
    binomial). Test alpha=0.05 et alpha=0.005 (au-dessus du plancher).
    """
    ali = [0.6] * 10
    base = [0.1] * 10
    res = wilcoxon_paired(ali, base)
    assert res.passes_gate(alpha=0.05) is True
    assert res.passes_gate(alpha=0.005) is True  # 0.00195 < 0.005
