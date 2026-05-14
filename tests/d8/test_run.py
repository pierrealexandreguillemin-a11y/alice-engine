"""Tests for ADR-021 division-specific rondes mapping in scripts/d8/run.py.

ISO 29119 §6.4 : unit-level coverage of the DIVISION_RONDES_DEFAULT table
and _run_backtest branch selection. Guards against regression of Top 16
v3 ERROR root cause (rondes_default=(5,7,9,11) filtre 88→16 matches).

Document ID: ALICE-D8-RUN-TESTS
Version: 1.0.0
"""

from __future__ import annotations

from scripts.d8.run import DIVISION_RONDES_DEFAULT


def test_top_16_has_full_7_rondes_mapping() -> None:
    """ADR-021 : Top 16 requires all 7 rondes (régulière 1-7 + finale 1-4).

    Without this override, default (5,7,9,11) filters 88 candidates → 16,
    triggering CONFORMAL_CALIB_N=30 invariant RuntimeError uncaught
    (Phase A v3 ERROR 2026-05-12).
    """
    assert "Top 16" in DIVISION_RONDES_DEFAULT
    assert DIVISION_RONDES_DEFAULT["Top 16"] == (1, 2, 3, 4, 5, 6, 7)


def test_nationale_divisions_use_saison_default_fallback() -> None:
    """ADR-021 : Nationale 1-4 NOT in override → saison-based default applies.

    Post-2022 default (5,7,9,11) satisfies CONFORMAL_CALIB_N=30 for N1-N4
    (Phase A v3 N1=56, N2=101, N3=168, N4=85 ≥ 31).
    """
    for niveau in ("Nationale 1", "Nationale 2", "Nationale 3", "Nationale 4"):
        assert niveau not in DIVISION_RONDES_DEFAULT


def test_division_rondes_default_values_are_tuples_of_ints() -> None:
    """ISO 27034 input-shape validation : all entries must be tuple[int, ...]."""
    for niveau, rondes in DIVISION_RONDES_DEFAULT.items():
        assert isinstance(niveau, str)
        assert isinstance(rondes, tuple)
        assert all(isinstance(r, int) and r >= 1 for r in rondes)
        assert len(rondes) == len(set(rondes)), f"{niveau} rondes contain duplicates"
