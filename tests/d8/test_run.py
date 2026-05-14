"""Tests for ADR-021 division-specific rondes mapping in scripts/d8/run.py.

ISO 29119 §6.4 : unit-level coverage of the DIVISION_RONDES_DEFAULT table
and _run_backtest branch selection. Guards against regression of Top 16
v3 ERROR root cause (rondes_default=(5,7,9,11) filtre 88→16 matches).

Pure-function tests also restore global coverage > 70% per
D-2026-05-12-coverage-restore.

Document ID: ALICE-D8-RUN-TESTS
Version: 1.1.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from scripts.d8.run import (
    DIVISION_RONDES_DEFAULT,
    _checkpoint_partial,
    _compute_calibration_stage,
    _compute_conformal_stage,
    _resolve_code_sha,
    _validate_per_match_finite,
    _validate_saison,
)


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


# --- ISO 27034 input validation pure-function tests (coverage restoration) ---


@pytest.mark.parametrize("saison", [2018, 2024, 2030])
def test_validate_saison_accepts_in_range(saison: int) -> None:
    """Happy path : 2018-2030 in SAISON_MIN..SAISON_MAX."""
    _validate_saison(saison)


@pytest.mark.parametrize("saison", [2017, 2031, 0, -1])
def test_validate_saison_rejects_out_of_range(saison: int) -> None:
    """ISO 27034 fail-fast on out-of-range saison."""
    with pytest.raises(ValueError, match=r"ALICE_SAISON.*outside"):
        _validate_saison(saison)


@dataclass
class _StubMatch:
    """Minimal MatchStats-like for finite validation tests."""

    recall_ali: float = 0.5
    brier_ali: float = 0.2
    ece_ali: float = 0.05
    bss: float = 0.1
    e_score_predicted: float = 4.0


def test_validate_per_match_finite_accepts_finite() -> None:
    """Happy path : all finite floats pass."""
    matches = [_StubMatch(), _StubMatch(recall_ali=0.7)]
    _validate_per_match_finite(matches, saison=2024)


@pytest.mark.parametrize(
    "bad_attr", ["recall_ali", "brier_ali", "ece_ali", "bss", "e_score_predicted"]
)
def test_validate_per_match_finite_rejects_nan(bad_attr: str) -> None:
    """ISO 27034 fail-fast on NaN attr."""
    m = _StubMatch()
    setattr(m, bad_attr, math.nan)
    with pytest.raises(RuntimeError, match=r"non-finite"):
        _validate_per_match_finite([m], saison=2024)


def test_validate_per_match_finite_rejects_inf() -> None:
    """ISO 27034 fail-fast on inf."""
    m = _StubMatch(recall_ali=math.inf)
    with pytest.raises(RuntimeError, match=r"non-finite"):
        _validate_per_match_finite([m], saison=2024)


# --- Calibration + conformal stages (pure aggregation over MatchStats) ---


@dataclass
class _CalibMatch:
    """MatchStats-like with ronde + ece_ali for calibration tests."""

    ronde: int = 5
    ece_ali: float = 0.04


def test_compute_calibration_stage_empty_returns_empty_dict() -> None:
    """Empty per_match → empty dict (no division-by-zero)."""
    assert _compute_calibration_stage([]) == {}


def test_compute_calibration_stage_aggregates_per_ronde() -> None:
    """Mean ECE per ronde grouping."""
    matches: list[Any] = [
        _CalibMatch(ronde=5, ece_ali=0.02),
        _CalibMatch(ronde=5, ece_ali=0.04),
        _CalibMatch(ronde=7, ece_ali=0.06),
    ]
    out = _compute_calibration_stage(matches)
    assert "by_ronde" in out
    assert out["by_ronde"]["5"] == pytest.approx(0.03)
    assert out["by_ronde"]["7"] == pytest.approx(0.06)


@dataclass
class _ConfMatch:
    """MatchStats-like with e_score_observed + e_score_predicted for conformal."""

    e_score_observed: float = 4.0
    e_score_predicted: float = 4.2


def test_compute_conformal_stage_returns_required_keys() -> None:
    """Conformal stage returns coverage, set_size_mean, set_size_relative, support_max, quantile_threshold, n_calibration."""
    matches = [_ConfMatch(e_score_observed=4.0 + 0.1 * i, e_score_predicted=4.0) for i in range(60)]
    out = _compute_conformal_stage(matches, support_max=8.0)
    assert set(out.keys()) >= {
        "coverage_global",
        "set_size_mean",
        "set_size_relative",
        "support_max",
        "quantile_threshold",
        "n_calibration",
    }
    assert out["support_max"] == 8.0
    assert 0.0 <= out["coverage_global"] <= 1.0
    assert out["n_calibration"] == 30  # CONFORMAL_CALIB_N


# --- Lineage helpers (env-var / file-IO paths) ---


def test_resolve_code_sha_uses_env_var_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """ISO 5259 lineage : ALICE_CODE_SHA env var has priority."""
    monkeypatch.setenv("ALICE_CODE_SHA", "deadbeef1234567")
    assert _resolve_code_sha() == "deadbeef1234567"


def test_resolve_code_sha_falls_back_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback when env var absent AND no Kaggle CODE_SHA.txt mounted."""
    monkeypatch.delenv("ALICE_CODE_SHA", raising=False)
    # Path("/kaggle/input").is_dir() returns False locally -> fallback "unknown"
    assert _resolve_code_sha() == "unknown"


def test_checkpoint_partial_writes_valid_json(tmp_path: Any) -> None:
    """Incremental save produces parseable d8.v1-partial JSON with required fields."""
    import json

    _checkpoint_partial(
        tmp_path,
        saison=2024,
        div_slug="top-16",
        lineage={"code_sha256": "abc123"},
        n_matches=88,
    )
    out_file = tmp_path / "d8_2024_top-16_partial.json"
    assert out_file.is_file()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "d8.v1-partial"
    assert payload["saison"] == 2024
    assert payload["division_slug"] == "top-16"
    assert payload["n_matches"] == 88
    assert "partial" in payload["_status"]
