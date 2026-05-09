"""Tests scripts/d8/gates — 19 gates G-A SOTA strict + case-by-case logger.

Document ID: ALICE-D8-TEST-GATES
Version: 1.0.0
"""

from __future__ import annotations

import math

import pytest

from scripts.d8.gates import (
    MAX_THRESHOLD_GATES,
    MIN_THRESHOLD_GATES,
    SOURCES,
    THRESHOLDS,
    evaluate_19_gates,
    evaluate_inconclusive,
    evaluate_max_threshold,
    evaluate_min_threshold,
    filter_failures,
    gates_summary,
    render_failure_analysis_md,
)
from scripts.d8.types import D8GateEvaluation, D8GateStatus

# -----------------------------------------------------------------------------
# Threshold registries integrity
# -----------------------------------------------------------------------------


def test_19_thresholds_registered() -> None:
    assert len(THRESHOLDS) == 19


def test_each_gate_has_a_source() -> None:
    assert set(THRESHOLDS.keys()) == set(SOURCES.keys())


def test_max_min_partitions_cover_all_gates() -> None:
    """Every gate must be classified as MAX or MIN exactly once."""
    union = MAX_THRESHOLD_GATES | MIN_THRESHOLD_GATES
    assert union == set(THRESHOLDS.keys())
    assert MAX_THRESHOLD_GATES.isdisjoint(MIN_THRESHOLD_GATES)


def test_max_partition_size() -> None:
    assert len(MAX_THRESHOLD_GATES) == 13


def test_min_partition_size() -> None:
    assert len(MIN_THRESHOLD_GATES) == 6


# -----------------------------------------------------------------------------
# evaluate_max_threshold
# -----------------------------------------------------------------------------


def test_max_threshold_pass_when_equal() -> None:
    out = evaluate_max_threshold("G_FAIR_01_max_gap_recall", 0.10)
    assert out.status is D8GateStatus.PASS


def test_max_threshold_pass_when_below() -> None:
    out = evaluate_max_threshold("G_FAIR_01_max_gap_recall", 0.05)
    assert out.status is D8GateStatus.PASS


def test_max_threshold_fail_when_above() -> None:
    out = evaluate_max_threshold("G_FAIR_01_max_gap_recall", 0.15)
    assert out.status is D8GateStatus.FAIL


def test_max_threshold_carries_measured_and_threshold() -> None:
    out = evaluate_max_threshold("G_FAIR_05_calibration_ECE_per_group", 0.04)
    assert out.measured_value == pytest.approx(0.04)
    assert out.threshold == pytest.approx(0.05)


def test_max_threshold_carries_source() -> None:
    out = evaluate_max_threshold("G_FAIR_01_max_gap_recall", 0.05)
    assert out.source == SOURCES["G_FAIR_01_max_gap_recall"]


def test_max_threshold_unknown_gate_raises() -> None:
    with pytest.raises(ValueError, match=r"Unknown gate"):
        evaluate_max_threshold("G_FAIR_99_unknown", 0.1)


def test_max_threshold_nan_raises() -> None:
    with pytest.raises(ValueError, match=r"finite"):
        evaluate_max_threshold("G_FAIR_01_max_gap_recall", float("nan"))


def test_max_threshold_inf_raises() -> None:
    with pytest.raises(ValueError, match=r"finite"):
        evaluate_max_threshold("G_FAIR_01_max_gap_recall", float("inf"))


def test_max_threshold_custom_override() -> None:
    out = evaluate_max_threshold("G_FAIR_01_max_gap_recall", 0.18, threshold=0.20)
    assert out.threshold == pytest.approx(0.20)
    assert out.status is D8GateStatus.PASS


# -----------------------------------------------------------------------------
# evaluate_min_threshold
# -----------------------------------------------------------------------------


def test_min_threshold_pass_when_equal() -> None:
    out = evaluate_min_threshold("G_FAIR_02_recall_per_group_min", 0.85)
    assert out.status is D8GateStatus.PASS


def test_min_threshold_pass_when_above() -> None:
    out = evaluate_min_threshold("G_FAIR_02_recall_per_group_min", 0.95)
    assert out.status is D8GateStatus.PASS


def test_min_threshold_fail_when_below() -> None:
    out = evaluate_min_threshold("G_FAIR_02_recall_per_group_min", 0.80)
    assert out.status is D8GateStatus.FAIL


def test_min_threshold_unknown_gate_raises() -> None:
    with pytest.raises(ValueError, match=r"Unknown gate"):
        evaluate_min_threshold("G_FAIR_99_unknown", 0.9)


def test_min_threshold_nan_raises() -> None:
    with pytest.raises(ValueError, match=r"finite"):
        evaluate_min_threshold("G_ROB_06_conformal_coverage_90", float("nan"))


# -----------------------------------------------------------------------------
# evaluate_inconclusive
# -----------------------------------------------------------------------------


def test_inconclusive_status() -> None:
    out = evaluate_inconclusive("G_ROB_08_DRO_eps_005_min")
    assert out.status is D8GateStatus.INCONCLUSIVE


def test_inconclusive_measured_is_nan() -> None:
    out = evaluate_inconclusive("G_ROB_08_DRO_eps_005_min")
    assert math.isnan(out.measured_value)


def test_inconclusive_threshold_default() -> None:
    out = evaluate_inconclusive("G_ROB_08_DRO_eps_005_min")
    assert out.threshold == pytest.approx(THRESHOLDS["G_ROB_08_DRO_eps_005_min"])


def test_inconclusive_unknown_gate_raises() -> None:
    with pytest.raises(ValueError, match=r"Unknown gate"):
        evaluate_inconclusive("G_FOO")


# -----------------------------------------------------------------------------
# evaluate_19_gates — happy path + per-gate pass/fail
# -----------------------------------------------------------------------------


def _all_pass_metrics() -> dict[str, float]:
    """Return a metrics dict where every gate passes."""
    return {
        "max_gap_recall_max_dim": 0.05,
        "recall_per_group_min": 0.95,
        "demographic_parity_diff": 0.05,
        "equalized_odds_diff": 0.05,
        "ece_per_group_max": 0.02,
        "multicalibration_alpha": 0.02,
        "tpr_ratio_min": 0.95,
        "brier_per_group_max": 0.20,
        "bss_per_group_min": 0.40,
        "psi_per_dim_max": 0.10,
        "recall_drop_1pct": 0.01,
        "recall_drop_5pct": 0.02,
        "recall_drop_10pct": 0.05,
        "roster_5pct_recall_drop": 0.02,
        "roster_20pct_recall_drop": 0.10,
        "coverage_global": 0.92,
        "conformal_set_size_mean": 2.0,
        "dro_eps_005_recall_worst": 0.80,
        "dro_eps_010_recall_worst": 0.65,
    }


def test_evaluate_19_gates_returns_19() -> None:
    out = evaluate_19_gates(_all_pass_metrics())
    assert len(out) == 19


def test_evaluate_19_gates_all_pass_when_metrics_are_clean() -> None:
    out = evaluate_19_gates(_all_pass_metrics())
    assert all(e.status is D8GateStatus.PASS for e in out)


def test_evaluate_19_gates_one_fail_propagates() -> None:
    metrics = _all_pass_metrics()
    metrics["max_gap_recall_max_dim"] = 0.50  # fail
    out = evaluate_19_gates(metrics)
    failed = [e for e in out if e.status is D8GateStatus.FAIL]
    assert len(failed) == 1
    assert failed[0].gate_id == "G_FAIR_01_max_gap_recall"


def test_evaluate_19_gates_multiple_fails() -> None:
    metrics = _all_pass_metrics()
    metrics["recall_per_group_min"] = 0.50
    metrics["coverage_global"] = 0.20
    metrics["dro_eps_005_recall_worst"] = 0.10
    out = evaluate_19_gates(metrics)
    failed_ids = {e.gate_id for e in out if e.status is D8GateStatus.FAIL}
    assert failed_ids == {
        "G_FAIR_02_recall_per_group_min",
        "G_ROB_06_conformal_coverage_90",
        "G_ROB_08_DRO_eps_005_min",
    }


def test_evaluate_19_gates_missing_key_raises() -> None:
    metrics = _all_pass_metrics()
    del metrics["coverage_global"]
    with pytest.raises(KeyError, match=r"coverage_global"):
        evaluate_19_gates(metrics)


def test_evaluate_19_gates_each_gate_id_present_exactly_once() -> None:
    out = evaluate_19_gates(_all_pass_metrics())
    ids = [e.gate_id for e in out]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(THRESHOLDS.keys())


# -----------------------------------------------------------------------------
# Per-gate pass/fail (parametrized 19 × 2 = 38 cases)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("gate_id", sorted(MAX_THRESHOLD_GATES))
def test_max_threshold_pass_for_each_gate(gate_id: str) -> None:
    """Every MAX gate passes when measured = threshold (boundary)."""
    out = evaluate_max_threshold(gate_id, THRESHOLDS[gate_id])
    assert out.status is D8GateStatus.PASS


@pytest.mark.parametrize("gate_id", sorted(MAX_THRESHOLD_GATES))
def test_max_threshold_fail_for_each_gate(gate_id: str) -> None:
    """Every MAX gate fails when measured = threshold * 2."""
    out = evaluate_max_threshold(gate_id, THRESHOLDS[gate_id] * 2.0 + 0.01)
    assert out.status is D8GateStatus.FAIL


@pytest.mark.parametrize("gate_id", sorted(MIN_THRESHOLD_GATES))
def test_min_threshold_pass_for_each_gate(gate_id: str) -> None:
    """Every MIN gate passes when measured = threshold (boundary)."""
    out = evaluate_min_threshold(gate_id, THRESHOLDS[gate_id])
    assert out.status is D8GateStatus.PASS


@pytest.mark.parametrize("gate_id", sorted(MIN_THRESHOLD_GATES))
def test_min_threshold_fail_for_each_gate(gate_id: str) -> None:
    """Every MIN gate fails when measured is half the threshold."""
    out = evaluate_min_threshold(gate_id, THRESHOLDS[gate_id] / 2.0)
    assert out.status is D8GateStatus.FAIL


# -----------------------------------------------------------------------------
# render_failure_analysis_md
# -----------------------------------------------------------------------------


def test_render_md_no_failures_returns_clean_message() -> None:
    md = render_failure_analysis_md([])
    assert "All 19 gates PASS" in md
    assert "FAIL" not in md.split("\n")[0]


def test_render_md_emits_heading_per_failure() -> None:
    failures = [
        D8GateEvaluation(
            gate_id="G_FAIR_01_max_gap_recall",
            threshold=0.10,
            measured_value=0.18,
            status=D8GateStatus.FAIL,
            source=SOURCES["G_FAIR_01_max_gap_recall"],
        ),
        D8GateEvaluation(
            gate_id="G_ROB_06_conformal_coverage_90",
            threshold=0.90,
            measured_value=0.78,
            status=D8GateStatus.FAIL,
            source=SOURCES["G_ROB_06_conformal_coverage_90"],
        ),
    ]
    md = render_failure_analysis_md(failures, today="2026-05-09")
    assert "G_FAIR_01_max_gap_recall FAIL" in md
    assert "G_ROB_06_conformal_coverage_90 FAIL" in md
    assert "2026-05-09" in md


def test_render_md_includes_delta_and_source() -> None:
    f = D8GateEvaluation(
        gate_id="G_FAIR_01_max_gap_recall",
        threshold=0.10,
        measured_value=0.18,
        status=D8GateStatus.FAIL,
        source=SOURCES["G_FAIR_01_max_gap_recall"],
    )
    md = render_failure_analysis_md([f], today="2026-05-09")
    assert "+0.0800" in md  # delta = 0.08
    assert SOURCES["G_FAIR_01_max_gap_recall"] in md


def test_render_md_includes_decision_template() -> None:
    f = D8GateEvaluation(
        gate_id="G_FAIR_01_max_gap_recall",
        threshold=0.10,
        measured_value=0.18,
        status=D8GateStatus.FAIL,
        source=SOURCES["G_FAIR_01_max_gap_recall"],
    )
    md = render_failure_analysis_md([f])
    assert "Décision user" in md
    assert "Bloquer Phase 4a" in md


def test_render_md_default_today_uses_iso_format() -> None:
    f = D8GateEvaluation(
        gate_id="G_FAIR_01_max_gap_recall",
        threshold=0.10,
        measured_value=0.18,
        status=D8GateStatus.FAIL,
        source=SOURCES["G_FAIR_01_max_gap_recall"],
    )
    md = render_failure_analysis_md([f])  # no `today` provided
    # Match YYYY-MM-DD literal — defaults to UTC today
    import re as _re

    assert _re.search(r"analysis \d{4}-\d{2}-\d{2}", md) is not None


# -----------------------------------------------------------------------------
# filter_failures + gates_summary
# -----------------------------------------------------------------------------


def test_filter_failures_excludes_pass_and_inconclusive() -> None:
    evaluations = [
        evaluate_max_threshold("G_FAIR_01_max_gap_recall", 0.05),  # pass
        evaluate_max_threshold("G_FAIR_03_demographic_parity_diff", 0.50),  # fail
        evaluate_inconclusive("G_ROB_08_DRO_eps_005_min"),
    ]
    failures = filter_failures(evaluations)
    assert len(failures) == 1
    assert failures[0].gate_id == "G_FAIR_03_demographic_parity_diff"


def test_gates_summary_counts() -> None:
    evaluations = [
        evaluate_max_threshold("G_FAIR_01_max_gap_recall", 0.05),  # pass
        evaluate_max_threshold("G_FAIR_03_demographic_parity_diff", 0.50),  # fail
        evaluate_max_threshold("G_FAIR_05_calibration_ECE_per_group", 0.10),  # fail
        evaluate_inconclusive("G_ROB_08_DRO_eps_005_min"),
    ]
    summary = gates_summary(evaluations)
    assert summary == {"pass": 1, "fail": 2, "inconclusive": 1}


def test_gates_summary_empty_list() -> None:
    assert gates_summary([]) == {"pass": 0, "fail": 0, "inconclusive": 0}
