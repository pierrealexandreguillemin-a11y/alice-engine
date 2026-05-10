"""Tests scripts/d8/aggregate — fuse 4 saisons + global gates + render markdown.

Document ID: ALICE-D8-TEST-AGGREGATE
Version: 1.0.0
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.d8.aggregate import (
    DEFAULT_SAISONS,
    PERTURBATION_GATES_INCONCLUSIVE,
    compute_global_metrics,
    evaluate_19_with_inconclusive,
    fuse_per_match,
    load_saison_reports,
    render_findings_md,
    verify_lineage_coherence,
)
from scripts.d8.types import D8FullReport, D8GateStatus


def _make_match(saison: int, ronde: int, recall: float = 0.92) -> dict[str, Any]:
    return {
        "saison": saison,
        "ronde": ronde,
        "user_team": f"T{saison}_U",
        "opponent_team": f"T{saison}_O",
        "recall_ali": recall,
        "accuracy_ali": recall * 0.9,
        "jaccard_ali": recall * 0.85,
        "brier_ali": 0.15,
        "ece_ali": 0.03,
        "recall_baseline": recall * 0.85,
        "brier_baseline": 0.20,
        "bss": 0.40,
        "e_score_predicted": 4.0,
        "e_score_observed": 4.1,
        "e_score_mae": 0.10,
        "ali_correct": True,
        "baseline_correct": False,
    }


def _make_lineage(mlp_sha: str = "abc123", temp_sha: str = "def456") -> dict[str, Any]:
    return {
        "joueurs_sha256": "joueurs_aaa",
        "echiquiers_sha256": "echiq_bbb",
        "mlp_artefact_sha256": mlp_sha,
        "temp_scaler_sha256": temp_sha,
        "code_sha256": "code_ccc",
        "ali_seed": 42,
        "ali_n_topk": 10,
        "ali_n_mc_pairs": 5,
        "ali_decay_lambda": 0.9,
        "kernel_id": "k",
        "kernel_version_kaggle": "v1",
        "run_at_utc": "2026-05-09T00:00:00",
    }


def _make_saison_report(saison: int, n_matches: int = 70) -> dict[str, Any]:
    return {
        "schema_version": "d8.v1",
        "saison": saison,
        "n_matches": n_matches,
        "lineage": _make_lineage(),
        "per_match": [_make_match(saison, r % 9 + 1) for r in range(n_matches)],
        "conformal": {"coverage_global": 0.92, "set_size_mean": 1.5},
    }


# -----------------------------------------------------------------------------
# load_saison_reports
# -----------------------------------------------------------------------------


def test_load_reads_default_saisons(tmp_path: Path) -> None:
    for s in DEFAULT_SAISONS:
        d = tmp_path / f"d8-saison-{s}"
        d.mkdir()
        (d / f"d8_saison_{s}.json").write_text(json.dumps(_make_saison_report(s)))
    out = load_saison_reports(tmp_path)
    assert set(out.keys()) == set(DEFAULT_SAISONS)
    assert all(out[s]["saison"] == s for s in DEFAULT_SAISONS)


def test_load_supports_flat_layout(tmp_path: Path) -> None:
    """Flat fallback layout : input_dir/d8_saison_{S}.json (test convenience)."""
    saisons = (2023, 2024)
    for s in saisons:
        (tmp_path / f"d8_saison_{s}.json").write_text(json.dumps(_make_saison_report(s)))
    out = load_saison_reports(tmp_path, saisons=saisons)
    assert set(out.keys()) == set(saisons)


def test_load_missing_saison_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"Missing saison"):
        load_saison_reports(tmp_path, saisons=(2099,))


# -----------------------------------------------------------------------------
# verify_lineage_coherence
# -----------------------------------------------------------------------------


def test_lineage_coherence_passes_when_identical() -> None:
    reports = {s: _make_saison_report(s) for s in DEFAULT_SAISONS}
    verify_lineage_coherence(reports)


def test_lineage_coherence_mlp_mismatch_raises() -> None:
    reports = {s: _make_saison_report(s) for s in DEFAULT_SAISONS}
    reports[2024]["lineage"]["mlp_artefact_sha256"] = "DIFFERENT"
    with pytest.raises(RuntimeError, match=r"MLP artefact SHA-256 mismatch"):
        verify_lineage_coherence(reports)


def test_lineage_coherence_temp_scaler_mismatch_raises() -> None:
    reports = {s: _make_saison_report(s) for s in DEFAULT_SAISONS}
    reports[2024]["lineage"]["temp_scaler_sha256"] = "DIFFERENT"
    with pytest.raises(RuntimeError, match=r"temp_scaler SHA-256 mismatch"):
        verify_lineage_coherence(reports)


# -----------------------------------------------------------------------------
# fuse_per_match
# -----------------------------------------------------------------------------


def test_fuse_concatenates_all_saisons() -> None:
    reports = {s: _make_saison_report(s, n_matches=5) for s in DEFAULT_SAISONS}
    fused = fuse_per_match(reports)
    assert len(fused) == 5 * len(DEFAULT_SAISONS)
    seen_saisons = {m["saison"] for m in fused}
    assert seen_saisons == set(DEFAULT_SAISONS)


def test_fuse_empty_reports_returns_empty() -> None:
    assert fuse_per_match({}) == []


# -----------------------------------------------------------------------------
# compute_global_metrics
# -----------------------------------------------------------------------------


def test_compute_global_metrics_returns_19_keys() -> None:
    reports = {s: _make_saison_report(s, n_matches=10) for s in DEFAULT_SAISONS}
    fused = fuse_per_match(reports)
    metrics = compute_global_metrics(fused)
    expected = {
        "max_gap_recall_max_dim",
        "recall_per_group_min",
        "demographic_parity_diff",
        "equalized_odds_diff",
        "ece_per_group_max",
        "multicalibration_alpha",
        "tpr_ratio_min",
        "brier_per_group_max",
        "bss_per_group_min",
        "psi_per_dim_max",
        "recall_drop_1pct",
        "recall_drop_5pct",
        "recall_drop_10pct",
        "roster_5pct_recall_drop",
        "roster_20pct_recall_drop",
        "coverage_global",
        "conformal_set_size_mean",
        "dro_eps_005_recall_worst",
        "dro_eps_010_recall_worst",
    }
    assert set(metrics.keys()) == expected


def test_compute_global_metrics_empty_raises() -> None:
    with pytest.raises(ValueError, match=r"non-empty"):
        compute_global_metrics([])


def test_compute_global_metrics_perturbation_gates_are_nan() -> None:
    import math

    reports = {s: _make_saison_report(s, n_matches=10) for s in DEFAULT_SAISONS}
    fused = fuse_per_match(reports)
    metrics = compute_global_metrics(fused)
    for key in (
        "recall_drop_1pct",
        "recall_drop_5pct",
        "recall_drop_10pct",
        "roster_5pct_recall_drop",
        "roster_20pct_recall_drop",
        "dro_eps_005_recall_worst",
        "dro_eps_010_recall_worst",
    ):
        assert math.isnan(metrics[key]), f"{key} should be NaN pending perturbation infra"


# -----------------------------------------------------------------------------
# evaluate_19_with_inconclusive
# -----------------------------------------------------------------------------


def test_evaluate_19_returns_19_evaluations() -> None:
    reports = {s: _make_saison_report(s, n_matches=10) for s in DEFAULT_SAISONS}
    metrics = compute_global_metrics(fuse_per_match(reports))
    out = evaluate_19_with_inconclusive(metrics)
    assert len(out) == 19


def test_perturbation_gates_marked_inconclusive() -> None:
    reports = {s: _make_saison_report(s, n_matches=10) for s in DEFAULT_SAISONS}
    metrics = compute_global_metrics(fuse_per_match(reports))
    out = evaluate_19_with_inconclusive(metrics)
    inconclusive_ids = {g.gate_id for g in out if g.status is D8GateStatus.INCONCLUSIVE}
    assert inconclusive_ids == set(PERTURBATION_GATES_INCONCLUSIVE)


def test_non_perturbation_gates_evaluated_normally() -> None:
    reports = {s: _make_saison_report(s, n_matches=10) for s in DEFAULT_SAISONS}
    metrics = compute_global_metrics(fuse_per_match(reports))
    out = evaluate_19_with_inconclusive(metrics)
    non_perturbation = [g for g in out if g.gate_id not in PERTURBATION_GATES_INCONCLUSIVE]
    assert all(g.status in (D8GateStatus.PASS, D8GateStatus.FAIL) for g in non_perturbation)


# -----------------------------------------------------------------------------
# render_findings_md
# -----------------------------------------------------------------------------


def test_render_findings_blocks_phase4a_when_inconclusive() -> None:
    reports = {s: _make_saison_report(s, n_matches=10) for s in DEFAULT_SAISONS}
    metrics = compute_global_metrics(fuse_per_match(reports))
    gates = evaluate_19_with_inconclusive(metrics)
    summary = {
        "pass": sum(1 for g in gates if g.status is D8GateStatus.PASS),
        "fail": sum(1 for g in gates if g.status is D8GateStatus.FAIL),
        "inconclusive": sum(1 for g in gates if g.status is D8GateStatus.INCONCLUSIVE),
    }
    report = D8FullReport(
        schema_version="d8-aggregator.v1",
        n_matches=40,
        saisons=list(DEFAULT_SAISONS),
        lineage_per_saison={s: _make_lineage() for s in DEFAULT_SAISONS},
        breakdowns_global={},
        multicalibration_global={},
        stress_elo_global={},
        stress_roster_global={},
        conformal_global={},
        dro_global={},
        gates_19=gates,
    )
    md = render_findings_md(report, datetime(2026, 5, 9, tzinfo=UTC), summary=summary)
    assert "Phase 3.5 STRICT" in md
    assert f"{summary['pass']}/19 PASS" in md
    if summary["inconclusive"] > 0:
        assert "INCONCLUSIVE" in md
        assert "BLOCKED" in md or "perturbation closures pending" in md


def test_render_findings_lists_each_gate_id() -> None:
    reports = {s: _make_saison_report(s, n_matches=10) for s in DEFAULT_SAISONS}
    metrics = compute_global_metrics(fuse_per_match(reports))
    gates = evaluate_19_with_inconclusive(metrics)
    report = D8FullReport(
        schema_version="d8-aggregator.v1",
        n_matches=40,
        saisons=list(DEFAULT_SAISONS),
        lineage_per_saison={s: _make_lineage() for s in DEFAULT_SAISONS},
        breakdowns_global={},
        multicalibration_global={},
        stress_elo_global={},
        stress_roster_global={},
        conformal_global={},
        dro_global={},
        gates_19=gates,
    )
    md = render_findings_md(
        report,
        datetime(2026, 5, 9, tzinfo=UTC),
        summary={"pass": 12, "fail": 0, "inconclusive": 7},
    )
    for g in gates:
        assert g.gate_id in md


# -----------------------------------------------------------------------------
# integration : full pipeline on stub saison reports
# -----------------------------------------------------------------------------


def test_pipeline_smoke_4_saisons(tmp_path: Path) -> None:
    """Smoke pipeline : 4 saisons stub → metrics → gates → render."""
    for s in DEFAULT_SAISONS:
        (tmp_path / f"d8_saison_{s}.json").write_text(
            json.dumps(_make_saison_report(s, n_matches=15))
        )
    reports = load_saison_reports(tmp_path)
    verify_lineage_coherence(reports)
    fused = fuse_per_match(reports)
    assert len(fused) == 60
    metrics = compute_global_metrics(fused)
    gates = evaluate_19_with_inconclusive(metrics)
    assert len(gates) == 19
    inconclusive = sum(1 for g in gates if g.status is D8GateStatus.INCONCLUSIVE)
    assert inconclusive == len(PERTURBATION_GATES_INCONCLUSIVE)
