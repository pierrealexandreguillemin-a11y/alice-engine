"""Tests scripts/d8/aggregate phase-a mode (ADR-019 : 1 saison × 5 divisions).

Document ID: ALICE-D8-TEST-AGGREGATE-PHASE-A
Version: 1.0.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.d8.aggregate import (
    PHASE_A_AUDITS,
    _build_full_report_phase_a,
    _parse_args,
    compute_global_metrics,
    fuse_per_match,
    load_audit_reports,
    verify_lineage_coherence,
)
from scripts.d8.types import D8FullReport


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


def _make_lineage(
    division: str,
    mlp_sha: str = "abc123",
    temp_sha: str = "def456",
    code_sha: str = "code_ccc",
) -> dict[str, Any]:
    return {
        "joueurs_sha256": "joueurs_aaa",
        "echiquiers_sha256": "echiq_bbb",
        "mlp_artefact_sha256": mlp_sha,
        "temp_scaler_sha256": temp_sha,
        "code_sha256": code_sha,
        "ali_seed": 42,
        "ali_n_topk": 10,
        "ali_n_mc_pairs": 5,
        "ali_decay_lambda": 0.9,
        "kernel_id": "k",
        "kernel_version_kaggle": "v1",
        "run_at_utc": "2026-05-14T00:00:00",
        "division": division,
        "saison": 2024,
    }


def _make_audit_report(saison: int, division: str, n_matches: int = 30) -> dict[str, Any]:
    return {
        "schema_version": "d8.v1",
        "saison": saison,
        "n_matches": n_matches,
        "lineage": _make_lineage(division),
        "per_match": [_make_match(saison, r % 9 + 1) for r in range(n_matches)],
        "conformal": {"coverage_global": 0.91, "set_size_mean": 1.2},
    }


# -----------------------------------------------------------------------------
# load_audit_reports : Phase A 5-divisions layout
# -----------------------------------------------------------------------------


def test_load_audit_reports_default_phase_a(tmp_path: Path) -> None:
    for saison, div in PHASE_A_AUDITS:
        d = tmp_path / f"d8-{saison}-{div}"
        d.mkdir()
        (d / f"d8_{saison}_{div}.json").write_text(json.dumps(_make_audit_report(saison, div)))
    out = load_audit_reports(tmp_path)
    assert set(out.keys()) == set(PHASE_A_AUDITS)


def test_load_audit_reports_supports_flat_layout(tmp_path: Path) -> None:
    audits = ((2024, "top-16"), (2024, "nationale-1"))
    for saison, div in audits:
        (tmp_path / f"d8_{saison}_{div}.json").write_text(
            json.dumps(_make_audit_report(saison, div))
        )
    out = load_audit_reports(tmp_path, audits=audits)
    assert set(out.keys()) == set(audits)


def test_load_audit_reports_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"Missing audit"):
        load_audit_reports(tmp_path, audits=((2024, "nope"),))


# -----------------------------------------------------------------------------
# verify_lineage_coherence : works on dict[tuple, dict] (phase-a keys)
# -----------------------------------------------------------------------------


def test_verify_lineage_coherence_accepts_tuple_keyed_dict() -> None:
    """MLP+temp SHA same cross-divisions → no raise even when code_sha differs."""
    reports: dict[tuple[int, str], dict[str, Any]] = {
        (2024, div): _make_audit_report(2024, div) for _, div in PHASE_A_AUDITS
    }
    # Simulate code_sha delta cette session : Top 16 v4 (84d2f6d) vs N1-N4 v3 (11db85f).
    reports[(2024, "top-16")]["lineage"]["code_sha256"] = "84d2f6d_top16"
    reports[(2024, "nationale-1")]["lineage"]["code_sha256"] = "11db85f_n1"
    # MLP + temp SHA identical → pass.
    verify_lineage_coherence(reports)


def test_verify_lineage_coherence_mlp_mismatch_phase_a_raises() -> None:
    reports: dict[tuple[int, str], dict[str, Any]] = {
        (2024, div): _make_audit_report(2024, div) for _, div in PHASE_A_AUDITS
    }
    reports[(2024, "top-16")]["lineage"]["mlp_artefact_sha256"] = "DIFFERENT_MLP"
    with pytest.raises(RuntimeError, match=r"MLP artefact SHA-256 mismatch"):
        verify_lineage_coherence(reports)


# -----------------------------------------------------------------------------
# fuse_per_match : works on dict[tuple, dict]
# -----------------------------------------------------------------------------


def test_fuse_per_match_concatenates_phase_a_divisions() -> None:
    reports: dict[tuple[int, str], dict[str, Any]] = {
        (2024, div): _make_audit_report(2024, div, n_matches=8) for _, div in PHASE_A_AUDITS
    }
    fused = fuse_per_match(reports)
    assert len(fused) == 8 * len(PHASE_A_AUDITS)


# -----------------------------------------------------------------------------
# compute_global_metrics : phase-a aggregation
# -----------------------------------------------------------------------------


def test_compute_global_metrics_phase_a_keys_unchanged() -> None:
    reports: dict[tuple[int, str], dict[str, Any]] = {
        (2024, div): _make_audit_report(2024, div, n_matches=8) for _, div in PHASE_A_AUDITS
    }
    fused = fuse_per_match(reports)
    metrics = compute_global_metrics(fused, reports)
    # 19-gates metric contract preserved.
    assert "max_gap_recall_max_dim" in metrics
    assert "coverage_global" in metrics
    assert "recall_drop_1pct" in metrics


# -----------------------------------------------------------------------------
# _build_full_report_phase_a : audit_mode + divisions populated
# -----------------------------------------------------------------------------


def test_build_full_report_phase_a_populates_new_fields() -> None:
    reports: dict[tuple[int, str], dict[str, Any]] = {
        (2024, div): _make_audit_report(2024, div, n_matches=8) for _, div in PHASE_A_AUDITS
    }
    fused = fuse_per_match(reports)
    metrics = compute_global_metrics(fused, reports)
    # Use empty gates list for unit purposes; main() injects real ones.
    full = _build_full_report_phase_a(reports, fused, metrics, gates_19=[])
    assert isinstance(full, D8FullReport)
    assert full.audit_mode == "phase-a"
    assert full.saisons == [2024]
    assert set(full.divisions) == {div for _, div in PHASE_A_AUDITS}
    assert set(full.lineage_per_saison.keys()) == {2024}


def test_build_full_report_phase_a_lineage_collapses_to_canonical_per_saison() -> None:
    """5 divisions × 1 saison → 1 lineage entry per saison (champion MLP invariant)."""
    reports: dict[tuple[int, str], dict[str, Any]] = {
        (2024, div): _make_audit_report(2024, div, n_matches=8) for _, div in PHASE_A_AUDITS
    }
    fused = fuse_per_match(reports)
    metrics = compute_global_metrics(fused, reports)
    full = _build_full_report_phase_a(reports, fused, metrics, gates_19=[])
    assert len(full.lineage_per_saison) == 1
    assert 2024 in full.lineage_per_saison


# -----------------------------------------------------------------------------
# _parse_args : CLI surface
# -----------------------------------------------------------------------------


def test_parse_args_defaults_saison_mode() -> None:
    args = _parse_args([])
    assert args.mode == "saison"
    assert args.input_dir == Path("/kaggle/input")
    assert args.output_dir == Path("/kaggle/working")


def test_parse_args_phase_a_mode_explicit() -> None:
    args = _parse_args(
        [
            "--mode",
            "phase-a",
            "--input-dir",
            "outputs/d8/2024",
            "--output-dir",
            "reports/d8/phase_a",
        ]
    )
    assert args.mode == "phase-a"
    assert args.input_dir == Path("outputs/d8/2024")
    assert args.output_dir == Path("reports/d8/phase_a")


def test_parse_args_rejects_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--mode", "phase-x"])
