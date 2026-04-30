"""Tests scripts/d8/types — D8 data structures (ISO 29119)."""

from __future__ import annotations

import pytest

from scripts.d8.types import (
    D8GateStatus,
    D8GroupBreakdown,
    D8GroupStats,
    D8Lineage,
    D8SaisonReport,
)


def test_d8_group_stats_frozen() -> None:
    g = D8GroupStats(
        group_key="M",
        n=100,
        recall_mean=0.85,
        jaccard_mean=0.70,
        brier_mean=0.20,
        ece_mean=0.04,
        mae_mean=0.5,
        bss_mean=0.60,
    )
    with pytest.raises(Exception):  # noqa: B017,PT011 - frozen dataclass raises FrozenInstanceError; broad catch intentional
        g.n = 50  # type: ignore[misc]


def test_d8_group_breakdown_indexable() -> None:
    b = D8GroupBreakdown(
        dim_name="gender",
        groups={"M": ..., "F": ...},  # type: ignore[dict-item]
    )
    assert "M" in b.groups
    assert b.dim_name == "gender"


def test_d8_lineage_required_fields() -> None:
    L = D8Lineage(
        joueurs_sha256="a" * 64,
        echiquiers_sha256="b" * 64,
        mlp_artefact_sha256="c" * 64,
        temp_scaler_sha256="d" * 64,
        code_sha256="e" * 7,
        ali_seed=42,
        ali_n_topk=10,
        ali_n_mc_pairs=5,
        ali_decay_lambda=0.9,
        kernel_id="pierrax/d8-saison-2024",
        kernel_version_kaggle="v1",
        run_at_utc="2026-04-30T22:15:33Z",
    )
    assert L.ali_seed == 42


def test_d8_gate_status_enum_values() -> None:
    assert D8GateStatus.PASS.value == "pass"
    assert D8GateStatus.FAIL.value == "fail"
    assert D8GateStatus.INCONCLUSIVE.value == "inconclusive"


def test_d8_saison_report_default_factory() -> None:
    r = D8SaisonReport(
        schema_version="d8.v1",
        saison=2024,
        n_matches=70,
        lineage=D8Lineage(
            joueurs_sha256="a" * 64,
            echiquiers_sha256="b" * 64,
            mlp_artefact_sha256="c" * 64,
            temp_scaler_sha256="d" * 64,
            code_sha256="e" * 7,
            ali_seed=42,
            ali_n_topk=10,
            ali_n_mc_pairs=5,
            ali_decay_lambda=0.9,
            kernel_id="pierrax/d8-saison-2024",
            kernel_version_kaggle="v1",
            run_at_utc="2026-04-30T22:15:33Z",
        ),
    )
    assert r.per_match == []
    assert r.breakdowns == {}
