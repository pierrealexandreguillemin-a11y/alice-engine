"""Tests T15 Stratified sampling — Plan 3 V2 Phase 3 (ISO 24027)."""

from __future__ import annotations

import pytest

from scripts.backtest.stratified_sampler import (
    StratifiedSamplerConfig,
    strata_coverage,
    stratified_sample,
    sufficient_strata,
)


def test_stratified_sample_balances_across_strata():
    items = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
    cfg = StratifiedSamplerConfig(min_per_stratum=3, max_per_stratum=3, seed=42)
    out = stratified_sample(items, strata_fn=lambda s: s[0], config=cfg)
    groups = {"A": 0, "B": 0}
    for x in out:
        groups[x[0]] += 1
    assert groups["A"] == 3
    assert groups["B"] == 3


def test_stratified_sample_drops_small_strata():
    items = ["A1", "A2", "B1", "B2", "B3", "B4", "B5"]
    cfg = StratifiedSamplerConfig(min_per_stratum=3, max_per_stratum=5, seed=42)
    out = stratified_sample(items, strata_fn=lambda s: s[0], config=cfg)
    # A has 2 < min=3 -> dropped ; B has 5 -> kept
    assert all(x[0] == "B" for x in out)
    assert len(out) == 5


def test_stratified_sample_reproducible_same_seed():
    items = [f"X{i}" for i in range(20)]
    cfg = StratifiedSamplerConfig(min_per_stratum=2, max_per_stratum=10, seed=42)
    out1 = stratified_sample(items, strata_fn=lambda s: s[0], config=cfg)
    out2 = stratified_sample(items, strata_fn=lambda s: s[0], config=cfg)
    assert out1 == out2


def test_stratified_sample_different_seeds_differ():
    items = [f"X{i}" for i in range(20)]
    cfg42 = StratifiedSamplerConfig(min_per_stratum=2, max_per_stratum=10, seed=42)
    cfg99 = StratifiedSamplerConfig(min_per_stratum=2, max_per_stratum=10, seed=99)
    assert stratified_sample(items, strata_fn=lambda s: s[0], config=cfg42) != stratified_sample(
        items, strata_fn=lambda s: s[0], config=cfg99
    )


def test_stratified_sample_empty():
    cfg = StratifiedSamplerConfig()
    assert stratified_sample([], strata_fn=lambda s: "x", config=cfg) == []


def test_strata_coverage_counts_correctly():
    items = ["A1", "A2", "B1", "B2", "B3"]
    cov = strata_coverage(items, strata_fn=lambda s: s[0])
    assert cov == {"A": 2, "B": 3}


def test_sufficient_strata_filter():
    cov = {"A": 2, "B": 5, "C": 10}
    suf = sufficient_strata(cov, min_per_stratum=3)
    assert suf == {"A": False, "B": True, "C": True}


def test_stratified_sample_max_per_stratum_caps():
    items = [f"A{i}" for i in range(50)]
    cfg = StratifiedSamplerConfig(min_per_stratum=2, max_per_stratum=10, seed=42)
    out = stratified_sample(items, strata_fn=lambda s: "A", config=cfg)
    assert len(out) == 10


def test_stratified_sample_config_frozen():
    cfg = StratifiedSamplerConfig()
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        cfg.seed = 99  # type: ignore[misc]
