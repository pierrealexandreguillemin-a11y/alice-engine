"""Tests RunnerConfig.max_matches env var override (D-2026-05-11)."""

from __future__ import annotations

import pytest

from scripts.backtest.runner_types import RunnerConfig, _max_matches_default


def test_max_matches_default_no_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ALICE_MAX_MATCHES set, default = 50 preserved."""
    monkeypatch.delenv("ALICE_MAX_MATCHES", raising=False)
    assert _max_matches_default() == 50
    cfg = RunnerConfig()
    assert cfg.max_matches == 50


def test_max_matches_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """ALICE_MAX_MATCHES=150 → RunnerConfig.max_matches=150."""
    monkeypatch.setenv("ALICE_MAX_MATCHES", "150")
    assert _max_matches_default() == 150
    cfg = RunnerConfig()
    assert cfg.max_matches == 150


def test_max_matches_env_var_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """ALICE_MAX_MATCHES=foo → ValueError (ISO 27034 input validation)."""
    monkeypatch.setenv("ALICE_MAX_MATCHES", "foo")
    with pytest.raises(ValueError, match=r"ALICE_MAX_MATCHES"):
        _max_matches_default()


def test_max_matches_env_var_negative_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """ALICE_MAX_MATCHES=0 → ValueError (must be >=1)."""
    monkeypatch.setenv("ALICE_MAX_MATCHES", "0")
    with pytest.raises(ValueError, match=r">=1"):
        _max_matches_default()


def test_max_matches_explicit_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit max_matches kwarg overrides env var (per-test isolation)."""
    monkeypatch.setenv("ALICE_MAX_MATCHES", "150")
    cfg = RunnerConfig(max_matches=42)
    assert cfg.max_matches == 42
