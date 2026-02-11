"""Tests Rollback Executor - ISO 29119.

Document ID: ALICE-TEST-ROLLBACK-EXECUTOR
Version: 1.0.0
Tests count: 8

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 23894:2023 - AI Risk Management

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.model_registry.rollback.executor import (
    execute_rollback,
    log_rollback_event,
)
from scripts.model_registry.rollback.types import RollbackDecision, RollbackResult


def _make_decision(rollback: bool = True) -> RollbackDecision:
    """Helper: cree une decision de rollback."""
    return RollbackDecision(
        should_rollback=rollback,
        reason="AUC drop 3.5% > threshold 2.0%",
        current_version="v20260115_120000",
        target_version="v20260101_120000",
        timestamp="2026-02-10T12:00:00",
    )


class TestExecuteRollback:
    """Tests pour l'execution du rollback."""

    def test_calls_versioning_rollback(self, models_dir: Path) -> None:
        """Appelle rollback_to_version du registre."""
        decision = _make_decision()
        result = execute_rollback(models_dir, decision)
        assert result.success is True
        assert result.rolled_back_to == "v20260101_120000"

    def test_returns_result_on_success(self, models_dir: Path) -> None:
        """Retourne un RollbackResult en cas de succes."""
        decision = _make_decision()
        result = execute_rollback(models_dir, decision)
        assert isinstance(result, RollbackResult)
        assert result.rolled_back_from == "v20260115_120000"

    def test_dry_run_does_not_modify(self, models_dir: Path) -> None:
        """Dry run ne modifie rien."""
        decision = _make_decision()
        result = execute_rollback(models_dir, decision, dry_run=True)
        assert result.success is True
        assert result.reason.startswith("DRY RUN")

    def test_logs_rollback_event(self, models_dir: Path) -> None:
        """L'evenement de rollback est logge."""
        decision = _make_decision()
        result = execute_rollback(models_dir, decision)
        history_path = models_dir / "rollback_history.jsonl"
        assert history_path.exists()

    def test_handles_rollback_failure(self, tmp_path: Path) -> None:
        """Gere l'echec du rollback gracieusement."""
        decision = RollbackDecision(
            should_rollback=True,
            reason="test failure",
            current_version="v20260101_120000",
            target_version="v20251201_120000",
        )
        result = execute_rollback(tmp_path, decision)
        assert result.success is False
        assert result.error_message is not None


class TestLogRollbackEvent:
    """Tests pour le logging des evenements."""

    def test_creates_history_file(self, tmp_path: Path) -> None:
        """Cree le fichier historique s'il n'existe pas."""
        result = RollbackResult(
            success=True,
            rolled_back_from="v2",
            rolled_back_to="v1",
            reason="test",
            timestamp="2026-02-10T12:00:00",
        )
        log_rollback_event(tmp_path, result)
        assert (tmp_path / "rollback_history.jsonl").exists()

    def test_appends_to_existing_history(self, tmp_path: Path) -> None:
        """Ajoute a l'historique existant."""
        result = RollbackResult(
            success=True,
            rolled_back_from="v2",
            rolled_back_to="v1",
            reason="test",
            timestamp="2026-02-10T12:00:00",
        )
        log_rollback_event(tmp_path, result)
        log_rollback_event(tmp_path, result)
        lines = (tmp_path / "rollback_history.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_entry_contains_timestamp(self, tmp_path: Path) -> None:
        """L'entree contient le timestamp."""
        result = RollbackResult(
            success=True,
            rolled_back_from="v2",
            rolled_back_to="v1",
            reason="test",
            timestamp="2026-02-10T12:00:00",
        )
        log_rollback_event(tmp_path, result)
        line = (tmp_path / "rollback_history.jsonl").read_text().strip()
        entry = json.loads(line)
        assert "timestamp" in entry
