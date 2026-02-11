"""Tests Error Paths - Rollback - ISO 29119.

Document ID: ALICE-TEST-ROLLBACK-ERROR-PATHS
Version: 1.1.0
Tests count: 17

Covers:
- Corrupted metadata.json
- Missing models directory
- No target version in decision
- Rollback exception handling
- JSONL write failure
- Pydantic validators on thresholds

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (error paths)
- ISO/IEC 23894:2023 - AI Risk Management

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.model_registry.rollback.detector import (
    _load_version_metrics,
    detect_degradation,
)
from scripts.model_registry.rollback.executor import execute_rollback, log_rollback_event
from scripts.model_registry.rollback.types import (
    DegradationThresholds,
    RollbackDecision,
    RollbackResult,
)


class TestCorruptedMetadata:
    """Tests pour les metadonnees corrompues."""

    def test_corrupted_json_returns_none(self, tmp_path: Path) -> None:
        """metadata.json corrompu retourne None."""
        v = tmp_path / "v20260101_120000"
        v.mkdir()
        (v / "metadata.json").write_text("{invalid json!!!")
        result = _load_version_metrics(tmp_path, "v20260101_120000")
        assert result is None

    def test_missing_best_model_returns_none(self, tmp_path: Path) -> None:
        """metadata.json sans best_model retourne None."""
        v = tmp_path / "v20260101_120000"
        v.mkdir()
        (v / "metadata.json").write_text(json.dumps({"metrics": {}}))
        result = _load_version_metrics(tmp_path, "v20260101_120000")
        assert result is None

    def test_empty_json_returns_none(self, tmp_path: Path) -> None:
        """metadata.json vide retourne None."""
        v = tmp_path / "v20260101_120000"
        v.mkdir()
        (v / "metadata.json").write_text("{}")
        result = _load_version_metrics(tmp_path, "v20260101_120000")
        assert result is None


class TestMissingModelsDir:
    """Tests pour le repertoire de modeles manquant."""

    def test_nonexistent_dir_no_rollback(self, tmp_path: Path) -> None:
        """Repertoire inexistant ne recommande pas de rollback."""
        missing = tmp_path / "nonexistent"
        decision = detect_degradation(missing, "v20260101_120000")
        assert decision.should_rollback is False
        assert "not found" in decision.reason.lower()


class TestExecuteRollbackErrors:
    """Tests pour les erreurs d'execution du rollback."""

    def test_no_target_version_returns_failure(self, tmp_path: Path) -> None:
        """Decision sans target_version retourne echec."""
        decision = RollbackDecision(
            should_rollback=True,
            reason="test",
            current_version="v1",
            target_version=None,
        )
        result = execute_rollback(tmp_path, decision)
        assert result.success is False
        assert "No target version" in result.error_message

    def test_dry_run_nonexistent_target_fails(self, tmp_path: Path) -> None:
        """Dry run avec target inexistant indique l'echec."""
        decision = RollbackDecision(
            should_rollback=True,
            reason="test",
            current_version="v20260101_120000",
            target_version="v20251201_120000",
        )
        result = execute_rollback(tmp_path, decision, dry_run=True)
        assert result.success is False
        assert result.error_message is not None


class TestLogRollbackEventErrors:
    """Tests pour les erreurs de logging JSONL."""

    def test_readonly_dir_does_not_crash(self, tmp_path: Path) -> None:
        """Ecriture dans un dossier en lecture seule ne crash pas."""
        result = RollbackResult(
            success=True,
            rolled_back_from="v2",
            rolled_back_to="v1",
            reason="test",
            timestamp="2026-02-10T12:00:00",
        )
        # Use a path that can't be written to
        fake_dir = tmp_path / "nonexistent_subdir" / "deep"
        # This should not raise - the function catches OSError
        log_rollback_event(fake_dir, result)


class TestThresholdsValidation:
    """Tests pour la validation Pydantic des seuils."""

    def test_default_thresholds_valid(self) -> None:
        """Les seuils par defaut sont valides."""
        t = DegradationThresholds()
        assert t.auc_drop_pct == 2.0
        assert t.accuracy_drop_pct == 3.0

    def test_rejects_negative_auc_threshold(self) -> None:
        """Seuil AUC negatif est rejete."""
        with pytest.raises(ValidationError):
            DegradationThresholds(auc_drop_pct=-1.0)

    def test_rejects_auc_threshold_above_100(self) -> None:
        """Seuil AUC > 100% est rejete."""
        with pytest.raises(ValidationError):
            DegradationThresholds(auc_drop_pct=100.1)

    def test_accepts_auc_threshold_exactly_100(self) -> None:
        """Seuil AUC = 100% est accepte (boundary)."""
        t = DegradationThresholds(auc_drop_pct=100.0)
        assert t.auc_drop_pct == 100.0

    def test_rejects_negative_accuracy_threshold(self) -> None:
        """Seuil accuracy negatif est rejete."""
        with pytest.raises(ValidationError):
            DegradationThresholds(accuracy_drop_pct=-1.0)

    def test_rejects_accuracy_threshold_above_100(self) -> None:
        """Seuil accuracy > 100% est rejete."""
        with pytest.raises(ValidationError):
            DegradationThresholds(accuracy_drop_pct=100.1)


class TestVersionFormatValidation:
    """Tests pour la validation du format de version (ISO 27034)."""

    def test_rejects_path_traversal_in_version(self, tmp_path: Path) -> None:
        """Version avec path traversal est rejetee."""
        with pytest.raises(ValueError, match="Invalid version format"):
            detect_degradation(tmp_path, "../../etc/passwd")

    def test_rejects_absolute_path_in_version(self, tmp_path: Path) -> None:
        """Version avec chemin absolu est rejetee."""
        with pytest.raises(ValueError, match="Invalid version format"):
            detect_degradation(tmp_path, "/etc/passwd")

    def test_rejects_invalid_format(self, tmp_path: Path) -> None:
        """Version au format invalide est rejetee."""
        with pytest.raises(ValueError, match="Invalid version format"):
            detect_degradation(tmp_path, "v_nonexistent")

    def test_rejects_invalid_target_in_executor(self, tmp_path: Path) -> None:
        """Target version invalide dans executor est rejetee."""
        decision = RollbackDecision(
            should_rollback=True,
            reason="test",
            current_version="v20260101_120000",
            target_version="../../etc/passwd",
        )
        with pytest.raises(ValueError, match="Invalid target version"):
            execute_rollback(tmp_path, decision)
