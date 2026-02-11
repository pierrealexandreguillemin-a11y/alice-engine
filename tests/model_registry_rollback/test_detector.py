"""Tests Rollback Detector - ISO 29119.

Document ID: ALICE-TEST-ROLLBACK-DETECTOR
Version: 1.1.0
Tests count: 20

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 23894:2023 - AI Risk Management

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.model_registry.drift_types import DriftMonitorResult

from scripts.model_registry.rollback.detector import (
    _compare_metrics,
    _find_previous_version,
    _load_version_metrics,
    detect_degradation,
)
from scripts.model_registry.rollback.types import DegradationThresholds


class TestFindPreviousVersion:
    """Tests pour trouver la version N-1."""

    def test_finds_n_minus_1_by_timestamp(self, models_dir: Path) -> None:
        """Trouve la version precedente par tri timestamp."""
        prev = _find_previous_version(models_dir, "v20260115_120000")
        assert prev == "v20260101_120000"

    def test_returns_none_if_only_one_version(self, tmp_path: Path) -> None:
        """Retourne None s'il n'y a qu'une seule version."""
        v1 = tmp_path / "v20260101_120000"
        v1.mkdir()
        (v1 / "metadata.json").write_text("{}")
        prev = _find_previous_version(tmp_path, "v20260101_120000")
        assert prev is None

    def test_ignores_current_symlink(self, models_dir: Path) -> None:
        """Ignore le dossier 'current'."""
        prev = _find_previous_version(models_dir, "v20260115_120000")
        assert prev != "current"


class TestLoadVersionMetrics:
    """Tests pour le chargement des metriques."""

    def test_loads_from_metadata_json(self, models_dir: Path) -> None:
        """Charge les metriques depuis metadata.json."""
        metrics = _load_version_metrics(models_dir, "v20260101_120000")
        assert metrics is not None
        assert "auc" in metrics

    def test_returns_none_if_no_metadata(self, tmp_path: Path) -> None:
        """Retourne None si pas de metadata.json."""
        v1 = tmp_path / "v20260101_120000"
        v1.mkdir()
        metrics = _load_version_metrics(tmp_path, "v20260101_120000")
        assert metrics is None

    def test_parses_best_model_auc(self, models_dir: Path) -> None:
        """Parse correctement l'AUC du best_model."""
        metrics = _load_version_metrics(models_dir, "v20260101_120000")
        assert metrics is not None
        assert metrics["auc"] == 0.85


class TestCompareMetrics:
    """Tests pour la comparaison des metriques."""

    def test_auc_drop_exceeds_threshold(self) -> None:
        """AUC drop > seuil detecte."""
        current = {"auc": 0.80, "accuracy": 0.80}
        previous = {"auc": 0.85, "accuracy": 0.80}
        thresholds = DegradationThresholds(auc_drop_pct=2.0)
        degraded, reason = _compare_metrics(current, previous, thresholds)
        assert degraded is True
        assert "AUC" in reason

    def test_accuracy_drop_exceeds_threshold(self) -> None:
        """Accuracy drop > seuil detecte."""
        current = {"auc": 0.85, "accuracy": 0.75}
        previous = {"auc": 0.85, "accuracy": 0.80}
        thresholds = DegradationThresholds(accuracy_drop_pct=3.0)
        degraded, reason = _compare_metrics(current, previous, thresholds)
        assert degraded is True
        assert "Accuracy" in reason

    def test_no_degradation_stable_metrics(self) -> None:
        """Pas de degradation si metriques stables."""
        current = {"auc": 0.85, "accuracy": 0.80}
        previous = {"auc": 0.85, "accuracy": 0.80}
        thresholds = DegradationThresholds()
        degraded, _ = _compare_metrics(current, previous, thresholds)
        assert degraded is False

    def test_improvement_not_flagged(self) -> None:
        """Amelioration pas flaggee comme degradation."""
        current = {"auc": 0.90, "accuracy": 0.85}
        previous = {"auc": 0.85, "accuracy": 0.80}
        thresholds = DegradationThresholds()
        degraded, _ = _compare_metrics(current, previous, thresholds)
        assert degraded is False

    def test_both_metrics_degraded_simultaneously(self) -> None:
        """AUC et accuracy degradees en meme temps reportent les deux."""
        current = {"auc": 0.80, "accuracy": 0.73}
        previous = {"auc": 0.85, "accuracy": 0.80}
        thresholds = DegradationThresholds(auc_drop_pct=2.0, accuracy_drop_pct=3.0)
        degraded, reason = _compare_metrics(current, previous, thresholds)
        assert degraded is True
        assert "AUC" in reason
        assert "Accuracy" in reason

    def test_previous_zero_auc_no_crash(self) -> None:
        """Previous AUC = 0 ne cause pas de division par zero."""
        current = {"auc": 0.85, "accuracy": 0.80}
        previous = {"auc": 0.0, "accuracy": 0.80}
        thresholds = DegradationThresholds()
        degraded, _ = _compare_metrics(current, previous, thresholds)
        assert degraded is False

    def test_missing_metric_keys_use_zero_default(self) -> None:
        """Cles metriques manquantes utilisent 0 par defaut (accuracy 0.8->0 = 100% drop)."""
        current = {"auc": 0.85}  # Missing accuracy -> defaults to 0
        previous = {"accuracy": 0.80}  # Missing auc -> prev_auc=0, skips check
        thresholds = DegradationThresholds()
        degraded, reason = _compare_metrics(current, previous, thresholds)
        assert degraded is True
        assert "Accuracy" in reason

    def test_drift_object_without_severity_attribute(self, models_dir: Path) -> None:
        """Drift object sans overall_severity ne crash pas."""

        class FakeDrift:
            pass

        decision = detect_degradation(
            models_dir,
            "v20260115_120000",
            drift_result=FakeDrift(),
        )
        # Should fall through to metric comparison (not crash on drift check)
        assert isinstance(decision.should_rollback, bool)

    def test_drift_none_does_not_trigger_rollback(self, tmp_path: Path) -> None:
        """drift_result=None ne declenche pas de rollback via drift."""
        _create_version(tmp_path, "v20260101_120000", 0.85, 0.80)
        _create_version(tmp_path, "v20260115_120000", 0.85, 0.80)
        decision = detect_degradation(tmp_path, "v20260115_120000", drift_result=None)
        assert decision.should_rollback is False


class TestDetectDegradation:
    """Tests d'integration pour la detection de degradation."""

    def test_returns_rollback_on_auc_drop(self, models_dir: Path) -> None:
        """Recommande rollback si AUC drop."""
        decision = detect_degradation(
            models_dir,
            "v20260115_120000",
        )
        # v2 AUC=0.82, v1 AUC=0.85 -> drop ~3.5% > 2%
        assert decision.should_rollback is True

    def test_returns_no_rollback_on_stable(self, tmp_path: Path) -> None:
        """Pas de rollback si metriques stables."""
        _create_version(tmp_path, "v20260101_120000", 0.85, 0.80)
        _create_version(tmp_path, "v20260115_120000", 0.85, 0.80)
        decision = detect_degradation(tmp_path, "v20260115_120000")
        assert decision.should_rollback is False

    def test_drift_critical_triggers_rollback(
        self,
        models_dir: Path,
        drift_critical: DriftMonitorResult,
    ) -> None:
        """Drift CRITICAL declenche rollback."""
        decision = detect_degradation(
            models_dir,
            "v20260115_120000",
            drift_result=drift_critical,
        )
        assert decision.should_rollback is True
        assert "drift" in decision.reason.lower()

    def test_custom_thresholds_respected(self, models_dir: Path) -> None:
        """Seuils custom sont respectes."""
        # Avec seuil AUC tres haut, le drop de 3.5% n'est pas flagge
        thresholds = DegradationThresholds(auc_drop_pct=10.0, accuracy_drop_pct=10.0)
        decision = detect_degradation(
            models_dir,
            "v20260115_120000",
            thresholds=thresholds,
        )
        assert decision.should_rollback is False

    def test_no_previous_version_no_rollback(self, tmp_path: Path) -> None:
        """Pas de version precedente -> pas de rollback."""
        _create_version(tmp_path, "v20260101_120000", 0.85, 0.80)
        decision = detect_degradation(tmp_path, "v20260101_120000")
        assert decision.should_rollback is False


def _create_version(
    base: Path,
    name: str,
    auc: float,
    accuracy: float,
) -> None:
    """Helper: cree une version avec metadata."""
    d = base / name
    d.mkdir(exist_ok=True)
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "version": name,
                "metrics": {"best_model": {"auc": auc, "accuracy": accuracy}},
            }
        )
    )
