"""Tests Drift Report & Status - ISO 29119.

Document ID: ALICE-TEST-MODEL-DRIFT-REPORT
Version: 1.0.0
Tests: 2 classes (TestDriftReport, TestDriftStatus)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)
- ISO/IEC 42001:2023 - AI Management (drift monitoring)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from pathlib import Path

import pandas as pd

from scripts.model_registry import (
    DriftMetrics,
    DriftReport,
    add_round_to_drift_report,
    check_drift_status,
    create_drift_report,
    load_drift_report,
    save_drift_report,
)


class TestDriftReport:
    """Tests pour rapport de drift."""

    def test_create_drift_report(self) -> None:
        """Test creation rapport drift."""
        training_elo = pd.Series([1500, 1600, 1400, 1550, 1650])
        report = create_drift_report(
            season="2025-2026",
            model_version="v20250101",
            training_elo=training_elo,
        )
        assert report.season == "2025-2026"
        assert report.model_version == "v20250101"
        assert len(report.rounds) == 0

    def test_add_round_to_report(self) -> None:
        """Test ajout ronde au rapport."""
        training_elo = pd.Series([1500] * 100)
        report = create_drift_report("2025-2026", "v1", training_elo)
        predictions = pd.DataFrame(
            {
                "predicted_proba": [0.6, 0.4, 0.7, 0.3],
                "elo_blanc": [1500, 1550, 1600, 1450],
                "elo_noir": [1480, 1520, 1580, 1500],
            }
        )
        actuals = pd.Series([1, 0, 1, 0])
        metrics = add_round_to_drift_report(
            report, 1, predictions, actuals, baseline_elo_distribution=training_elo
        )
        assert len(report.rounds) == 1
        assert metrics.round_number == 1

    def test_drift_report_summary(self) -> None:
        """Test resume du rapport."""
        report = DriftReport(
            season="2025-2026",
            model_version="v1",
            baseline_elo_mean=1500,
            baseline_elo_std=150,
            rounds=[
                DriftMetrics(1, "2025-01-01", 100, 0.75, 0.80, 10, 5, 0.05, False, False, []),
                DriftMetrics(2, "2025-02-01", 100, 0.73, 0.78, 15, 8, 0.08, False, False, []),
                DriftMetrics(3, "2025-03-01", 100, 0.70, 0.75, 20, 10, 0.12, True, False, ["W"]),
            ],
        )
        summary = report.get_summary()
        assert summary["rounds_monitored"] == 3
        assert summary["alerts"]["warnings"] == 1

    def test_save_load_drift_report(self, tmp_path: Path) -> None:
        """Test sauvegarde/chargement rapport."""
        report = DriftReport(
            season="2025-2026",
            model_version="v1",
            baseline_elo_mean=1500,
            baseline_elo_std=150,
            rounds=[
                DriftMetrics(1, "2025-01-01", 100, 0.75, 0.80, 10, 5, 0.05, False, False, []),
            ],
        )
        report_path = tmp_path / "drift_report.json"
        save_drift_report(report, report_path)
        loaded = load_drift_report(report_path)
        assert loaded is not None
        assert loaded.season == report.season

    def test_load_missing_report(self, tmp_path: Path) -> None:
        """Test chargement rapport inexistant."""
        result = load_drift_report(tmp_path / "nonexistent.json")
        assert result is None


class TestDriftStatus:
    """Tests pour verification statut drift."""

    def test_status_ok(self) -> None:
        """Test statut OK."""
        report = DriftReport(
            season="2025-2026",
            model_version="v1",
            baseline_elo_mean=1500,
            baseline_elo_std=150,
            rounds=[
                DriftMetrics(1, "t", 100, 0.75, 0.80, 10, 5, 0.05, False, False, []),
                DriftMetrics(2, "t", 100, 0.74, 0.79, 12, 6, 0.06, False, False, []),
            ],
        )
        status = check_drift_status(report)
        assert status["status"] == "OK"

    def test_status_monitor_closely(self) -> None:
        """Test statut degradation legere."""
        report = DriftReport(
            season="2025-2026",
            model_version="v1",
            baseline_elo_mean=1500,
            baseline_elo_std=150,
            rounds=[
                DriftMetrics(1, "t", 100, 0.80, 0.85, 10, 5, 0.05, False, False, []),
                DriftMetrics(2, "t", 100, 0.74, 0.78, 15, 8, 0.08, False, False, []),
            ],
        )
        status = check_drift_status(report)
        assert status["status"] == "MONITOR_CLOSELY"

    def test_status_retrain_recommended(self) -> None:
        """Test statut retraining recommande."""
        report = DriftReport(
            season="2025-2026",
            model_version="v1",
            baseline_elo_mean=1500,
            baseline_elo_std=150,
            rounds=[
                DriftMetrics(1, "t", 100, 0.75, 0.80, 10, 5, 0.30, True, True, ["C"]),
            ],
        )
        status = check_drift_status(report)
        assert status["status"] == "RETRAIN_RECOMMENDED"

    def test_status_no_data(self) -> None:
        """Test statut sans donnees."""
        report = DriftReport(
            season="2025-2026",
            model_version="v1",
            baseline_elo_mean=1500,
            baseline_elo_std=150,
            rounds=[],
        )
        status = check_drift_status(report)
        assert status["status"] == "NO_DATA"
