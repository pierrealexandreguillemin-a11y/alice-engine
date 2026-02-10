"""Fixtures Model Rollback - ISO 29119.

Document ID: ALICE-TEST-ROLLBACK-CONFTEST
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.model_registry.drift_types import DriftMonitorResult, DriftSeverity


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    """Repertoire de modeles avec 2 versions."""
    v1 = tmp_path / "v20260101_120000"
    v1.mkdir()
    (v1 / "metadata.json").write_text(
        json.dumps(
            {
                "version": "v20260101_120000",
                "metrics": {
                    "best_model": {"auc": 0.85, "accuracy": 0.80},
                },
                "created_at": "2026-01-01T12:00:00",
            }
        )
    )

    v2 = tmp_path / "v20260115_120000"
    v2.mkdir()
    (v2 / "metadata.json").write_text(
        json.dumps(
            {
                "version": "v20260115_120000",
                "metrics": {
                    "best_model": {"auc": 0.82, "accuracy": 0.76},
                },
                "created_at": "2026-01-15T12:00:00",
            }
        )
    )

    # current symlink (directory for portability)
    current = tmp_path / "current"
    current.mkdir()
    (current / "metadata.json").write_text(
        json.dumps(
            {
                "version": "v20260115_120000",
            }
        )
    )

    return tmp_path


@pytest.fixture
def drift_critical() -> DriftMonitorResult:
    """Drift result avec severite CRITICAL."""
    return DriftMonitorResult(
        timestamp="2026-02-10T12:00:00",
        model_version="v20260115_120000",
        baseline_samples=1000,
        current_samples=1000,
        overall_severity=DriftSeverity.CRITICAL,
        drift_detected=True,
    )


@pytest.fixture
def drift_none() -> DriftMonitorResult:
    """Drift result sans drift."""
    return DriftMonitorResult(
        timestamp="2026-02-10T12:00:00",
        model_version="v20260115_120000",
        baseline_samples=1000,
        current_samples=1000,
        overall_severity=DriftSeverity.NONE,
        drift_detected=False,
    )
