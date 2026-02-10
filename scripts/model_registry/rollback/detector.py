"""Detecteur de degradation - ISO 23894.

Fonctions:
- detect_degradation: detection principale
- _find_previous_version: tri par pattern vYYYYMMDD_HHMMSS
- _load_version_metrics: parse metadata.json
- _compare_metrics: comparaison pourcentage

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 5055:2021 - Code Quality (<130 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.model_registry.rollback.types import (
    DegradationThresholds,
    RollbackDecision,
)

logger = logging.getLogger(__name__)

_VERSION_PATTERN = re.compile(r"^v\d{8}_\d{6}$")


def detect_degradation(
    models_dir: Path,
    current_version: str,
    thresholds: DegradationThresholds | None = None,
    drift_result: Any | None = None,
) -> RollbackDecision:
    """Detecte la degradation et recommande un rollback.

    Args:
    ----
        models_dir: Repertoire des modeles versiones
        current_version: Version courante
        thresholds: Seuils de degradation
        drift_result: Resultat du drift monitor (optionnel)

    Returns:
    -------
        RollbackDecision avec should_rollback et raison
    """
    if thresholds is None:
        thresholds = DegradationThresholds()

    timestamp = datetime.now(tz=UTC).isoformat()
    prev = _find_previous_version(models_dir, current_version)

    if prev is None:
        return RollbackDecision(
            should_rollback=False,
            reason="No previous version available",
            current_version=current_version,
            timestamp=timestamp,
        )

    # Check drift first
    if drift_result is not None and _is_drift_critical(drift_result):
        return RollbackDecision(
            should_rollback=True,
            reason=f"Critical drift detected (severity={drift_result.overall_severity.value})",
            current_version=current_version,
            target_version=prev,
            timestamp=timestamp,
        )

    # Compare metrics
    current_metrics = _load_version_metrics(models_dir, current_version)
    previous_metrics = _load_version_metrics(models_dir, prev)

    if current_metrics is None or previous_metrics is None:
        return RollbackDecision(
            should_rollback=False,
            reason="Cannot load metrics for comparison",
            current_version=current_version,
            timestamp=timestamp,
        )

    degraded, reason = _compare_metrics(current_metrics, previous_metrics, thresholds)

    return RollbackDecision(
        should_rollback=degraded,
        reason=reason,
        current_version=current_version,
        target_version=prev if degraded else None,
        metrics_comparison={
            "current": current_metrics,
            "previous": previous_metrics,
        },
        timestamp=timestamp,
    )


def _find_previous_version(
    models_dir: Path,
    current_version: str,
) -> str | None:
    """Trouve la version N-1 par tri de timestamp."""
    versions = sorted(
        d.name
        for d in models_dir.iterdir()
        if d.is_dir() and _VERSION_PATTERN.match(d.name) and d.name != current_version
    )
    if not versions:
        return None
    # Versions triees, la derniere avant current
    earlier = [v for v in versions if v < current_version]
    return earlier[-1] if earlier else None


def _load_version_metrics(
    models_dir: Path,
    version: str,
) -> dict[str, float] | None:
    """Charge les metriques depuis metadata.json."""
    metadata_path = models_dir / version / "metadata.json"
    if not metadata_path.exists():
        return None
    data = json.loads(metadata_path.read_text())
    best_model = data.get("metrics", {}).get("best_model", {})
    if not best_model:
        return None
    return {
        "auc": best_model.get("auc", 0.0),
        "accuracy": best_model.get("accuracy", 0.0),
    }


def _compare_metrics(
    current: dict[str, float],
    previous: dict[str, float],
    thresholds: DegradationThresholds,
) -> tuple[bool, str]:
    """Compare les metriques et detecte la degradation."""
    reasons = []
    cur_auc, prev_auc = current.get("auc", 0), previous.get("auc", 0)
    cur_acc, prev_acc = current.get("accuracy", 0), previous.get("accuracy", 0)

    if prev_auc > 0:
        auc_drop = (prev_auc - cur_auc) / prev_auc * 100
        if auc_drop > thresholds.auc_drop_pct:
            reasons.append(f"AUC drop {auc_drop:.1f}% > threshold {thresholds.auc_drop_pct}%")

    if prev_acc > 0:
        acc_drop = (prev_acc - cur_acc) / prev_acc * 100
        if acc_drop > thresholds.accuracy_drop_pct:
            reasons.append(
                f"Accuracy drop {acc_drop:.1f}% > threshold {thresholds.accuracy_drop_pct}%"
            )

    if reasons:
        return True, "; ".join(reasons)
    return False, "No degradation detected"


def _is_drift_critical(drift_result: Any) -> bool:
    """Verifie si le drift est critique."""
    from scripts.model_registry.drift_types import DriftSeverity  # noqa: PLC0415

    return getattr(drift_result, "overall_severity", None) == DriftSeverity.CRITICAL
