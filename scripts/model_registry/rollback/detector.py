"""Detecteur de degradation - ISO 23894.

Fonctions:
- detect_degradation: detection principale
- _find_previous_version: tri par pattern vYYYYMMDD_HHMMSS
- _load_version_metrics: parse metadata.json (avec error handling)
- _compare_metrics: comparaison pourcentage

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 5055:2021 - Code Quality (SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-11
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.model_registry.drift_types import DriftSeverity
from scripts.model_registry.rollback.types import (
    DegradationThresholds,
    RollbackDecision,
    validate_version_format,
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

    if not validate_version_format(current_version):
        msg = f"Invalid version format: {current_version!r}"
        raise ValueError(msg)

    try:
        prev = _find_previous_version(models_dir, current_version)
    except FileNotFoundError:
        logger.warning("Models directory not found")
        return _no_rollback(current_version, "Models directory not found", timestamp)

    if prev is None:
        return _no_rollback(current_version, "No previous version available", timestamp)

    if drift_result is not None and _is_drift_critical(drift_result):
        return RollbackDecision(
            should_rollback=True,
            reason=f"Critical drift detected (severity={drift_result.overall_severity.value})",
            current_version=current_version,
            target_version=prev,
            timestamp=timestamp,
        )

    return _check_metric_degradation(
        models_dir,
        current_version,
        prev,
        thresholds,
        timestamp,
    )


def _no_rollback(current_version: str, reason: str, timestamp: str) -> RollbackDecision:
    """Construit une decision sans rollback."""
    return RollbackDecision(
        should_rollback=False,
        reason=reason,
        current_version=current_version,
        timestamp=timestamp,
    )


def _check_metric_degradation(
    models_dir: Path,
    current_version: str,
    prev: str,
    thresholds: DegradationThresholds,
    timestamp: str,
) -> RollbackDecision:
    """Charge et compare les metriques entre versions."""
    current_metrics = _load_version_metrics(models_dir, current_version)
    previous_metrics = _load_version_metrics(models_dir, prev)

    if current_metrics is None or previous_metrics is None:
        logger.warning(
            "Cannot load metrics: current=%s previous=%s",
            current_metrics is not None,
            previous_metrics is not None,
        )
        return _no_rollback(current_version, "Cannot load metrics for comparison", timestamp)

    degraded, reason = _compare_metrics(current_metrics, previous_metrics, thresholds)

    return RollbackDecision(
        should_rollback=degraded,
        reason=reason,
        current_version=current_version,
        target_version=prev if degraded else None,
        metrics_comparison={"current": current_metrics, "previous": previous_metrics},
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
    """Charge les metriques depuis metadata.json (avec error handling)."""
    metadata_path = models_dir / version / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        data = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to parse %s: %s", metadata_path, e)
        return None
    best_model = data.get("metrics", {}).get("best_model", {})
    if not best_model:
        logger.warning("No best_model metrics in %s", metadata_path)
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
    return getattr(drift_result, "overall_severity", None) == DriftSeverity.CRITICAL
