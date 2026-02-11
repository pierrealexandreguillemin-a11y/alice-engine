"""Executeur de rollback - ISO 23894.

Fonctions:
- execute_rollback: execute le rollback via versioning.py (avec safety)
- log_rollback_event: append JSONL historique (avec error handling)

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 5055:2021 - Code Quality (SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-11
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from scripts.model_registry.rollback.types import (
    RollbackDecision,
    RollbackResult,
    validate_version_format,
)
from scripts.model_registry.versioning import rollback_to_version

logger = logging.getLogger(__name__)


def execute_rollback(
    models_dir: Path,
    decision: RollbackDecision,
    *,
    dry_run: bool = False,
) -> RollbackResult:
    """Execute le rollback vers la version cible.

    Args:
    ----
        models_dir: Repertoire des modeles
        decision: Decision de rollback
        dry_run: Si True, ne modifie rien

    Returns:
    -------
        RollbackResult avec success et details
    """
    timestamp = datetime.now(tz=UTC).isoformat()

    if decision.target_version and not validate_version_format(decision.target_version):
        msg = f"Invalid target version format: {decision.target_version!r}"
        raise ValueError(msg)

    if dry_run:
        return _execute_dry_run(models_dir, decision, timestamp)

    if not decision.target_version:
        return RollbackResult(
            success=False,
            rolled_back_from=decision.current_version,
            rolled_back_to="",
            reason=decision.reason,
            timestamp=timestamp,
            error_message="No target version specified",
        )

    return _execute_real_rollback(models_dir, decision, timestamp)


def _execute_dry_run(
    models_dir: Path,
    decision: RollbackDecision,
    timestamp: str,
) -> RollbackResult:
    """Execute un rollback en mode dry run (verification seule)."""
    target_exists = (
        (models_dir / decision.target_version).exists() if decision.target_version else False
    )
    logger.info(
        "DRY RUN: would rollback %s -> %s (exists=%s)",
        decision.current_version,
        decision.target_version,
        target_exists,
    )
    return RollbackResult(
        success=target_exists,
        rolled_back_from=decision.current_version,
        rolled_back_to=decision.target_version or "",
        reason=f"DRY RUN: {decision.reason}",
        timestamp=timestamp,
        error_message=None if target_exists else "Target version not found",
    )


def _execute_real_rollback(
    models_dir: Path,
    decision: RollbackDecision,
    timestamp: str,
) -> RollbackResult:
    """Execute le rollback reel via versioning.py."""
    try:
        success = rollback_to_version(models_dir, decision.target_version)
    except Exception as e:
        logger.exception(
            "Rollback raised exception: %s -> %s",
            decision.current_version,
            decision.target_version,
        )
        return RollbackResult(
            success=False,
            rolled_back_from=decision.current_version,
            rolled_back_to=decision.target_version,
            reason=decision.reason,
            timestamp=timestamp,
            error_message=str(e),
        )

    result = RollbackResult(
        success=success,
        rolled_back_from=decision.current_version,
        rolled_back_to=decision.target_version,
        reason=decision.reason,
        timestamp=timestamp,
        error_message=None if success else f"Failed to rollback to {decision.target_version}",
    )
    log_rollback_event(models_dir, result)
    if success:
        logger.info("Rolled back: %s -> %s", decision.current_version, decision.target_version)
    else:
        logger.error("Rollback failed: %s -> %s", decision.current_version, decision.target_version)
    return result


def log_rollback_event(models_dir: Path, result: RollbackResult) -> None:
    """Enregistre l'evenement de rollback dans l'historique JSONL."""
    history_path = models_dir / "rollback_history.jsonl"
    entry = {
        "timestamp": result.timestamp,
        "success": result.success,
        "rolled_back_from": result.rolled_back_from,
        "rolled_back_to": result.rolled_back_to,
        "reason": result.reason,
        "error_message": result.error_message,
    }
    try:
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.error("Failed to write rollback history: %s", e)
