"""Executeur de rollback - ISO 23894.

Fonctions:
- execute_rollback: execute le rollback via versioning.py
- log_rollback_event: append JSONL historique

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 5055:2021 - Code Quality (<95 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from scripts.model_registry.rollback.types import RollbackDecision, RollbackResult
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

    if dry_run:
        result = RollbackResult(
            success=True,
            rolled_back_from=decision.current_version,
            rolled_back_to=decision.target_version or "",
            reason=f"DRY RUN: {decision.reason}",
            timestamp=timestamp,
        )
        logger.info(
            "DRY RUN: would rollback %s -> %s", decision.current_version, decision.target_version
        )
        return result

    if not decision.target_version:
        return RollbackResult(
            success=False,
            rolled_back_from=decision.current_version,
            rolled_back_to="",
            reason=decision.reason,
            timestamp=timestamp,
            error_message="No target version specified",
        )

    success = rollback_to_version(models_dir, decision.target_version)

    result = RollbackResult(
        success=success,
        rolled_back_from=decision.current_version,
        rolled_back_to=decision.target_version,
        reason=decision.reason,
        timestamp=timestamp,
        error_message=None if success else f"Failed to rollback to {decision.target_version}",
    )

    if success:
        log_rollback_event(models_dir, result)
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
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
