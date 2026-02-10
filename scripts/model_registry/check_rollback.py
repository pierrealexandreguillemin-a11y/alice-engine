"""Runner CLI pour check rollback - ISO 23894.

Usage:
    python -m scripts.model_registry.check_rollback
    python -m scripts.model_registry.check_rollback --dry-run
    python -m scripts.model_registry.check_rollback --models-dir ./models

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 5055:2021 - Code Quality (<50 lignes)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts.model_registry.rollback import detect_degradation, execute_rollback
from scripts.model_registry.versioning import get_current_version

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Verifie la degradation et execute rollback si necessaire."""
    parser = argparse.ArgumentParser(description="Model Rollback Check (ISO 23894)")
    parser.add_argument("--models-dir", type=Path, default=Path("models/production"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    current = get_current_version(args.models_dir)
    if not current:
        logger.warning("No current version found")
        return

    decision = detect_degradation(args.models_dir, current)
    logger.info("Decision: %s", json.dumps(decision.to_dict(), indent=2))

    if decision.should_rollback:
        result = execute_rollback(args.models_dir, decision, dry_run=args.dry_run)
        logger.info("Result: success=%s", result.success)


if __name__ == "__main__":
    main()
