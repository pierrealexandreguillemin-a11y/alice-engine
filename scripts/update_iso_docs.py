#!/usr/bin/env python3
"""Module: update_iso_docs.py - ISO Documentation Generator.

Mise a jour documentation ISO automatique.

ISO Compliance:
- ISO/IEC 15289 - Documentation Lifecycle (quality records)
- ISO/IEC 25010 - System Quality (documentation)
- ISO/IEC 5055 - Code Quality (metriques)

Genere: docs/iso/IMPLEMENTATION_STATUS.md
Trace conformite aux normes ISO applicables.

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.iso_docs import (
    MODULES,
    check_installed,
    generate_report,
    get_complexity_avg,
    get_test_coverage,
)

# Re-exports for backward compatibility
__all__ = [
    "MODULES",
    "check_installed",
    "generate_report",
    "get_complexity_avg",
    "get_test_coverage",
    "main",
]

ROOT = Path(__file__).parent.parent
DOCS_ISO = ROOT / "docs" / "iso"


def main() -> int:
    """Run ISO documentation update."""
    print("Mise a jour documentation ISO...")

    DOCS_ISO.mkdir(parents=True, exist_ok=True)

    report = generate_report()
    output = DOCS_ISO / "IMPLEMENTATION_STATUS.md"
    output.write_text(report, encoding="utf-8")

    print(f"OK Documentation ISO mise a jour: {output}")

    # Count installed
    installed = sum(1 for m in MODULES if check_installed(m["check"])[0])
    total = len(MODULES)
    score = round((installed / total) * 100)

    print(f"\nScore DevOps: {score}/100")
    print(f"Modules installes: {installed}/{total}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
