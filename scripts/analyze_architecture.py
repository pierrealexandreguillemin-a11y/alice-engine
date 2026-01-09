#!/usr/bin/env python3
"""Module: analyze_architecture.py - Analyse Architecture.

Detection spaghetti code et sante architecture.

ISO Compliance:
- ISO/IEC 42010 - Architecture (analyse dependances)
- ISO/IEC 25010 - System Quality (maintenabilite)
- ISO/IEC 5055 - Code Quality (coupling, cohesion)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from __future__ import annotations

import sys
from datetime import datetime

from scripts.architecture import (
    analyze_dependencies,
    calculate_coupling,
    calculate_health_score,
    detect_circular_imports,
    print_report,
    save_report,
)


def print_header(title: str) -> None:
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> int:
    """Run architecture health analysis."""
    print_header("ANALYSE SANTE ARCHITECTURE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Analyze dependencies
    deps = analyze_dependencies()
    files = list(deps.keys())

    if not files:
        print("\n! Aucun fichier Python trouve dans app/ ou services/")
        return 1

    # Calculate metrics
    coupling_data = calculate_coupling(deps)
    circular = detect_circular_imports(deps)

    total_deps = sum(len(d) for d in deps.values())
    avg_deps = total_deps / len(files) if files else 0
    max_deps = max(len(d) for d in deps.values()) if deps else 0

    # Calculate health score
    score, issues = calculate_health_score(coupling_data, circular, avg_deps, max_deps)

    # Print and save report
    print_report(coupling_data, circular, files, total_deps, avg_deps, max_deps, score, issues)
    save_report(coupling_data, circular, files, total_deps, avg_deps, max_deps, score, issues)

    return 0 if score >= 60 else 1


if __name__ == "__main__":
    sys.exit(main())
