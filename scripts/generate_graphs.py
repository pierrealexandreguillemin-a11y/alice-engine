#!/usr/bin/env python3
"""Module: generate_graphs.py - Graphs Architecture.

Generation de graphs architecture et qualite.

ISO Compliance:
- ISO/IEC 42010 - Architecture (dependency graphs)
- ISO/IEC 25010 - System Quality (visualisation)
- ISO/IEC 5055 - Code Quality (complexity reports)

Genere:
- graphs/dependencies.svg (pydeps)
- graphs/imports.svg (imports structure)
- reports/complexity/ (radon)
- reports/duplication/ (pylint)

Voir scripts/graphs/ pour l'implementation.

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

import sys
from datetime import datetime

from scripts.graphs import (
    check_circular_imports,
    check_duplication,
    generate_complexity_report,
    generate_dependency_graph,
    generate_imports_graph,
    print_header,
)
from scripts.graphs.utils import GRAPHS_DIR, REPORTS_DIR, ROOT


def main() -> int:
    """Run graph generation pipeline.

    Returns
    -------
        0 si succes, 1 sinon
    """
    print_header("GENERATION GRAPHS ARCHITECTURE & QUALITE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Repertoire: {ROOT}")

    # Create directories
    GRAPHS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    results = []

    results.append(("Dependencies Graph", generate_dependency_graph()))
    results.append(("Imports Graph", generate_imports_graph()))
    results.append(("Complexity Report", generate_complexity_report()))
    results.append(("Duplication Check", check_duplication()))
    results.append(("Circular Imports", check_circular_imports()))

    # Summary
    print_header("RESUME")

    success = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "OK" if ok else "!"
        print(f"  [{status}] {name}")

    print(f"\nScore: {success}/{total}")

    if success == total:
        print("\nTous les graphs generes avec succes!")
        return 0
    else:
        print("\nCertains graphs n'ont pas pu etre generes.")
        print("Verifier les dependances: pip install pydeps radon pylint")
        return 1


if __name__ == "__main__":
    sys.exit(main())
