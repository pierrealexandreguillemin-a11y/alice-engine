#!/usr/bin/env python3
"""Module: audit_iso_conformity.py - Audit ISO Conformite.

Verification automatique de la conformite aux normes ISO.

ISO Compliance:
- ISO/IEC 5055 - Code Quality (maintenabilite, < 300 lignes)
- ISO/IEC 29119 - Software Testing (couverture > 80%)
- ISO/IEC 15289 - Documentation Lifecycle (structure)
- ISO/IEC 5259:2024 - Data Quality for ML (tracabilite)
- ISO/IEC 25010 - System Quality (SRP)
- ISO/IEC 42001:2023 - AI Management (audit trail)

Usage:
    python scripts/audit_iso_conformity.py
    python scripts/audit_iso_conformity.py --strict
    python scripts/audit_iso_conformity.py --fix

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.iso_audit import (
    AuditResult,
    audit_iso_5055_maintainability,
    audit_iso_5259_data_quality,
    audit_iso_15289_documentation,
    audit_iso_25010_quality,
    audit_iso_29119_testing,
    generate_report,
)

PROJECT_ROOT = Path(__file__).parent.parent


def run_audit(
    output_dir: Path, *, generate_reports: bool = False, quick: bool = False
) -> AuditResult:
    """Exécute l'audit complet."""
    result = AuditResult()

    audit_iso_5055_maintainability(result, PROJECT_ROOT)
    if not quick:
        audit_iso_29119_testing(result, PROJECT_ROOT)
    else:
        print("\n[ISO 29119] Skipped (--quick mode)")
    audit_iso_15289_documentation(result, PROJECT_ROOT)
    audit_iso_5259_data_quality(result, PROJECT_ROOT)
    audit_iso_25010_quality(result, PROJECT_ROOT)

    # Déterminer conformité
    critical_count = sum(1 for v in result.violations if v.severity == "CRITICAL")
    high_count = sum(1 for v in result.violations if v.severity == "HIGH")
    result.compliant = critical_count == 0 and high_count == 0

    if generate_reports:
        generate_report(result, output_dir)

    return result


def print_summary(result: AuditResult) -> None:
    """Affiche le résumé de l'audit."""
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)

    critical = sum(1 for v in result.violations if v.severity == "CRITICAL")
    high = sum(1 for v in result.violations if v.severity == "HIGH")
    medium = sum(1 for v in result.violations if v.severity == "MEDIUM")
    low = sum(1 for v in result.violations if v.severity == "LOW")

    print(f"Violations CRITICAL: {critical}")
    print(f"Violations HIGH: {high}")
    print(f"Violations MEDIUM: {medium}")
    print(f"Violations LOW: {low}")

    status = "[OK] CONFORME" if result.compliant else "[FAIL] NON CONFORME"
    print(f"\nStatut: {status}")


def main() -> int:
    """Point d'entrée."""
    parser = argparse.ArgumentParser(description="Audit ISO conformité ALICE Engine")
    parser.add_argument("--strict", action="store_true", help="Échoue si non conforme")
    parser.add_argument("--fix", action="store_true", help="Génère rapport correctif")
    parser.add_argument("--quick", action="store_true", help="Skip tests (audit statique)")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "reports", help="Répertoire sortie"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AUDIT ISO CONFORMITÉ - ALICE ENGINE")
    if args.quick:
        print("(Mode rapide - tests skippés)")
    print("=" * 60)

    result = run_audit(args.output, generate_reports=args.fix, quick=args.quick)
    print_summary(result)

    if args.strict and not result.compliant:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
