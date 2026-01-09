"""Auditeurs ISO - ISO 5055.

Fonctions d'audit pour chaque norme ISO.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - subprocess used for internal dev tools only
import sys
from pathlib import Path

from scripts.iso_audit.types import AuditResult

# Configuration
MAX_FILE_LINES = 300  # ISO 5055
MIN_COVERAGE = 80  # ISO 29119
EXCLUDED_DIRS = {".venv", ".git", "__pycache__", ".mypy_cache", "node_modules"}
EXCLUDED_FILES = {"__init__.py", "conftest.py"}


def count_lines(file_path: Path) -> int:
    """Compte les lignes d'un fichier."""
    try:
        return len(file_path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return 0


def audit_iso_5055_maintainability(result: AuditResult, project_root: Path) -> None:
    """ISO 5055: Vérifie maintenabilité (taille fichiers)."""
    print("\n[ISO 5055] Audit maintenabilité code...")

    python_files = []
    for py_file in project_root.rglob("*.py"):
        if any(excl in py_file.parts for excl in EXCLUDED_DIRS):
            continue
        if "test_" in py_file.name:
            continue
        if py_file.name in EXCLUDED_FILES:
            continue

        lines = count_lines(py_file)
        rel_path = py_file.relative_to(project_root)
        python_files.append((str(rel_path), lines))

        _check_file_size(result, rel_path, lines)

    python_files.sort(key=lambda x: -x[1])
    result.stats["iso_5055"] = {
        "total_files": len(python_files),
        "compliant_files": sum(1 for _, lines in python_files if lines <= MAX_FILE_LINES),
        "largest_files": python_files[:10],
    }

    compliant = result.stats["iso_5055"]["compliant_files"]
    total = result.stats["iso_5055"]["total_files"]
    print(f"  Fichiers conformes: {compliant}/{total}")


def _check_file_size(result: AuditResult, rel_path: Path, lines: int) -> None:
    """Vérifie la taille d'un fichier."""
    if lines > MAX_FILE_LINES * 3:  # > 900 = CRITICAL
        result.add_violation(
            norm="ISO 5055",
            severity="CRITICAL",
            file=str(rel_path),
            message=f"{lines} lignes (max: {MAX_FILE_LINES})",
            fix=f"Refactorer en {lines // MAX_FILE_LINES + 1} modules",
        )
    elif lines > MAX_FILE_LINES * 2:  # > 600 = HIGH
        result.add_violation(
            norm="ISO 5055",
            severity="HIGH",
            file=str(rel_path),
            message=f"{lines} lignes (max: {MAX_FILE_LINES})",
            fix=f"Refactorer en {lines // MAX_FILE_LINES + 1} modules",
        )
    elif lines > MAX_FILE_LINES:  # > 300 = MEDIUM
        result.add_violation(
            norm="ISO 5055",
            severity="MEDIUM",
            file=str(rel_path),
            message=f"{lines} lignes (max: {MAX_FILE_LINES})",
            fix="Extraire fonctions dans module séparé",
        )


def audit_iso_29119_testing(result: AuditResult, project_root: Path) -> None:
    """ISO 29119: Vérifie couverture tests."""
    print("\n[ISO 29119] Audit couverture tests...")

    try:
        subprocess.run(  # nosec B603, B607 - trusted pytest command for internal audit
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=scripts",
                "--cov=services",
                "--cov=app",
                "--cov-report=json",
                "-q",
                "--tb=no",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,
        )
        _process_coverage_report(result, project_root)
    except subprocess.TimeoutExpired:
        result.add_violation(
            norm="ISO 29119",
            severity="HIGH",
            file="tests/",
            message="Timeout exécution tests (>5min)",
            fix="Optimiser tests ou paralléliser",
        )
    except Exception as e:
        result.add_violation(
            norm="ISO 29119",
            severity="HIGH",
            file="tests/",
            message=f"Erreur exécution tests: {e}",
            fix="Vérifier configuration pytest",
        )


def _process_coverage_report(result: AuditResult, project_root: Path) -> None:
    """Traite le rapport de couverture."""
    cov_file = project_root / "coverage.json"
    if not cov_file.exists():
        return

    cov_data = json.loads(cov_file.read_text(encoding="utf-8"))
    total_coverage = cov_data.get("totals", {}).get("percent_covered", 0)

    files_coverage = []
    for file_path, file_data in cov_data.get("files", {}).items():
        pct = file_data.get("summary", {}).get("percent_covered", 0)
        try:
            rel_path = Path(file_path).relative_to(project_root)
        except ValueError:
            rel_path = Path(file_path)
        files_coverage.append((str(rel_path), pct))
        _check_coverage(result, rel_path, pct)

    files_coverage.sort(key=lambda x: x[1])
    result.stats["iso_29119"] = {
        "total_coverage": total_coverage,
        "min_required": MIN_COVERAGE,
        "compliant": total_coverage >= MIN_COVERAGE,
        "lowest_coverage": files_coverage[:10],
    }

    print(f"  Couverture globale: {total_coverage:.1f}% (min: {MIN_COVERAGE}%)")
    cov_file.unlink()


def _check_coverage(result: AuditResult, rel_path: Path, pct: float) -> None:
    """Vérifie la couverture d'un fichier."""
    if pct == 0:
        result.add_violation(
            norm="ISO 29119",
            severity="CRITICAL",
            file=str(rel_path),
            message="Couverture 0%",
            fix="Créer tests unitaires",
        )
    elif pct < 50:
        result.add_violation(
            norm="ISO 29119",
            severity="HIGH",
            file=str(rel_path),
            message=f"Couverture {pct:.0f}% (min: {MIN_COVERAGE}%)",
            fix="Ajouter tests pour branches manquantes",
        )
    elif pct < MIN_COVERAGE:
        result.add_violation(
            norm="ISO 29119",
            severity="MEDIUM",
            file=str(rel_path),
            message=f"Couverture {pct:.0f}% (min: {MIN_COVERAGE}%)",
            fix="Compléter tests edge cases",
        )


def audit_iso_15289_documentation(result: AuditResult, project_root: Path) -> None:
    """ISO 15289: Vérifie documentation."""
    print("\n[ISO 15289] Audit documentation...")

    required_docs = {
        "docs/architecture/ARCHITECTURE.md": "Architecture Description (AD)",
        "docs/api/API_CONTRACT.md": "Interface Design (IDD)",
        "docs/requirements/FEATURE_SPECIFICATION.md": "System Requirements (SyRS)",
        "docs/operations/DEPLOIEMENT_RENDER.md": "Operations Manual (OpsMan)",
        "docs/development/CONTRIBUTING.md": "Development Guide",
        "CLAUDE.md": "Project Guide",
        "README.md": "Software User Documentation (SUD)",
    }

    found = 0
    for doc_path, doc_type in required_docs.items():
        full_path = project_root / doc_path
        if full_path.exists():
            found += 1
            content = full_path.read_text(encoding="utf-8")
            if len(content) < 500:
                result.add_violation(
                    norm="ISO 15289",
                    severity="LOW",
                    file=doc_path,
                    message=f"{doc_type}: Contenu trop court ({len(content)} chars)",
                    fix="Compléter documentation",
                )
        else:
            result.add_violation(
                norm="ISO 15289",
                severity="MEDIUM",
                file=doc_path,
                message=f"{doc_type}: Document manquant",
                fix=f"Créer {doc_path}",
            )

    result.stats["iso_15289"] = {
        "required_docs": len(required_docs),
        "found_docs": found,
        "compliant": found == len(required_docs),
    }
    print(f"  Documents trouvés: {found}/{len(required_docs)}")


def audit_iso_5259_data_quality(result: AuditResult, project_root: Path) -> None:
    """ISO 5259: Vérifie qualité données ML."""
    print("\n[ISO 5259] Audit qualité données ML...")

    fe_files = [
        f
        for f in project_root.rglob("**/feature*.py")
        if not any(excl in f.parts for excl in EXCLUDED_DIRS)
    ]

    for fe_file in fe_files:
        content = fe_file.read_text(encoding="utf-8")
        if "fillna(0.5)" in content or "fillna(0)" in content:
            result.add_violation(
                norm="ISO 5259",
                severity="HIGH",
                file=str(fe_file.relative_to(project_root)),
                message="fillna() avec valeur arbitraire détecté",
                fix="Utiliser NaN + data_quality flag",
            )

    spec_file = project_root / "docs/requirements/FEATURE_SPECIFICATION.md"
    if spec_file.exists():
        spec_content = spec_file.read_text(encoding="utf-8")
        if "Source" not in spec_content and "source" not in spec_content:
            result.add_violation(
                norm="ISO 5259",
                severity="MEDIUM",
                file="docs/requirements/FEATURE_SPECIFICATION.md",
                message="Traçabilité sources features non documentée",
                fix="Ajouter colonne Source pour chaque feature",
            )

    result.stats["iso_5259"] = {"files_checked": len(fe_files)}
    print(f"  Fichiers vérifiés: {len(fe_files)}")


def audit_iso_25010_quality(result: AuditResult, project_root: Path) -> None:
    """ISO 25010: Vérifie qualité système (SRP)."""
    print("\n[ISO 25010] Audit qualité système (SRP)...")

    srp_violations = []
    for py_file in project_root.rglob("*.py"):
        if any(excl in py_file.parts for excl in EXCLUDED_DIRS):
            continue
        if "test_" in py_file.name:
            continue

        content = py_file.read_text(encoding="utf-8")
        func_count = content.count("\ndef ") + content.count("\nasync def ")

        if func_count > 20:
            rel_path = str(py_file.relative_to(project_root))
            srp_violations.append((rel_path, func_count))
            result.add_violation(
                norm="ISO 25010",
                severity="MEDIUM",
                file=rel_path,
                message=f"{func_count} fonctions (SRP: max ~10)",
                fix="Diviser en modules spécialisés",
            )

    result.stats["iso_25010"] = {
        "srp_violations": len(srp_violations),
        "worst_files": sorted(srp_violations, key=lambda x: -x[1])[:5],
    }
    print(f"  Violations SRP: {len(srp_violations)}")
