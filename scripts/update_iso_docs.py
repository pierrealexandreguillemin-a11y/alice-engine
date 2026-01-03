#!/usr/bin/env python3
"""
Mise a jour documentation ISO automatique
Equivalent: chess-app/scripts/update-iso-docs.cjs

Genere: docs/iso/IMPLEMENTATION_STATUS.md
Trace conformite aux normes ISO 25010, 27001, 5055, etc.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS_ISO = ROOT / "docs" / "iso"

# Modules et leur conformite ISO
MODULES = [
    # === QUALITE CODE ===
    {
        "name": "ruff",
        "category": "Qualite Code",
        "iso": "ISO 25010, ISO 5055",
        "check": "ruff --version",
    },
    {
        "name": "mypy",
        "category": "Type Safety",
        "iso": "ISO 25010, ISO 5055",
        "check": "mypy --version",
    },
    {"name": "black", "category": "Formatage", "iso": "ISO 25010", "check": "black --version"},
    {"name": "isort", "category": "Formatage", "iso": "ISO 25010", "check": "isort --version"},
    # === SECURITE ===
    {
        "name": "bandit",
        "category": "Securite",
        "iso": "ISO 27001, ISO 27034, OWASP",
        "check": "bandit --version",
    },
    {"name": "safety", "category": "Securite", "iso": "ISO 27001", "check": "safety --version"},
    {
        "name": "pip-audit",
        "category": "Securite",
        "iso": "ISO 27001",
        "check": "pip-audit --version",
    },
    {
        "name": "gitleaks",
        "category": "Secrets",
        "iso": "ISO 27001 (P0-CRITICAL)",
        "check": "gitleaks version",
    },
    # === TESTS ===
    {"name": "pytest", "category": "Tests", "iso": "ISO 29119", "check": "pytest --version"},
    {
        "name": "pytest-cov",
        "category": "Coverage",
        "iso": "ISO 29119",
        "check": "pytest --co -q 2>&1 | head -1",
    },
    {
        "name": "pytest-asyncio",
        "category": "Tests Async",
        "iso": "ISO 29119",
        "check": "python -c 'import pytest_asyncio'",
    },
    # === COMPLEXITE ===
    {
        "name": "radon",
        "category": "Complexite",
        "iso": "ISO 25010, ISO 5055",
        "check": "radon --version",
    },
    {"name": "xenon", "category": "Complexite", "iso": "ISO 25010", "check": "xenon --version"},
    # === ARCHITECTURE ===
    {"name": "pydeps", "category": "Architecture", "iso": "ISO 42010", "check": "pydeps --version"},
    {
        "name": "import-linter",
        "category": "Architecture",
        "iso": "ISO 42010",
        "check": "lint-imports --version",
    },
    # === HOOKS ===
    {
        "name": "pre-commit",
        "category": "Git Hooks",
        "iso": "ISO 12207",
        "check": "pre-commit --version",
    },
    {"name": "commitizen", "category": "Commits", "iso": "ISO 12207", "check": "cz version"},
    # === DOCUMENTATION ===
    {
        "name": "mkdocs",
        "category": "Documentation",
        "iso": "ISO 26514",
        "check": "mkdocs --version",
    },
    {"name": "pdoc", "category": "API Docs", "iso": "ISO 26514", "check": "pdoc --version"},
]


def check_installed(cmd: str) -> tuple[bool, str]:
    """Check if a tool is installed."""
    try:
        result = subprocess.run(  # nosec B603, B607
            cmd.split(),
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = result.stdout.strip() or result.stderr.strip()
        # Extract version number
        version = version.split("\n")[0][:50] if version else "installed"
        return result.returncode == 0, version
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "non installe"


def get_test_coverage() -> str:
    """Get current test coverage."""
    try:
        result = subprocess.run(  # nosec B603, B607
            ["pytest", "--cov=app", "--cov=services", "--cov-report=term", "-q", "--tb=no"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=60,
        )
        for line in result.stdout.split("\n"):
            if "TOTAL" in line:
                parts = line.split()
                for part in parts:
                    if "%" in part:
                        return part
        return "N/A"
    except Exception:
        return "N/A"


def get_complexity_avg() -> str:
    """Get average complexity."""
    try:
        result = subprocess.run(  # nosec B603, B607
            ["radon", "cc", "app", "services", "-a"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
        )
        for line in result.stdout.split("\n"):
            if "Average complexity" in line:
                return line.split(":")[-1].strip()
        return "N/A"
    except Exception:
        return "N/A"


def generate_report() -> str:
    """Generate ISO implementation status report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Check all modules
    results = []
    for mod in MODULES:
        installed, version = check_installed(mod["check"])
        results.append(
            {
                **mod,
                "installed": installed,
                "version": version,
            }
        )

    # Calculate scores
    total = len(results)
    installed = sum(1 for r in results if r["installed"])
    score = round((installed / total) * 100)

    # Group by category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "installed": 0}
        categories[cat]["total"] += 1
        if r["installed"]:
            categories[cat]["installed"] += 1

    # Get metrics
    coverage = get_test_coverage()
    complexity = get_complexity_avg()

    # Generate markdown
    report = f"""# IMPLEMENTATION DEVOPS - STATUT ISO

**Date de mise a jour:** {timestamp}
**Generateur:** scripts/update_iso_docs.py (automatique)

---

## SCORE DEVOPS GLOBAL

```
Score actuel: {score}/100

Modules installes:   {installed}/{total}
Modules manquants:   {total - installed}
```

---

## CONFORMITE PAR NORME ISO

### ISO 25010 - Qualite Produit Logiciel
Couvre: Maintenabilite, Fiabilite, Securite, Performance

| Outil | Status | Version |
|-------|--------|---------|
"""

    for r in results:
        if "ISO 25010" in r["iso"]:
            status = "" if r["installed"] else ""
            report += f"| {r['name']} | {status} | {r['version']} |\n"

    report += """
### ISO 27001 / ISO 27034 - Securite
Protection OWASP Top 10, Secret scanning

| Outil | Status | Version |
|-------|--------|---------|
"""

    for r in results:
        if "ISO 27001" in r["iso"] or "ISO 27034" in r["iso"]:
            status = "" if r["installed"] else ""
            report += f"| {r['name']} | {status} | {r['version']} |\n"

    report += """
### ISO 29119 - Tests Logiciels
Pyramide de tests, Coverage enforcement

| Outil | Status | Version |
|-------|--------|---------|
"""

    for r in results:
        if "ISO 29119" in r["iso"]:
            status = "" if r["installed"] else ""
            report += f"| {r['name']} | {status} | {r['version']} |\n"

    report += """
### ISO 42010 - Architecture
Visualisation dependances, Detection cycles

| Outil | Status | Version |
|-------|--------|---------|
"""

    for r in results:
        if "ISO 42010" in r["iso"]:
            status = "" if r["installed"] else ""
            report += f"| {r['name']} | {status} | {r['version']} |\n"

    report += f"""
---

## METRIQUES QUALITE ACTUELLES

| Metrique | Valeur Actuelle | Seuil ISO | Statut |
|----------|-----------------|-----------|--------|
| Test Coverage | {coverage} | >80% (ISO 29119) | {"" if coverage != "N/A" and float(coverage.replace("%", "")) >= 80 else ""} |
| Complexite Moyenne | {complexity} | <B (ISO 25010) | {"" if complexity.startswith("A") or complexity.startswith("B") else ""} |
| Vulnerabilites deps | A verifier | 0 (ISO 27001) |  |
| Imports circulaires | A verifier | 0 (ISO 42010) |  |

---

## TABLEAU COMPLET

| Module | Categorie | Statut | Version | Norme ISO |
|--------|-----------|--------|---------|-----------|
"""

    for r in sorted(results, key=lambda x: (not x["installed"], x["category"])):
        status = "" if r["installed"] else ""
        report += f"| {r['name']} | {r['category']} | {status} | {r['version']} | {r['iso']} |\n"

    report += """
---

## MODULES MANQUANTS (actions requises)

"""

    missing = [r for r in results if not r["installed"]]
    if missing:
        for r in missing:
            report += f"- **{r['name']}** ({r['category']}) - {r['iso']}\n"
        report += "\n```bash\npip install " + " ".join(r["name"] for r in missing) + "\n```\n"
    else:
        report += "_Tous les modules recommandes sont installes_\n"

    report += f"""
---

## COMMANDES RAPIDES

```bash
# Qualite code
make quality          # Lint + Format + Typecheck + Security

# Tests
make test-cov         # Tests avec coverage

# Graphs
python scripts/generate_graphs.py

# Documentation ISO
python scripts/update_iso_docs.py

# Audit complet
make all
```

---

**Genere automatiquement par:** scripts/update_iso_docs.py
**Derniere mise a jour:** {timestamp}
**Score DevOps:** {score}/100
"""

    return report


def main() -> int:
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
