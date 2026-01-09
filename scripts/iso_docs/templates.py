"""Report template generation for ISO documentation."""

from datetime import datetime

from .checkers import check_installed, get_complexity_avg, get_test_coverage
from .modules import MODULES


def _generate_iso_table(results: list[dict], iso_filter: str) -> str:
    """Generate a markdown table for modules matching an ISO standard."""
    lines = []
    for r in results:
        if iso_filter in r["iso"]:
            status = "" if r["installed"] else ""
            lines.append(f"| {r['name']} | {status} | {r['version']} |")
    return "\n".join(lines)


def _generate_header(timestamp: str, score: int, installed: int, total: int) -> str:
    """Generate report header section."""
    return f"""# IMPLEMENTATION DEVOPS - STATUT ISO

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


def _generate_iso_sections(results: list[dict]) -> str:
    """Generate all ISO standard sections."""
    sections = []

    # ISO 25010
    sections.append(_generate_iso_table(results, "ISO 25010"))

    # ISO 27001 / ISO 27034
    sections.append("""
### ISO 27001 / ISO 27034 - Securite
Protection OWASP Top 10, Secret scanning

| Outil | Status | Version |
|-------|--------|---------|
""")
    for r in results:
        if "ISO 27001" in r["iso"] or "ISO 27034" in r["iso"]:
            status = "" if r["installed"] else ""
            sections.append(f"| {r['name']} | {status} | {r['version']} |")

    # ISO 29119
    sections.append("""
### ISO 29119 - Tests Logiciels
Pyramide de tests, Coverage enforcement

| Outil | Status | Version |
|-------|--------|---------|
""")
    sections.append(_generate_iso_table(results, "ISO 29119"))

    # ISO 42010
    sections.append("""
### ISO 42010 - Architecture
Visualisation dependances, Detection cycles

| Outil | Status | Version |
|-------|--------|---------|
""")
    sections.append(_generate_iso_table(results, "ISO 42010"))

    return "\n".join(sections)


def _generate_metrics_section(coverage: str, complexity: str) -> str:
    """Generate quality metrics section."""
    cov_ok = coverage != "N/A" and float(coverage.replace("%", "")) >= 80
    cplx_ok = complexity.startswith("A") or complexity.startswith("B")

    return f"""
---

## METRIQUES QUALITE ACTUELLES

| Metrique | Valeur Actuelle | Seuil ISO | Statut |
|----------|-----------------|-----------|--------|
| Test Coverage | {coverage} | >80% (ISO 29119) | {"" if cov_ok else ""} |
| Complexite Moyenne | {complexity} | <B (ISO 25010) | {"" if cplx_ok else ""} |
| Vulnerabilites deps | A verifier | 0 (ISO 27001) |  |
| Imports circulaires | A verifier | 0 (ISO 42010) |  |

---

## TABLEAU COMPLET

| Module | Categorie | Statut | Version | Norme ISO |
|--------|-----------|--------|---------|-----------|
"""


def _generate_full_table(results: list[dict]) -> str:
    """Generate full module table sorted by status and category."""
    lines = []
    for r in sorted(results, key=lambda x: (not x["installed"], x["category"])):
        status = "" if r["installed"] else ""
        lines.append(f"| {r['name']} | {r['category']} | {status} | {r['version']} | {r['iso']} |")
    return "\n".join(lines)


def _generate_missing_section(results: list[dict]) -> str:
    """Generate section listing missing modules."""
    missing = [r for r in results if not r["installed"]]
    lines = ["\n---\n\n## MODULES MANQUANTS (actions requises)\n\n"]

    if missing:
        for r in missing:
            lines.append(f"- **{r['name']}** ({r['category']}) - {r['iso']}")
        lines.append("\n```bash")
        lines.append("pip install " + " ".join(r["name"] for r in missing))
        lines.append("```\n")
    else:
        lines.append("_Tous les modules recommandes sont installes_\n")

    return "\n".join(lines)


def _generate_footer(timestamp: str, score: int) -> str:
    """Generate report footer with commands."""
    return f"""
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


def generate_report() -> str:
    """Generate ISO implementation status report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Check all modules
    results = []
    for mod in MODULES:
        installed, version = check_installed(mod["check"])
        results.append({**mod, "installed": installed, "version": version})

    # Calculate scores
    total = len(results)
    installed_count = sum(1 for r in results if r["installed"])
    score = round((installed_count / total) * 100)

    # Get metrics
    coverage = get_test_coverage()
    complexity = get_complexity_avg()

    # Build report from sections
    report_parts = [
        _generate_header(timestamp, score, installed_count, total),
        _generate_iso_sections(results),
        _generate_metrics_section(coverage, complexity),
        _generate_full_table(results),
        _generate_missing_section(results),
        _generate_footer(timestamp, score),
    ]

    return "".join(report_parts)
