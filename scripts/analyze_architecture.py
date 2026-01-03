#!/usr/bin/env python3
"""
Analyse sante architecture - Detection spaghetti code
Equivalent: chess-app/scripts/analyze-architecture-health.js

Metriques:
- Coupling (afferent/efferent)
- Imports circulaires
- Score sante global
- Recommandations
"""

import ast
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
SRC_DIRS = ["app", "services"]


class ImportAnalyzer(ast.NodeVisitor):
    """Analyse les imports d'un fichier Python."""

    def __init__(self):
        self.imports: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append(node.module.split(".")[0])


def get_imports(file_path: Path) -> list[str]:
    """Extract imports from a Python file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        # Filter only internal imports
        internal = {"app", "services", "models", "tests"}
        return [i for i in analyzer.imports if i in internal]
    except Exception:
        return []


def analyze_dependencies() -> dict:
    """Analyze all dependencies in the project."""
    deps = defaultdict(list)

    for src_dir in SRC_DIRS:
        src_path = ROOT / src_dir
        if not src_path.exists():
            continue

        for py_file in src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            rel_path = str(py_file.relative_to(ROOT))
            imports = get_imports(py_file)
            deps[rel_path] = imports

    return dict(deps)


def calculate_coupling(deps: dict) -> list[dict]:
    """Calculate coupling metrics for each file."""
    files = list(deps.keys())
    coupling_data = []

    for file in files:
        # Efferent (Ce): modules this file depends on
        efferent = len(deps[file])

        # Afferent (Ca): modules that depend on this file
        file_module = file.split("/")[0]  # app or services
        afferent = sum(
            1 for other_deps in deps.values()
            if file_module in other_deps
        )

        total = afferent + efferent
        instability = efferent / total if total > 0 else 0

        coupling_data.append({
            "file": file,
            "afferent": afferent,
            "efferent": efferent,
            "total": total,
            "instability": round(instability * 100, 1),
        })

    return sorted(coupling_data, key=lambda x: -x["total"])


def detect_circular_imports(deps: dict) -> list[list[str]]:
    """Detect circular import chains."""
    # Build module-level dependency graph
    module_deps = defaultdict(set)

    for file, imports in deps.items():
        file_module = file.split("/")[0]
        for imp in imports:
            if imp != file_module:  # Skip self-references
                module_deps[file_module].add(imp)

    # Simple cycle detection
    cycles = []
    for module in module_deps:
        for dep in module_deps[module]:
            if module in module_deps.get(dep, set()):
                cycle = sorted([module, dep])
                if cycle not in cycles:
                    cycles.append(cycle)

    return cycles


def calculate_health_score(
    coupling_data: list[dict],
    circular: list,
    avg_deps: float,
    max_deps: int,
) -> tuple[int, list[str]]:
    """Calculate architecture health score (0-100)."""
    score = 100
    issues = []

    # Penalty for high average dependencies
    if avg_deps > 10:
        score -= 20
        issues.append(f" Moyenne deps/fichier trop elevee ({avg_deps:.1f} > 10)")
    elif avg_deps > 6:
        score -= 10
        issues.append(f" Moyenne deps/fichier elevee ({avg_deps:.1f} > 6)")

    # Penalty for high max dependencies
    if max_deps > 20:
        score -= 30
        issues.append(f" Fichier avec {max_deps} deps (hot spot critique)")
    elif max_deps > 10:
        score -= 15
        issues.append(f" Fichier avec {max_deps} deps (hot spot)")

    # Penalty for circular imports
    if circular:
        score -= len(circular) * 10
        issues.append(f" {len(circular)} dependance(s) circulaire(s)")

    # Penalty for high coupling files
    high_coupling = sum(1 for c in coupling_data if c["total"] > 15)
    if high_coupling > 3:
        score -= 20
        issues.append(f" {high_coupling} fichiers fortement couples (>15 deps)")
    elif high_coupling > 0:
        score -= 10
        issues.append(f" {high_coupling} fichier(s) fortement couple(s)")

    return max(0, score), issues


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> int:
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

    # Print coupling report
    print_header("METRIQUES COUPLING")
    print("\nTOP 10 FICHIERS COUPLES:\n")

    for i, item in enumerate(coupling_data[:10], 1):
        if item["total"] > 15:
            status = " CRITIQUE"
        elif item["total"] > 8:
            status = " WARNING"
        else:
            status = " OK"

        print(f"{i}. {status} {item['file']}")
        print(f"   Ca: {item['afferent']} | Ce: {item['efferent']} | Total: {item['total']}")
        print(f"   Instability: {item['instability']}%\n")

    # Print global metrics
    print_header("METRIQUES GLOBALES")
    print(f"\nFichiers analyses: {len(files)}")
    print(f"Dependances totales: {total_deps}")
    print(f"Moyenne deps/fichier: {avg_deps:.2f}")
    print(f"Max deps (fichier): {max_deps}")
    print(f"Imports circulaires: {len(circular)}")

    if circular:
        print("\nCycles detectes:")
        for cycle in circular:
            print(f"  - {' <-> '.join(cycle)}")

    # Calculate health score
    score, issues = calculate_health_score(coupling_data, circular, avg_deps, max_deps)

    print_header("SCORE SANTE ARCHITECTURE")
    print(f"\nScore: {score}/100\n")

    if score >= 80:
        print(" EXCELLENT - Architecture saine")
    elif score >= 60:
        print(" MOYEN - Ameliorations recommandees")
    elif score >= 40:
        print(" FAIBLE - Refactoring requis")
    else:
        print(" CRITIQUE - Spaghetti code detecte!")

    if issues:
        print("\nProblemes detectes:")
        for issue in issues:
            print(f"  {issue}")

    # Recommendations
    print_header("RECOMMANDATIONS")

    if avg_deps > 5:
        print("\n1.  Creer des modules/packages separes")
        print("   -> Regrouper fichiers lies en sous-modules")

    if max_deps > 10:
        print("\n2.  Refactorer fichiers hot spots")
        print("   -> Appliquer Dependency Inversion Principle")

    if circular:
        print("\n3.  Eliminer dependances circulaires")
        print("   -> Introduire interfaces/abstractions")

    print("\n4.  Definir regles architecture:")
    print("   -> Max 10 deps/fichier")
    print("   -> 0 circular deps")
    print("   -> Layered architecture (api -> services -> models)")

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "architecture-health.json"

    report = {
        "date": datetime.now().isoformat(),
        "score": score,
        "metrics": {
            "files": len(files),
            "totalDeps": total_deps,
            "avgDeps": round(avg_deps, 2),
            "maxDeps": max_deps,
            "circular": len(circular),
            "highCoupling": sum(1 for c in coupling_data if c["total"] > 15),
        },
        "topCoupled": coupling_data[:10],
        "issues": issues,
        "circularDeps": circular,
    }

    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n Rapport sauvegarde: {report_path}")

    return 0 if score >= 60 else 1


if __name__ == "__main__":
    sys.exit(main())
