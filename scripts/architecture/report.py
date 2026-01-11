"""Rapport architecture - ISO 15289."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"


def print_header(title: str) -> None:
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_report(
    coupling_data: list[dict],
    circular: list[list[str]],
    files: list[str],
    total_deps: int,
    avg_deps: float,
    max_deps: int,
    score: int,
    issues: list[str],
) -> None:
    """Print full architecture report."""
    _print_coupling_section(coupling_data)
    _print_global_metrics(files, total_deps, avg_deps, max_deps, circular)
    _print_score_section(score, issues)
    _print_recommendations(avg_deps, max_deps, circular)


def _print_coupling_section(coupling_data: list[dict]) -> None:
    """Print coupling metrics section."""
    print_header("METRIQUES COUPLING")
    print("\nTOP 10 FICHIERS COUPLES:\n")

    for i, item in enumerate(coupling_data[:10], 1):
        status = _get_coupling_status(item["total"])
        print(f"{i}. [{status}] {item['file']}")
        print(f"   Ca: {item['afferent']} | Ce: {item['efferent']} | Total: {item['total']}")
        print(f"   Instability: {item['instability']}%\n")


def _get_coupling_status(total: int) -> str:
    """Determine coupling status."""
    if total > 15:
        return "CRITIQUE"
    if total > 8:
        return "WARNING"
    return "OK"


def _print_global_metrics(
    files: list[str], total_deps: int, avg_deps: float, max_deps: int, circular: list[list[str]]
) -> None:
    """Print global metrics section."""
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


def _print_score_section(score: int, issues: list[str]) -> None:
    """Print score section."""
    print_header("SCORE SANTE ARCHITECTURE")
    print(f"\nScore: {score}/100\n")
    print(_get_score_label(score))

    if issues:
        print("\nProblemes detectes:")
        for issue in issues:
            print(f"  - {issue}")


def _get_score_label(score: int) -> str:
    """Get score label."""
    if score >= 80:
        return "[EXCELLENT] Architecture saine"
    if score >= 60:
        return "[MOYEN] Ameliorations recommandees"
    if score >= 40:
        return "[FAIBLE] Refactoring requis"
    return "[CRITIQUE] Spaghetti code detecte!"


def _print_recommendations(avg_deps: float, max_deps: int, circular: list[list[str]]) -> None:
    """Print architecture recommendations."""
    print_header("RECOMMANDATIONS")

    if avg_deps > 5:
        print("\n1. Creer des modules/packages separes")
        print("   -> Regrouper fichiers lies en sous-modules")

    if max_deps > 10:
        print("\n2. Refactorer fichiers hot spots")
        print("   -> Appliquer Dependency Inversion Principle")

    if circular:
        print("\n3. Eliminer dependances circulaires")
        print("   -> Introduire interfaces/abstractions")

    print("\n4. Definir regles architecture:")
    print("   -> Max 10 deps/fichier")
    print("   -> 0 circular deps")
    print("   -> Layered architecture (api -> services -> models)")


def save_report(
    coupling_data: list[dict],
    circular: list[list[str]],
    files: list[str],
    total_deps: int,
    avg_deps: float,
    max_deps: int,
    score: int,
    issues: list[str],
) -> Path:
    """Save architecture report to JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "architecture-health.json"

    report: dict[str, Any] = {
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
    print(f"\nRapport sauvegarde: {report_path}")

    return report_path
