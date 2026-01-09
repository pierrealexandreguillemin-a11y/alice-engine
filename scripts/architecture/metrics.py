"""Métriques architecture - ISO 25010/5055."""

from __future__ import annotations

from collections import defaultdict


def calculate_coupling(deps: dict[str, list[str]]) -> list[dict]:
    """Calculate coupling metrics for each file."""
    files = list(deps.keys())
    coupling_data = []

    for file in files:
        efferent = len(deps[file])
        file_module = file.split("/")[0]
        afferent = sum(1 for other_deps in deps.values() if file_module in other_deps)

        total = afferent + efferent
        instability = efferent / total if total > 0 else 0

        coupling_data.append(
            {
                "file": file,
                "afferent": afferent,
                "efferent": efferent,
                "total": total,
                "instability": round(instability * 100, 1),
            }
        )

    return sorted(coupling_data, key=lambda x: -x["total"])


def detect_circular_imports(deps: dict[str, list[str]]) -> list[list[str]]:
    """Detect circular import chains."""
    module_deps: dict[str, set[str]] = defaultdict(set)

    for file, imports in deps.items():
        file_module = file.split("/")[0]
        for imp in imports:
            if imp != file_module:
                module_deps[file_module].add(imp)

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
    circular: list[list[str]],
    avg_deps: float,
    max_deps: int,
) -> tuple[int, list[str]]:
    """Calculate architecture health score (0-100)."""
    score = 100
    issues = []

    if avg_deps > 10:
        score -= 20
        issues.append(f"Moyenne deps/fichier trop elevee ({avg_deps:.1f} > 10)")
    elif avg_deps > 6:
        score -= 10
        issues.append(f"Moyenne deps/fichier elevee ({avg_deps:.1f} > 6)")

    if max_deps > 20:
        score -= 30
        issues.append(f"Fichier avec {max_deps} deps (hot spot critique)")
    elif max_deps > 10:
        score -= 15
        issues.append(f"Fichier avec {max_deps} deps (hot spot)")

    if circular:
        score -= len(circular) * 10
        issues.append(f"{len(circular)} dependance(s) circulaire(s)")

    high_coupling = sum(1 for c in coupling_data if c["total"] > 15)
    if high_coupling > 3:
        score -= 20
        issues.append(f"{high_coupling} fichiers fortement couples (>15 deps)")
    elif high_coupling > 0:
        score -= 10
        issues.append(f"{high_coupling} fichier(s) fortement couple(s)")

    return max(0, score), issues
