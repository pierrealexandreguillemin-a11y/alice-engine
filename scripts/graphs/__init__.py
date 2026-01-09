"""Graphs et rapports architecture - ISO 42010/25010.

Ce package genere les graphs et rapports qualite:
- dependencies.py: Graph dependances (pydeps)
- complexity.py: Rapport complexite (radon)
- utils.py: Utilitaires communs

Conformite ISO 42010 (Architecture) + ISO 25010 (Qualite).
"""

from scripts.graphs.complexity import (
    check_duplication,
    generate_complexity_html,
    generate_complexity_report,
)
from scripts.graphs.dependencies import (
    check_circular_imports,
    generate_dependency_graph,
    generate_imports_graph,
)
from scripts.graphs.utils import print_header, run_cmd

__all__ = [
    # Utils
    "run_cmd",
    "print_header",
    # Dependencies
    "generate_dependency_graph",
    "generate_imports_graph",
    "check_circular_imports",
    # Complexity
    "generate_complexity_report",
    "generate_complexity_html",
    "check_duplication",
]
