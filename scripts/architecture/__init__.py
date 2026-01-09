"""Package architecture - ISO 42010/25010/5055.

Analyse santé architecture et détection spaghetti code.
"""

from scripts.architecture.analyzer import analyze_dependencies, get_imports
from scripts.architecture.metrics import (
    calculate_coupling,
    calculate_health_score,
    detect_circular_imports,
)
from scripts.architecture.report import print_report, save_report

__all__ = [
    "analyze_dependencies",
    "get_imports",
    "calculate_coupling",
    "calculate_health_score",
    "detect_circular_imports",
    "print_report",
    "save_report",
]
