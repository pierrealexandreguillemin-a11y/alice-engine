"""Utilitaires graphs - ISO 5055.

Ce module contient les fonctions utilitaires communes:
- run_cmd: Execute commande shell
- print_header: Affiche en-tete formatee

Conformite ISO/IEC 5055 (<300 lignes, SRP).
"""

from __future__ import annotations

import subprocess  # nosec B404 - subprocess used for internal dev tools only
import sys
from pathlib import Path

# Chemins
ROOT = Path(__file__).parent.parent.parent
GRAPHS_DIR = ROOT / "graphs"
REPORTS_DIR = ROOT / "reports"
SRC_DIRS = ["app", "services"]


def run_cmd(cmd: list[str], capture: bool = True) -> tuple[int, str]:
    """Execute commande et retourne code + output.

    Args:
    ----
        cmd: Liste commande et arguments
        capture: Capturer stdout/stderr

    Returns:
    -------
        Tuple (code retour, output combine)
    """
    # Convert tool commands to python -m for cross-platform compatibility
    PYTHON_M_TOOLS = {"pydeps", "radon", "pylint"}
    if cmd and cmd[0] in PYTHON_M_TOOLS:
        cmd = [sys.executable, "-m"] + cmd

    try:
        result = subprocess.run(  # nosec B603, B607
            cmd,
            capture_output=capture,
            text=True,
            cwd=ROOT,
        )
        return result.returncode, result.stdout + result.stderr
    except FileNotFoundError:
        return -1, f"Command not found: {cmd[0]}"


def print_header(title: str) -> None:
    """Print formatted header.

    Args:
    ----
        title: Titre a afficher
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
