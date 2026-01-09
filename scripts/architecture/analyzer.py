"""Analyseur de dépendances - ISO 42010."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SRC_DIRS = ["app", "services"]


class ImportAnalyzer(ast.NodeVisitor):
    """Analyse les imports d'un fichier Python."""

    def __init__(self) -> None:
        """Initialize import list."""
        self.imports: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        """Visit import node."""
        for alias in node.names:
            self.imports.append(alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        """Visit import from node."""
        if node.module:
            self.imports.append(node.module.split(".")[0])


def get_imports(file_path: Path) -> list[str]:
    """Extract imports from a Python file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        internal = {"app", "services", "models", "tests"}
        return [i for i in analyzer.imports if i in internal]
    except Exception:
        return []


def analyze_dependencies() -> dict[str, list[str]]:
    """Analyze all dependencies in the project."""
    deps: dict[str, list[str]] = defaultdict(list)

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
