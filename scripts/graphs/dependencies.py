"""Generation graphs dependances - ISO 42010.

Ce module genere les graphs de dependances:
- generate_dependency_graph: Graph principal (pydeps)
- generate_imports_graph: Graph imports structure
- check_circular_imports: Detection cycles

Conformite ISO 42010 (Architecture) + ISO/IEC 5055.
"""

from __future__ import annotations

from scripts.graphs.utils import GRAPHS_DIR, ROOT, run_cmd


def generate_dependency_graph() -> bool:
    """Generate dependency graph with pydeps.

    Returns
    -------
        True si generation reussie
    """
    print("\n[1/5] Graph dependances (pydeps)...")

    # Check if pydeps installed
    code, _ = run_cmd(["pydeps", "--version"])
    if code != 0:
        print("  ! pydeps non installe: pip install pydeps")
        return False

    # Check if graphviz installed
    code, _ = run_cmd(["dot", "-V"])
    if code != 0:
        print("  ! Graphviz non installe")
        print("    Windows: choco install graphviz")
        print("    Mac: brew install graphviz")
        return False

    output = GRAPHS_DIR / "dependencies.svg"

    # Generate for app module
    code, out = run_cmd(
        [
            "pydeps",
            "app",
            "--cluster",
            "--max-bacon=2",
            "--no-show",
            "-o",
            str(output),
        ]
    )

    if code == 0 and output.exists():
        print(f"  OK {output.relative_to(ROOT)}")
        return True
    else:
        print(f"  ! Echec: {out[:200]}")
        return False


def generate_imports_graph() -> bool:
    """Generate imports structure visualization.

    Returns
    -------
        True si generation reussie
    """
    print("\n[2/5] Graph structure imports...")

    output = GRAPHS_DIR / "imports.svg"

    # Try with pydeps on services
    code, out = run_cmd(
        [
            "pydeps",
            "services",
            "--cluster",
            "--max-bacon=3",
            "--no-show",
            "-o",
            str(output),
        ]
    )

    if code == 0 and output.exists():
        print(f"  OK {output.relative_to(ROOT)}")
        return True
    else:
        print("  ! Echec generation imports graph")
        return False


def check_circular_imports() -> bool:
    """Check for circular imports.

    Returns
    -------
        True si aucun import circulaire
    """
    print("\n[5/5] Detection imports circulaires...")

    output_file = GRAPHS_DIR / "circular-imports.txt"

    # Use pydeps to detect cycles
    code, out = run_cmd(
        [
            "pydeps",
            "app",
            "--show-cycles",
            "--no-output",
        ]
    )

    if "No circular" in out or code == 0:
        output_file.write_text("No circular imports detected.\n")
        print("  OK Aucun import circulaire")
        return True
    else:
        output_file.write_text(out)
        print(f"  ! Imports circulaires detectes: {output_file.relative_to(ROOT)}")
        return False
