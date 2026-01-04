#!/usr/bin/env python3
"""
Generation de graphs architecture et qualite - Equivalent chess-app
ISO 42010 (Architecture) + ISO 25010 (Qualite)

Genere:
- graphs/dependencies.svg      (pydeps)
- graphs/imports.svg           (imports structure)
- reports/complexity/          (radon)
- reports/duplication/         (pylint)
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Chemins
ROOT = Path(__file__).parent.parent
GRAPHS_DIR = ROOT / "graphs"
REPORTS_DIR = ROOT / "reports"
SRC_DIRS = ["app", "services"]


def run_cmd(cmd: list[str], capture: bool = True) -> tuple[int, str]:
    """Execute commande et retourne code + output."""
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
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def generate_dependency_graph() -> bool:
    """Generate dependency graph with pydeps."""
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
    """Generate imports structure visualization."""
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


def generate_complexity_report() -> bool:
    """Generate complexity report with radon."""
    print("\n[3/5] Rapport complexite (radon)...")

    # Check radon
    code, _ = run_cmd(["radon", "--version"])
    if code != 0:
        print("  ! radon non installe: pip install radon")
        return False

    report_dir = REPORTS_DIR / "complexity"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Cyclomatic complexity
    cc_file = report_dir / "cyclomatic.txt"
    code, out = run_cmd(["radon", "cc", "app", "services", "-a", "-s"])
    if code == 0:
        cc_file.write_text(out)
        print(f"  OK {cc_file.relative_to(ROOT)}")

    # Maintainability index
    mi_file = report_dir / "maintainability.txt"
    code, out = run_cmd(["radon", "mi", "app", "services", "-s"])
    if code == 0:
        mi_file.write_text(out)
        print(f"  OK {mi_file.relative_to(ROOT)}")

    # Raw metrics
    raw_file = report_dir / "raw_metrics.txt"
    code, out = run_cmd(["radon", "raw", "app", "services", "-s"])
    if code == 0:
        raw_file.write_text(out)
        print(f"  OK {raw_file.relative_to(ROOT)}")

    # Generate HTML summary
    html_report = report_dir / "index.html"
    generate_complexity_html(report_dir, html_report)

    return True


def generate_complexity_html(report_dir: Path, output: Path) -> None:
    """Generate HTML report from radon output."""
    cc_file = report_dir / "cyclomatic.txt"
    mi_file = report_dir / "maintainability.txt"

    cc_content = cc_file.read_text() if cc_file.exists() else "N/A"
    mi_content = mi_file.read_text() if mi_file.exists() else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport Complexite - Alice Engine</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .card {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        pre {{ background: #f8f9fa; padding: 15px; overflow-x: auto; border-radius: 4px; }}
        .metric {{ display: inline-block; background: #007bff; color: white; padding: 5px 15px; border-radius: 20px; margin: 5px; }}
        .legend {{ background: #e9ecef; padding: 10px; border-radius: 4px; margin-top: 10px; }}
        .legend span {{ margin-right: 15px; }}
        .A {{ color: #28a745; }} .B {{ color: #20c997; }} .C {{ color: #ffc107; }} .D {{ color: #fd7e14; }} .E {{ color: #dc3545; }} .F {{ color: #6c757d; }}
    </style>
</head>
<body>
    <h1>Rapport Complexite - Alice Engine</h1>
    <p>Genere le: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

    <div class="card">
        <h2>Complexite Cyclomatique (Radon CC)</h2>
        <div class="legend">
            <span class="A">A (1-5)</span>
            <span class="B">B (6-10)</span>
            <span class="C">C (11-20)</span>
            <span class="D">D (21-30)</span>
            <span class="E">E (31-40)</span>
            <span class="F">F (41+)</span>
        </div>
        <pre>{cc_content}</pre>
    </div>

    <div class="card">
        <h2>Indice de Maintenabilite (Radon MI)</h2>
        <div class="legend">
            <span class="A">A (20+) Excellent</span>
            <span class="B">B (10-19) Bon</span>
            <span class="C">C (0-9) Moyen</span>
        </div>
        <pre>{mi_content}</pre>
    </div>

    <div class="card">
        <h2>Conformite ISO</h2>
        <ul>
            <li><strong>ISO 25010</strong> - Maintenabilite: Analysabilite, Modifiabilite</li>
            <li><strong>ISO 5055</strong> - Qualite code: Complexite cognitive</li>
        </ul>
    </div>
</body>
</html>"""

    output.write_text(html, encoding="utf-8")
    print(f"  OK {output.relative_to(ROOT)}")


def check_duplication() -> bool:
    """Check code duplication with pylint."""
    print("\n[4/5] Detection duplication (pylint)...")

    report_dir = REPORTS_DIR / "duplication"
    report_dir.mkdir(parents=True, exist_ok=True)

    output_file = report_dir / "duplicates.txt"

    # Use pylint similarities checker
    code, out = run_cmd(
        [
            "pylint",
            "--disable=all",
            "--enable=duplicate-code",
            "app",
            "services",
        ]
    )

    output_file.write_text(out)
    print(f"  OK {output_file.relative_to(ROOT)}")

    # Count duplicates
    dup_count = out.count("Similar lines in")
    if dup_count > 0:
        print(f"  ! {dup_count} blocs dupliques detectes")
    else:
        print("  OK Aucune duplication majeure")

    return True


def check_circular_imports() -> bool:
    """Check for circular imports."""
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


def main() -> int:
    print_header("GENERATION GRAPHS ARCHITECTURE & QUALITE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Repertoire: {ROOT}")

    # Create directories
    GRAPHS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    results = []

    results.append(("Dependencies Graph", generate_dependency_graph()))
    results.append(("Imports Graph", generate_imports_graph()))
    results.append(("Complexity Report", generate_complexity_report()))
    results.append(("Duplication Check", check_duplication()))
    results.append(("Circular Imports", check_circular_imports()))

    # Summary
    print_header("RESUME")

    success = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "OK" if ok else "!"
        print(f"  [{status}] {name}")

    print(f"\nScore: {success}/{total}")

    if success == total:
        print("\nTous les graphs generes avec succes!")
        return 0
    else:
        print("\nCertains graphs n'ont pas pu etre generes.")
        print("Verifier les dependances: pip install pydeps radon pylint")
        return 1


if __name__ == "__main__":
    sys.exit(main())
