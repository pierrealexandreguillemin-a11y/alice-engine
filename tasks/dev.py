"""Tasks développement."""

import shutil
from pathlib import Path

from invoke import Context, task


@task
def install(c: Context) -> None:
    """Install prod and dev dependencies."""
    c.run("pip install -r requirements.txt")
    c.run("pip install -r requirements-dev.txt")
    print("\nDependances installees!")
    print("Executez 'inv hooks' pour installer les git hooks")


@task
def hooks(c: Context) -> None:
    """Install pre-commit hooks."""
    c.run("pre-commit install")
    c.run("pre-commit install --hook-type commit-msg")
    c.run("pre-commit install --hook-type pre-push")
    print("\nHooks installes:")
    print("  - pre-commit: lint, format, typecheck, security")
    print("  - commit-msg: conventional commits")
    print("  - pre-push: tests, coverage, audit, graphs, iso-docs")


@task
def dev(c: Context) -> None:
    """Start development server."""
    c.run("uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")


@task
def clean(c: Context) -> None:
    """Clean temporary files."""
    print("Nettoyage fichiers temporaires...")

    dirs_to_clean = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
    ]
    for d in dirs_to_clean:
        for path in Path(".").rglob(d):
            shutil.rmtree(path, ignore_errors=True)

    for pyc in Path(".").rglob("*.pyc"):
        pyc.unlink(missing_ok=True)

    Path(".coverage").unlink(missing_ok=True)
    print("Nettoyage termine!")
