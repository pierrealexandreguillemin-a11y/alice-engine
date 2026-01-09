"""Tasks qualité code - ISO 5055/27034."""

from invoke import Context, task

SRC = "app/ services/ scripts/"
TESTS = "tests/"


@task
def lint(c: Context) -> None:
    """Lint code with ruff."""
    print("Lint Python (Ruff)...")
    c.run(f"ruff check {SRC} {TESTS} --fix")


@task
def format(c: Context) -> None:
    """Format code with ruff."""
    print("Format Python (Ruff)...")
    c.run(f"ruff format {SRC} {TESTS}")


@task
def typecheck(c: Context) -> None:
    """Check types with mypy."""
    print("Type check (MyPy)...")
    c.run("mypy app/ services/ --ignore-missing-imports")


@task
def security(c: Context) -> None:
    """Scan security with bandit."""
    print("Security scan (Bandit)...")
    c.run("bandit -c pyproject.toml -r app/ services/")


@task
def quality(c: Context) -> None:
    """Run all quality checks (lint + format + typecheck + security)."""
    lint(c)
    format(c)
    typecheck(c)
    security(c)
    print("\nQuality checks passed!")
