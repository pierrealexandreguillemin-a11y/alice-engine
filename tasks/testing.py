"""Tasks tests et audit - ISO 29119."""

from invoke import Context, task

from tasks.architecture import architecture, graphs, iso_docs
from tasks.iso_audit import iso_audit
from tasks.quality import quality

TESTS = "tests/"


@task
def test(c: Context) -> None:
    """Run unit tests with pytest."""
    print("Tests unitaires (pytest)...")
    c.run(f"pytest {TESTS} -v --tb=short")


@task(name="test-cov")
def test_cov(c: Context) -> None:
    """Run tests with coverage (min 80%)."""
    print("Tests + Coverage...")
    c.run(
        f"pytest {TESTS} --cov=app --cov=services --cov-report=html "
        "--cov-report=term --cov-fail-under=80"
    )
    print("\nRapport HTML: htmlcov/index.html")


@task
def audit(c: Context) -> None:
    """Audit dependencies with pip-audit."""
    print("Audit dependances (pip-audit)...")
    c.run("pip-audit --strict --desc")


@task
def complexity(c: Context) -> None:
    """Analyze code complexity with radon and xenon."""
    print("\n=== Complexite Cyclomatique (Radon) ===")
    c.run("radon cc app/ services/ -a -nc")
    print("\n=== Indice de Maintenabilite ===")
    c.run("radon mi app/ services/ -s")
    print("\n=== Seuils Complexite (Xenon) ===")
    c.run(
        "xenon --max-absolute=B --max-modules=B --max-average=A app/ services/",
        warn=True,
    )


@task
def validate(c: Context) -> None:
    """Run complete validation (quality + test-cov + audit)."""
    quality(c)
    test_cov(c)
    audit(c)
    print("\nValidation complete passee!")


@task(name="all")
def all_checks(c: Context) -> None:
    """Run all checks (quality + tests + audit + graphs + ISO)."""
    quality(c)
    test_cov(c)
    audit(c)
    complexity(c)
    architecture(c)
    graphs(c)
    iso_docs(c)
    print("\n" + "=" * 44)
    print("  TOUS LES CHECKS PASSES!")
    print("  Ready for push.")
    print("=" * 44)
    print("\nArtefacts generes:")
    print("  - graphs/dependencies.svg")
    print("  - reports/complexity/index.html")
    print("  - reports/architecture-health.json")
    print("  - docs/iso/IMPLEMENTATION_STATUS.md")


@task(name="all-iso")
def all_iso(c: Context) -> None:
    """Run complete ISO validation (all + iso-audit)."""
    all_checks(c)
    iso_audit(c)
    print("\n" + "=" * 44)
    print("  VALIDATION ISO COMPLETE PASSEE!")
    print("=" * 44)
    print("\nNormes validees:")
    print("  - ISO 5055  (Code Quality)")
    print("  - ISO 5259  (Data Quality ML)")
    print("  - ISO 15289 (Documentation)")
    print("  - ISO 25010 (System Quality)")
    print("  - ISO 27001 (Security)")
    print("  - ISO 27034 (Secure Coding)")
    print("  - ISO 29119 (Testing)")
    print("  - ISO 42001 (AI Management)")
    print("  - ISO 42010 (Architecture)")
