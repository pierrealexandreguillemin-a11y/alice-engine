"""Tasks architecture - ISO 42010."""

from invoke import Context, task


@task
def graphs(c: Context) -> None:
    """Generate SVG dependency graphs."""
    print("Generation graphs architecture...")
    c.run("python scripts/generate_graphs.py")
    print("\nGraphs generes dans: graphs/")


@task
def architecture(c: Context) -> None:
    """Analyze architecture health."""
    print("Analyse sante architecture...")
    c.run("python scripts/analyze_architecture.py")
    print("\nRapport: reports/architecture-health.json")


@task(name="iso-docs")
def iso_docs(c: Context) -> None:
    """Update ISO documentation."""
    print("MAJ documentation ISO...")
    c.run("python scripts/update_iso_docs.py")
    print("\nDocumentation: docs/iso/IMPLEMENTATION_STATUS.md")


@task
def reports(c: Context) -> None:
    """Generate all reports (then git add)."""
    architecture(c)
    graphs(c)
    iso_docs(c)
    print("\n" + "=" * 44)
    print("  RAPPORTS GENERES")
    print("=" * 44)
    c.run("git status --porcelain reports/ graphs/ docs/iso/")
    print("\nExecutez maintenant:")
    print("  git add reports/ graphs/ docs/iso/")
    print("  git commit -m 'chore(reports): update generated reports'")
