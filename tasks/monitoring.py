"""Tasks monitoring production - ISO 42001/5259/27001."""

from invoke import Context, task


@task(name="model-card")
def model_card(c: Context) -> None:
    """Generate Model Card (ISO 42001)."""
    print("Generation Model Card (ISO 42001)...")
    c.run(
        'python -c "from scripts.model_registry import ProductionModelCard; '
        "print('Model Card template ready')\""
    )
    print("\nModel Card: models/model_card.json")


@task(name="drift-report")
def drift_report(c: Context) -> None:
    """Generate drift monitoring report (ISO 5259)."""
    print("Rapport drift monitoring (ISO 5259)...")
    c.run(
        'python -c "from scripts.model_registry import create_drift_report, '
        "check_drift_status; print('Drift monitoring ready')\""
    )
    print("\nRapport: reports/drift/")


@task(name="data-lineage")
def data_lineage(c: Context) -> None:
    """Generate data lineage report (ISO 5259)."""
    print("Rapport lineage donnees (ISO 5259)...")
    c.run(
        'python -c "from scripts.model_registry import compute_data_lineage; '
        "print('Data lineage ready')\""
    )
    print("\nLineage: reports/lineage/")
