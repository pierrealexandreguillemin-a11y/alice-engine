"""Tasks ML lifecycle - ISO 42001/5259/25059."""

from invoke import Context, task


@task(name="parse-data")
def parse_data(c: Context) -> None:
    """Parse FFE dataset (ISO 5259)."""
    print("Parsing dataset FFE (ISO 5259)...")
    c.run("python -m scripts.parse_dataset")
    print("\nDataset parse dans: data/")


@task
def features(c: Context) -> None:
    """Run feature engineering pipeline (ISO 5259)."""
    print("Feature engineering (ISO 5259)...")
    c.run("python -m scripts.feature_engineering")
    print("\nFeatures generees")


@task
def train(c: Context) -> None:
    """Train ML models (ISO 42001)."""
    print("Entrainement modeles (ISO 42001)...")
    c.run("python -m scripts.train_models_parallel")
    print("\nModeles entraines dans: models/")


@task
def evaluate(c: Context) -> None:
    """Evaluate and benchmark models (ISO 25059)."""
    print("Evaluation modeles (ISO 25059)...")
    c.run("python -m scripts.evaluate_models")
    print("\nRapport: reports/evaluation/")


@task
def ensemble(c: Context) -> None:
    """Build stacking ensemble (ISO 42001)."""
    print("Stacking ensemble (ISO 42001)...")
    c.run("python -m scripts.ensemble_stacking")
    print("\nEnsemble cree")


@task(name="ml-pipeline")
def ml_pipeline(c: Context) -> None:
    """Run complete ML pipeline (parse -> train -> eval)."""
    parse_data(c)
    features(c)
    train(c)
    evaluate(c)
    ensemble(c)
    print("\n" + "=" * 44)
    print("  PIPELINE ML COMPLET TERMINE")
    print("=" * 44)
