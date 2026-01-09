"""Tracking MLflow pour l'entraînement - ISO 5055.

Ce module contient les fonctions d'intégration MLflow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.ml_types import ModelResults

logger = logging.getLogger(__name__)


def setup_mlflow(config: dict[str, object]) -> bool:
    """Configure MLflow si disponible."""
    try:
        import mlflow

        mlflow_config = config.get("mlflow", {})
        if not isinstance(mlflow_config, dict):
            mlflow_config = {}

        experiment_name = mlflow_config.get("experiment_name", "alice-ml-training")
        tracking_uri = mlflow_config.get("tracking_uri", "./mlruns")

        if not isinstance(experiment_name, str):
            experiment_name = "alice-ml-training"
        if not isinstance(tracking_uri, str):
            tracking_uri = "./mlruns"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracking: {tracking_uri} / {experiment_name}")
        return True
    except ImportError:
        logger.warning("MLflow not installed, skipping experiment tracking")
        return False


def log_to_mlflow(
    results: ModelResults,
    config: dict[str, object],
) -> None:
    """Log les resultats dans MLflow."""
    try:
        import mlflow

        with mlflow.start_run(run_name="parallel_training"):
            _log_global_config(config)
            _log_model_results(results, config)
            logger.info("Results logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


def _log_global_config(config: dict[str, object]) -> None:
    """Log la configuration globale."""
    import mlflow

    mlflow.log_param("parallel_workers", 3)
    global_config = config.get("global", {})
    if isinstance(global_config, dict):
        mlflow.log_param("random_seed", global_config.get("random_seed", 42))


def _log_model_results(results: ModelResults, config: dict[str, object]) -> None:
    """Log les résultats de chaque modèle."""
    import mlflow

    for name, result in results.items():
        if result.model is None:
            continue

        with mlflow.start_run(run_name=name, nested=True):
            _log_model_params(config, name)
            _log_model_metrics(result)


def _log_model_params(config: dict[str, object], name: str) -> None:
    """Log les paramètres d'un modèle."""
    import mlflow

    model_config = config.get(name.lower(), {})
    if isinstance(model_config, dict):
        for key, value in model_config.items():
            if isinstance(value, int | float | str | bool):
                mlflow.log_param(key, value)


def _log_model_metrics(result: object) -> None:
    """Log les métriques d'un modèle."""
    import mlflow

    metrics_dict = result.metrics.to_dict()  # type: ignore[union-attr]
    for metric_name, metric_value in metrics_dict.items():
        if isinstance(metric_value, int | float):
            mlflow.log_metric(metric_name, metric_value)
