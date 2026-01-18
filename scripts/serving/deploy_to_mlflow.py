#!/usr/bin/env python3
"""Déploiement des modèles ALICE vers MLflow Registry.

Ce script enregistre les modèles entraînés (AutoGluon ou Baseline)
dans MLflow Model Registry pour déploiement sur Render.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management (Model Registry)
- ISO/IEC 5055:2021 - Code Quality

Usage:
    # Enregistrer le modèle AutoGluon
    python -m scripts.serving.deploy_to_mlflow --model autogluon

    # Enregistrer un baseline
    python -m scripts.serving.deploy_to_mlflow --model catboost

    # Générer config Render
    python -m scripts.serving.deploy_to_mlflow --model autogluon --render

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.serving.pyfunc_wrapper import (
    create_render_deployment_config,
    register_model_to_mlflow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent.parent


def main() -> None:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Deploy ALICE model to MLflow")
    parser.add_argument(
        "--model",
        type=str,
        choices=["autogluon", "catboost", "xgboost", "lightgbm"],
        required=True,
        help="Model type to deploy",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Custom model path (auto-detected if not specified)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Generate Render deployment config",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ALICE",
        help="Model name in MLflow Registry",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ALICE ENGINE - MODEL DEPLOYMENT")
    logger.info("MLflow Registry + Render Configuration")
    logger.info("=" * 60)

    # Déterminer le chemin du modèle
    model_path, encoders_path = _get_model_paths(args.model, args.model_path)

    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Train the model first:")
        if args.model == "autogluon":
            logger.info("  python -m scripts.autogluon.trainer")
        else:
            logger.info(f"  python -m scripts.baseline.{args.model}_baseline")
        return

    # Enregistrer dans MLflow
    logger.info(f"\n[1/2] Registering {args.model} model to MLflow...")
    model_uri = register_model_to_mlflow(
        model_path=model_path,
        model_name=args.model_name,
        encoders_path=encoders_path,
    )
    logger.info(f"Model registered: {model_uri}")

    # Générer config Render si demandé
    if args.render:
        logger.info("\n[2/2] Generating Render deployment config...")
        render_path = PROJECT_DIR / "render.yaml"
        create_render_deployment_config(model_uri, str(render_path))
        logger.info(f"Config created: {render_path}")
        logger.info("\nTo deploy to Render:")
        logger.info("  1. Push render.yaml to your repository")
        logger.info("  2. Connect repository to Render")
        logger.info("  3. Deploy from Render dashboard")

    logger.info("\n" + "=" * 60)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info("=" * 60)


def _get_model_paths(model_type: str, custom_path: Path | None) -> tuple[str, str | None]:
    """Détermine les chemins du modèle et des encoders."""
    if custom_path:
        return str(custom_path), None

    if model_type == "autogluon":
        return _get_autogluon_path()

    return _get_baseline_path(model_type)


def _get_autogluon_path() -> tuple[str, str | None]:
    """Retourne le chemin du dernier modèle AutoGluon."""
    ag_dir = PROJECT_DIR / "models" / "autogluon"
    if ag_dir.exists():
        models = sorted(ag_dir.glob("autogluon_*"), reverse=True)
        if models:
            return str(models[0]), None
    return str(ag_dir / "autogluon_latest"), None


def _get_baseline_path(model_type: str) -> tuple[str, str | None]:
    """Retourne le chemin d'un modèle baseline."""
    baseline_dir = PROJECT_DIR / "models" / "baseline"
    paths = {
        "catboost": ("catboost_baseline.cbm", "catboost_encoders.pkl"),
        "xgboost": ("xgboost_baseline.json", "xgboost_encoders.pkl"),
        "lightgbm": ("lightgbm_baseline.txt", "lightgbm_encoders.pkl"),
    }
    model_file, encoder_file = paths.get(model_type, (f"{model_type}_baseline", None))
    encoder_path = str(baseline_dir / encoder_file) if encoder_file else None
    return str(baseline_dir / model_file), encoder_path


if __name__ == "__main__":
    main()
