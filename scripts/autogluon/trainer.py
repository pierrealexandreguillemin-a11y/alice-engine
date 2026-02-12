"""Module: scripts/autogluon/trainer.py - AutoGluon Training Pipeline.

Document ID: ALICE-MOD-AUTOGLUON-TRAINER-001
Version: 1.0.0

Ce module implemente l'entrainement AutoML avec AutoGluon
pour comparaison avec l'ensemble CatBoost/XGBoost/LightGBM actuel.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (Tracabilite)
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from scripts.autogluon.config import AutoGluonConfig, load_autogluon_config
from scripts.autogluon.trainer_types import AutoGluonTrainingResult

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor

logger = logging.getLogger(__name__)

__all__ = ["AutoGluonTrainer", "AutoGluonTrainingResult", "train_autogluon"]


class AutoGluonTrainer:
    """Pipeline d'entrainement AutoGluon avec tracking MLflow.

    ISO 42001: Tracabilite complete du processus ML.

    Attributes
    ----------
        config: Configuration AutoGluon
        save_path: Chemin de sauvegarde des modeles
    """

    def __init__(
        self,
        config: AutoGluonConfig | None = None,
        save_path: str | Path = "models/autogluon",
    ) -> None:
        """Initialise le trainer.

        Args:
        ----
            config: Configuration AutoGluon (defaut: charge depuis YAML)
            save_path: Chemin de sauvegarde des modeles
        """
        self.config = config or load_autogluon_config()
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.predictor: TabularPredictor | None = None
        self.run_id: str | None = None

    def _init_training_run(self) -> Path:
        """Initialize run_id and model path for training.

        Returns
        -------
            Path to the model output directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"autogluon_{timestamp}"
        return self.save_path / self.run_id

    def _fit_predictor(
        self,
        train_data: pd.DataFrame,
        label: str,
        tuning_data: pd.DataFrame | None,
        model_path: Path,
    ) -> float:
        """Create and fit the TabularPredictor.

        Returns
        -------
            Training time in seconds.
        """
        from autogluon.tabular import TabularPredictor

        start_time = time.time()
        self.predictor = TabularPredictor(
            label=label,
            eval_metric=self.config.eval_metric,
            path=str(model_path),
        )
        self.predictor.fit(
            train_data=train_data,
            tuning_data=tuning_data,
            **self.config.to_fit_kwargs(),
        )
        return time.time() - start_time

    def _extract_leaderboard_metrics(self) -> tuple[pd.DataFrame, str, dict[str, float]]:
        """Extract leaderboard and metrics from trained predictor.

        Returns
        -------
            Tuple of (leaderboard, best_model_name, metrics_dict).
        """
        leaderboard = self.predictor.leaderboard(silent=True)
        best_model = leaderboard.iloc[0]["model"]
        metrics = {
            "score_val": float(leaderboard.iloc[0]["score_val"]),
            "pred_time_val": float(leaderboard.iloc[0]["pred_time_val"]),
            "fit_time": float(leaderboard.iloc[0]["fit_time"]),
            "num_models": len(leaderboard),
        }
        return leaderboard, best_model, metrics

    def train(
        self,
        train_data: pd.DataFrame,
        label: str,
        tuning_data: pd.DataFrame | None = None,
    ) -> AutoGluonTrainingResult:
        """Entraine le modele AutoGluon avec tracking MLflow.

        Args:
        ----
            train_data: Donnees d'entrainement
            label: Colonne cible
            tuning_data: Donnees de validation (optionnel)

        Returns:
        -------
            AutoGluonTrainingResult avec predictor et metriques

        ISO 42001: Logging complet pour reproductibilite.
        """
        model_path = self._init_training_run()
        data_hash = self._compute_data_hash(train_data)

        logger.info(f"[AutoGluon] Starting training with preset='{self.config.presets}'")
        logger.info(f"[AutoGluon] Model path: {model_path}")
        logger.info(f"[AutoGluon] Data hash: {data_hash[:16]}...")

        train_time = self._fit_predictor(train_data, label, tuning_data, model_path)
        leaderboard, best_model, metrics = self._extract_leaderboard_metrics()

        logger.info(f"[AutoGluon] Training completed in {train_time:.1f}s")
        logger.info(f"[AutoGluon] Best model: {best_model}")
        logger.info(f"[AutoGluon] Best score: {metrics['score_val']:.4f}")

        self._log_to_mlflow(train_data, metrics, leaderboard)

        return AutoGluonTrainingResult(
            predictor=self.predictor,
            train_time=train_time,
            leaderboard=leaderboard,
            best_model=best_model,
            model_path=model_path,
            data_hash=data_hash,
            config=self.config,
            metrics=metrics,
        )

    def evaluate(
        self,
        test_data: pd.DataFrame,
        label: str | None = None,
    ) -> dict[str, float]:
        """Evalue le modele sur un jeu de test.

        Args:
        ----
            test_data: Donnees de test
            label: Colonne cible (optionnel si dans predictor)

        Returns:
        -------
            Dict des metriques d'evaluation

        Raises:
        ------
            ValueError: Si le modele n'est pas entraine
        """
        if self.predictor is None:
            msg = "Model not trained. Call train() first."
            raise ValueError(msg)

        return self.predictor.evaluate(test_data)

    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """Calcule un hash SHA256 des donnees pour tracabilite.

        ISO 42001: Tracabilite des donnees d'entrainement.
        """
        data_bytes = pd.util.hash_pandas_object(data).values.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def _log_to_mlflow(
        self,
        train_data: pd.DataFrame,
        metrics: dict[str, float],
        leaderboard: pd.DataFrame,
    ) -> None:
        """Log les resultats vers MLflow si disponible.

        ISO 42001: Integration tracking experiments.
        """
        try:
            import mlflow

            mlflow.set_experiment("alice_autogluon")

            with mlflow.start_run(run_name=self.run_id):
                # Log parametres
                mlflow.log_params(
                    {
                        "preset": self.config.presets,
                        "time_limit": self.config.time_limit,
                        "eval_metric": self.config.eval_metric,
                        "num_bag_folds": self.config.num_bag_folds,
                        "num_stack_levels": self.config.num_stack_levels,
                        "train_samples": len(train_data),
                        "features": len(train_data.columns) - 1,
                    }
                )

                # Log metriques
                mlflow.log_metrics(metrics)

                # Log leaderboard comme artifact
                leaderboard_path = self.save_path / self.run_id / "leaderboard.csv"
                leaderboard.to_csv(leaderboard_path, index=False)
                mlflow.log_artifact(str(leaderboard_path))

                logger.info(f"[MLflow] Logged run: {self.run_id}")

        except ImportError:
            logger.debug("MLflow not available, skipping tracking")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


def train_autogluon(
    train_data: pd.DataFrame,
    label: str,
    valid_data: pd.DataFrame | None = None,
    config: AutoGluonConfig | None = None,
    output_dir: Path | None = None,
) -> AutoGluonTrainingResult:
    """Fonction principale pour entrainer AutoGluon sur ALICE.

    Args:
    ----
        train_data: Donnees d'entrainement
        label: Colonne cible
        valid_data: Donnees de validation (optionnel)
        config: Configuration AutoGluon
        output_dir: Repertoire de sortie

    Returns:
    -------
        AutoGluonTrainingResult avec predictor et metriques

    ISO 42001: Point d'entree documente.
    """
    save_path = output_dir or Path("models/autogluon")
    trainer = AutoGluonTrainer(config=config, save_path=save_path)

    return trainer.train(
        train_data=train_data,
        label=label,
        tuning_data=valid_data,
    )
