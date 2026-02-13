"""MLflow PyFunc Wrapper pour ALICE Engine.

Ce wrapper permet de servir les modèles ALICE via MLflow Model Serving
pour déploiement sur Render (https://alice-engine.onrender.com).

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management (Model Serving API)
- ISO/IEC 5259:2024 - Data Quality for ML (Input Validation)
- ISO/IEC 27034 - Secure Coding (Input Sanitization)

Usage MLflow:
    # Enregistrer le modèle
    mlflow.pyfunc.log_model(
        artifact_path="alice_model",
        python_model=AliceModelWrapper(),
        artifacts={"model_path": "models/autogluon/..."}
    )

    # Servir le modèle
    mlflow models serve -m "models:/ALICE/Production" -p 5001

Render Deployment:
    # render.yaml
    services:
      - type: web
        name: alice-engine
        env: python
        buildCommand: pip install mlflow[extras]
        startCommand: mlflow models serve -m models/alice -p 8080

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

# Features: canonical source is scripts.training.features (DRY - ISO 24027)
from scripts.training.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AliceModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper pour modèles ALICE.

    Ce wrapper encapsule les modèles AutoGluon ou Baseline
    pour un déploiement unifié via MLflow serving.

    Attributes
    ----------
        model: Modèle chargé (AutoGluon ou sklearn-compatible)
        model_type: Type de modèle ("autogluon", "catboost", "xgboost", "lightgbm")
        encoders: Label encoders pour features catégorielles
        feature_names: Noms des features attendues
    """

    def __init__(self) -> None:
        """Initialise le wrapper (modèle chargé via load_context)."""
        self.model: Any = None
        self.model_type: str = "unknown"
        self.encoders: dict = {}
        self.feature_names: list[str] = []

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Charge le modèle et les artefacts.

        Args:
        ----
            context: Contexte MLflow avec chemins des artefacts
        """
        artifacts = context.artifacts

        # Charger le modèle selon le type
        model_path = artifacts.get("model_path", "")

        if "autogluon" in model_path.lower():
            self._load_autogluon(model_path)
        elif "catboost" in model_path.lower():
            self._load_catboost(model_path)
        elif "xgboost" in model_path.lower():
            self._load_xgboost(model_path)
        elif "lightgbm" in model_path.lower():
            self._load_lightgbm(model_path)
        else:
            self._load_generic(model_path)

        # Charger les encoders si disponibles
        encoders_path = artifacts.get("encoders_path")
        if encoders_path and Path(encoders_path).exists():
            import joblib

            self.encoders = joblib.load(encoders_path)
            logger.info(f"Loaded encoders from {encoders_path}")

        # Charger les feature names si disponibles
        features_path = artifacts.get("features_path")
        if features_path and Path(features_path).exists():
            with open(features_path) as f:
                self.feature_names = json.load(f)
            logger.info(f"Loaded {len(self.feature_names)} feature names")

        logger.info(f"Model loaded: type={self.model_type}")

    def _load_autogluon(self, model_path: str) -> None:
        """Charge un modèle AutoGluon."""
        from autogluon.tabular import TabularPredictor

        self.model = TabularPredictor.load(model_path)
        self.model_type = "autogluon"
        logger.info(f"Loaded AutoGluon model from {model_path}")

    def _load_catboost(self, model_path: str) -> None:
        """Charge un modèle CatBoost."""
        from catboost import CatBoostClassifier

        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self.model_type = "catboost"
        logger.info(f"Loaded CatBoost model from {model_path}")

    def _load_xgboost(self, model_path: str) -> None:
        """Charge un modèle XGBoost."""
        from xgboost import XGBClassifier

        self.model = XGBClassifier()
        self.model.load_model(model_path)
        self.model_type = "xgboost"
        logger.info(f"Loaded XGBoost model from {model_path}")

    def _load_lightgbm(self, model_path: str) -> None:
        """Charge un modèle LightGBM."""
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=model_path)
        self.model_type = "lightgbm"
        logger.info(f"Loaded LightGBM model from {model_path}")

    def _load_generic(self, model_path: str) -> None:
        """Charge un modèle générique via joblib."""
        import joblib

        self.model = joblib.load(model_path)
        self.model_type = "generic"
        logger.info(f"Loaded generic model from {model_path}")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,  # noqa: ARG002
        model_input: pd.DataFrame,
    ) -> np.ndarray:
        """Prédit les résultats pour les entrées données.

        Args:
        ----
            context: Contexte MLflow (non utilisé)
            model_input: DataFrame avec les features

        Returns:
        -------
            Array de probabilités (classe positive)
        """
        # Prétraitement
        X = self._preprocess(model_input)

        # Prédiction selon le type de modèle
        if self.model_type == "autogluon":
            proba = self.model.predict_proba(X).iloc[:, 1].values
        elif self.model_type == "lightgbm":
            proba = self.model.predict(X)
        elif hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[:, 1]
        else:
            proba = self.model.predict(X)

        return proba

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prétraite les features d'entrée.

        Args:
        ----
            df: DataFrame brut d'entrée

        Returns:
        -------
            DataFrame prétraité prêt pour prédiction
        """
        df = df.copy()

        # Créer diff_elo si manquant
        if "diff_elo" not in df.columns and "blanc_elo" in df.columns:
            df["diff_elo"] = df["blanc_elo"] - df.get("noir_elo", 0)

        # Features numériques
        X_parts = []
        num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        if num_cols:
            X_num = df[num_cols].copy()
            for col in num_cols:
                if X_num[col].isna().any():
                    X_num[col] = X_num[col].fillna(X_num[col].median())
            X_parts.append(X_num)

        # Features catégorielles
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        if cat_cols and self.encoders:
            cat_data = df[cat_cols].fillna("UNKNOWN").astype(str)
            for col in cat_cols:
                if col in self.encoders:
                    le = self.encoders[col]
                    known = set(le.classes_)
                    cat_data[col] = cat_data[col].apply(
                        lambda x, k=known: x if x in k else "UNKNOWN"
                    )
                    if "UNKNOWN" not in known:
                        le.classes_ = np.append(le.classes_, "UNKNOWN")
                    cat_data[col] = le.transform(cat_data[col])
            X_parts.append(cat_data.reset_index(drop=True))

        if X_parts:
            return pd.concat([p.reset_index(drop=True) for p in X_parts], axis=1)
        return df
