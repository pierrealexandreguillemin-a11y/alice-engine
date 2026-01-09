"""Main evaluation pipeline for model comparison.

This module orchestrates the complete evaluation workflow:
loading data, preparing features, training models, and reporting results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from scripts.evaluation.constants import CATEGORICAL_FEATURES
from scripts.evaluation.data import prepare_features
from scripts.evaluation.metrics import evaluate_model
from scripts.evaluation.trainers import train_catboost, train_lightgbm, train_xgboost

logger = logging.getLogger(__name__)


def run_evaluation(data_dir: Path, sample_size: int | None = None) -> None:
    """Pipeline complet d'evaluation.

    Args:
    ----
        data_dir: Directory containing train.parquet, valid.parquet, test.parquet
        sample_size: Optional sample size for faster testing
    """
    logger.info("=" * 60)
    logger.info("ALICE Engine - Evaluation ML")
    logger.info("CatBoost vs XGBoost vs LightGBM")
    logger.info("=" * 60)

    # Charger donnees
    logger.info("\n[1/4] Chargement donnees...")
    train = pd.read_parquet(data_dir / "train.parquet")
    valid = pd.read_parquet(data_dir / "valid.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")

    logger.info(f"  Train: {len(train):,} echiquiers")
    logger.info(f"  Valid: {len(valid):,} echiquiers")
    logger.info(f"  Test:  {len(test):,} echiquiers")

    # Sous-echantillonner si demande
    if sample_size and sample_size < len(train):
        logger.info(f"\n  Sous-echantillonnage train: {sample_size:,}")
        train = train.sample(n=sample_size, random_state=42)

    # Preparer features
    logger.info("\n[2/4] Preparation features...")
    X_train, y_train, encoders = prepare_features(train, fit_encoders=True)
    X_valid, y_valid, _ = prepare_features(valid, label_encoders=encoders)
    X_test, y_test, _ = prepare_features(test, label_encoders=encoders)

    logger.info(f"  Features: {list(X_train.columns)}")
    logger.info(f"  Train shape: {X_train.shape}")
    logger.info(f"  Target balance: {y_train.mean():.2%} victoires blancs")

    # Features categorielles pour CatBoost/LightGBM
    cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    # Entrainer modeles
    logger.info("\n[3/4] Entrainement modeles...")
    results = _train_all_models(X_train, y_train, X_valid, y_valid, X_test, y_test, cat_features)

    # Resultats
    _report_results(results, data_dir)


def _train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_features: list[str],
) -> list[dict]:
    """Train all models and collect results.

    Args:
    ----
        X_train: Training feature matrix
        y_train: Training target labels
        X_valid: Validation feature matrix
        y_valid: Validation target labels
        X_test: Test feature matrix
        y_test: Test target labels
        cat_features: List of categorical feature names

    Returns:
    -------
        List of result dictionaries for each model and dataset
    """
    results = []

    # CatBoost
    try:
        model_cb, time_cb = train_catboost(X_train, y_train, X_valid, y_valid, cat_features)
        metrics_cb_valid = evaluate_model(model_cb, X_valid, y_valid, "CatBoost")
        metrics_cb_test = evaluate_model(model_cb, X_test, y_test, "CatBoost")
        metrics_cb_valid["train_time_s"] = time_cb
        metrics_cb_valid["set"] = "valid"
        metrics_cb_test["train_time_s"] = time_cb
        metrics_cb_test["set"] = "test"
        results.extend([metrics_cb_valid, metrics_cb_test])
        logger.info(f"  CatBoost: OK (train={time_cb:.1f}s)")
    except Exception as e:
        logger.error(f"  CatBoost: FAILED - {e}")

    # XGBoost
    try:
        model_xgb, time_xgb = train_xgboost(X_train, y_train, X_valid, y_valid)
        metrics_xgb_valid = evaluate_model(model_xgb, X_valid, y_valid, "XGBoost")
        metrics_xgb_test = evaluate_model(model_xgb, X_test, y_test, "XGBoost")
        metrics_xgb_valid["train_time_s"] = time_xgb
        metrics_xgb_valid["set"] = "valid"
        metrics_xgb_test["train_time_s"] = time_xgb
        metrics_xgb_test["set"] = "test"
        results.extend([metrics_xgb_valid, metrics_xgb_test])
        logger.info(f"  XGBoost: OK (train={time_xgb:.1f}s)")
    except Exception as e:
        logger.error(f"  XGBoost: FAILED - {e}")

    # LightGBM
    try:
        model_lgb, time_lgb = train_lightgbm(X_train, y_train, X_valid, y_valid, cat_features)
        metrics_lgb_valid = evaluate_model(model_lgb, X_valid, y_valid, "LightGBM")
        metrics_lgb_test = evaluate_model(model_lgb, X_test, y_test, "LightGBM")
        metrics_lgb_valid["train_time_s"] = time_lgb
        metrics_lgb_valid["set"] = "valid"
        metrics_lgb_test["train_time_s"] = time_lgb
        metrics_lgb_test["set"] = "test"
        results.extend([metrics_lgb_valid, metrics_lgb_test])
        logger.info(f"  LightGBM: OK (train={time_lgb:.1f}s)")
    except Exception as e:
        logger.error(f"  LightGBM: FAILED - {e}")

    return results


def _report_results(results: list[dict], data_dir: Path) -> None:
    """Report and save evaluation results.

    Args:
    ----
        results: List of result dictionaries
        data_dir: Directory for saving output
    """
    logger.info("\n[4/4] Resultats...")
    df_results = pd.DataFrame(results)

    # Afficher tableau comparatif
    logger.info("\n" + "=" * 60)
    logger.info("RESULTATS COMPARATIFS")
    logger.info("=" * 60)

    # Par set
    for dataset in ["valid", "test"]:
        df_set = df_results[df_results["set"] == dataset]
        logger.info(f"\n--- {dataset.upper()} SET ---")
        for _, row in df_set.iterrows():
            logger.info(
                f"  {row['model']:10s} | AUC: {row['auc_roc']:.4f} | "
                f"Acc: {row['accuracy']:.4f} | "
                f"Train: {row['train_time_s']:.1f}s | "
                f"Infer: {row['inference_time_ms']:.1f}ms"
            )

    # Meilleur modele
    best = df_results[df_results["set"] == "test"].sort_values("auc_roc", ascending=False).iloc[0]
    logger.info("\n" + "=" * 60)
    logger.info(f"MEILLEUR MODELE: {best['model']}")
    logger.info(f"  AUC-ROC (test): {best['auc_roc']:.4f}")
    logger.info(f"  Accuracy (test): {best['accuracy']:.4f}")
    logger.info("=" * 60)

    # Sauvegarder resultats
    output_path = data_dir.parent / "ml_evaluation_results.csv"
    df_results.to_csv(output_path, index=False)
    logger.info(f"\nResultats sauvegardes: {output_path}")
