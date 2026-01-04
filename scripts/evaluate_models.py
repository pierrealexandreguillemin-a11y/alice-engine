#!/usr/bin/env python3
"""
Evaluation comparative CatBoost vs XGBoost vs LightGBM pour ALICE.

Ce script compare les 3 modeles de gradient boosting sur le dataset
prepare pour identifier le meilleur candidat pour ALI.

Metriques evaluees:
- AUC-ROC (qualite des probabilites)
- Accuracy (precision globale)
- Temps d'entrainement
- Temps d'inference

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --sample 100000  # Sous-echantillon
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Configuration paths
PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "features"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# PREPARATION DES DONNEES
# ==============================================================================

# Features numeriques
NUMERIC_FEATURES = [
    "blanc_elo",
    "noir_elo",
    "diff_elo",
    "echiquier",
    "niveau",
    "ronde",
]

# Features categorielles
CATEGORICAL_FEATURES = [
    "type_competition",
    "division",
    "ligue_code",
    "blanc_titre",
    "noir_titre",
    "jour_semaine",
]


def prepare_features(
    df: pd.DataFrame,
    label_encoders: dict | None = None,
    fit_encoders: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Prepare les features pour l'entrainement.

    Args:
        df: DataFrame brut
        label_encoders: encodeurs existants (pour valid/test)
        fit_encoders: True pour train, False pour valid/test

    Returns:
        X, y, label_encoders
    """
    df = df.copy()

    # Target: resultat blanc (0=defaite, 0.5=nulle, 1=victoire)
    # Convertir en classification binaire: victoire (1) vs non-victoire (0)
    y = (df["resultat_blanc"] == 1.0).astype(int)

    # Features numeriques
    X_numeric = df[NUMERIC_FEATURES].fillna(0)

    # Features categorielles - encodage
    if label_encoders is None:
        label_encoders = {}

    X_cat_encoded = pd.DataFrame()
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue

        df[col] = df[col].fillna("UNKNOWN").astype(str)

        if fit_encoders:
            le = LabelEncoder()
            X_cat_encoded[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is None:
                continue
            # Gerer les valeurs inconnues (capture via default arg to avoid B023)
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x, kc=known_classes: x if x in kc else "UNKNOWN")
            if "UNKNOWN" not in known_classes:
                le.classes_ = np.append(le.classes_, "UNKNOWN")
            X_cat_encoded[col] = le.transform(df[col])

    # Combiner
    X = pd.concat([X_numeric.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1)

    return X, y, label_encoders


# ==============================================================================
# ENTRAINEMENT ET EVALUATION
# ==============================================================================


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
) -> tuple[object, float]:
    """Entraine CatBoost et retourne le modele + temps."""
    from catboost import CatBoostClassifier

    logger.info("  Training CatBoost...")

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        early_stopping_rounds=50,
        eval_metric="AUC",
        verbose=100,
        random_seed=42,
    )

    start = time.time()
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=100)
    train_time = time.time() - start

    return model, train_time


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[object, float]:
    """Entraine XGBoost et retourne le modele + temps."""
    from xgboost import XGBClassifier

    logger.info("  Training XGBoost...")

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="auc",
        verbosity=1,
        random_state=42,
    )

    start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100,
    )
    train_time = time.time() - start

    return model, train_time


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: list[str],
) -> tuple[object, float]:
    """Entraine LightGBM et retourne le modele + temps."""
    import lightgbm as lgb

    logger.info("  Training LightGBM...")

    # Convertir cat_features en indices
    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        categorical_feature=cat_indices,
        early_stopping_rounds=50,
        verbose=100,
        random_state=42,
    )

    start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
    )
    train_time = time.time() - start

    return model, train_time


def evaluate_model(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
) -> dict:
    """Evalue un modele et retourne les metriques."""
    start = time.time()
    y_pred_proba = model.predict_proba(X)[:, 1]
    inference_time = time.time() - start

    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y, y_pred_proba)
    acc = accuracy_score(y, y_pred)

    return {
        "model": name,
        "auc_roc": auc,
        "accuracy": acc,
        "inference_time_ms": inference_time * 1000,
        "samples": len(y),
    }


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================


def run_evaluation(data_dir: Path, sample_size: int | None = None) -> None:
    """Pipeline complet d'evaluation."""
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

    # Resultats
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


def main() -> None:
    """Point d'entree."""
    parser = argparse.ArgumentParser(description="Evaluation ML ALICE")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Repertoire des donnees features",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sous-echantillon pour tests rapides",
    )
    args = parser.parse_args()

    run_evaluation(args.data_dir, args.sample)


if __name__ == "__main__":
    main()
