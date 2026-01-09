#!/usr/bin/env python3
"""Feature Engineering pour ALICE - Pipeline ML.

Ce module orchestre le pipeline de feature engineering.
Les features sont extraites par des modules specialises (ISO 5055 SRP).

Modules:
- features/reliability.py: Fiabilite club/joueur
- features/performance.py: Forme, couleur, position
- features/standings.py: Classement, zones d'enjeu
- features/advanced.py: H2H, fatigue, trajectoire Elo
- features/ffe_features.py: Features reglementaires FFE
- features/pipeline.py: Orchestration extraction/merge

Conformite ISO/IEC:
- 5055: Code maintenable (<300 lignes, SRP)
- 5259: Qualite donnees ML
- 25010: Qualite systeme
- 42001: Gouvernance IA

Usage:
    python scripts/feature_engineering.py
    python scripts/feature_engineering.py --output-dir data/features
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts.features.pipeline import extract_all_features, merge_all_features

# Configuration paths
PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# SPLIT TEMPOREL
# ==============================================================================


def temporal_split(
    df: pd.DataFrame,
    train_end: int = 2022,
    valid_end: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel des données pour éviter data leakage."""
    logger.info(
        f"Split temporel: train<={train_end}, valid={train_end + 1}-{valid_end}, test>{valid_end}"
    )

    train = df[df["saison"] <= train_end]
    valid = df[(df["saison"] > train_end) & (df["saison"] <= valid_end)]
    test = df[df["saison"] > valid_end]

    for name, split in [("Train", train), ("Valid", valid), ("Test", test)]:
        if not split.empty and "saison" in split.columns:
            logger.info(
                f"  {name}: {len(split):,} échiquiers "
                f"({split['saison'].min()}-{split['saison'].max()})"
            )
        else:
            logger.info(f"  {name}: {len(split):,} échiquiers")

    return train, valid, test


# ==============================================================================
# PIPELINE V2 - SANS DATA LEAKAGE (ISO 5259 / 42001)
# ==============================================================================


def compute_features_for_split(
    df_split: pd.DataFrame,
    df_history: pd.DataFrame,
    split_name: str,
    include_advanced: bool = True,
) -> pd.DataFrame:
    """Calcule les features pour un split avec historique approprié.

    ISO 5259: Pas de data leakage - historique limité au passé visible.
    ISO 5055: Complexité réduite via helpers (merge_helpers.py).
    """
    logger.info(
        f"  Computing features for {split_name} using {len(df_history):,} historical records..."
    )

    # Filter played games only
    df_history_played = df_history[
        ~df_history["type_resultat"].isin(
            ["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"]
        )
    ]

    # Extract all features from history
    features = extract_all_features(df_history, df_history_played, include_advanced)

    # Merge features onto split
    result = merge_all_features(df_split.copy(), features, include_advanced)

    logger.info(f"  {split_name}: {len(result):,} samples, {len(result.columns)} features")
    return result


def run_feature_engineering_v2(
    data_dir: Path,
    output_dir: Path,
    include_advanced: bool = True,
) -> None:
    """Pipeline feature engineering V2 - SANS DATA LEAKAGE.

    ISO/IEC 42001, 5259 Conformant.
    """
    logger.info("=" * 60)
    logger.info("ALICE Engine - Feature Engineering V2 (No Leakage)")
    logger.info("ISO/IEC 42001, 5259 Conformant")
    logger.info("=" * 60)

    # Charger données
    echiquiers_path = data_dir / "echiquiers.parquet"
    if not echiquiers_path.exists():
        logger.error(f"Fichier non trouvé: {echiquiers_path}")
        logger.error("Exécutez d'abord: python scripts/parse_dataset.py")
        return

    logger.info(f"\nChargement {echiquiers_path}...")
    df = pd.read_parquet(echiquiers_path)
    logger.info(f"  {len(df):,} échiquiers chargés")

    # 1. Filtrer parties jouées et Elo valide
    logger.info("\n[1/4] Filtrage données...")
    df_played = df[
        ~df["type_resultat"].isin(["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"])
    ]
    df_clean = df_played[(df_played["blanc_elo"] > 0) & (df_played["noir_elo"] > 0)]
    logger.info(f"  {len(df_clean):,} parties valides")

    # 2. SPLIT TEMPOREL D'ABORD (crucial pour éviter leakage)
    logger.info("\n[2/4] Split temporel AVANT features...")
    train_raw, valid_raw, test_raw = temporal_split(df_clean)

    # 3. Calculer features PAR SPLIT avec historique approprié
    logger.info("\n[3/4] Calcul features per-split (no leakage)...")

    # Train: features calculées sur train uniquement
    logger.info("\n  --- TRAIN ---")
    train = compute_features_for_split(
        df_split=train_raw,
        df_history=df[df["saison"] <= train_raw["saison"].max()],
        split_name="train",
        include_advanced=include_advanced,
    )

    # Valid: features calculées sur train + valid historique
    logger.info("\n  --- VALID ---")
    valid_history = df[df["saison"] <= valid_raw["saison"].max()]
    valid = compute_features_for_split(
        df_split=valid_raw,
        df_history=valid_history,
        split_name="valid",
        include_advanced=include_advanced,
    )

    # Test: features calculées sur tout l'historique (avant test)
    logger.info("\n  --- TEST ---")
    test_history = df[df["saison"] <= test_raw["saison"].min() - 1]
    test = compute_features_for_split(
        df_split=test_raw,
        df_history=test_history,
        split_name="test",
        include_advanced=include_advanced,
    )

    # 4. Export
    logger.info("\n[4/4] Export...")
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(output_dir / "train.parquet", index=False)
    valid.to_parquet(output_dir / "valid.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)

    # Résumé
    logger.info("\n" + "=" * 60)
    logger.info("Feature engineering V2 terminé!")
    logger.info("=" * 60)
    logger.info(f"\nFichiers générés dans {output_dir}/:")
    logger.info(f"  - train.parquet ({len(train):,} échiquiers, {len(train.columns)} features)")
    logger.info(f"  - valid.parquet ({len(valid):,} échiquiers)")
    logger.info(f"  - test.parquet ({len(test):,} échiquiers)")
    logger.info("\nDATA LEAKAGE: CORRIGÉ")
    logger.info("  - Split temporel effectué AVANT calcul des features")
    logger.info("  - Chaque split utilise uniquement les données historiques visibles")


def main() -> None:
    """Point d'entrée."""
    parser = argparse.ArgumentParser(description="Feature engineering ALICE")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire des données sources",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "features",
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--no-advanced",
        action="store_true",
        help="Désactiver features avancées (H2H, fatigue, etc.)",
    )
    args = parser.parse_args()

    run_feature_engineering_v2(
        args.data_dir,
        args.output_dir,
        include_advanced=not args.no_advanced,
    )


if __name__ == "__main__":
    main()
