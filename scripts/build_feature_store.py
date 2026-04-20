"""Build feature_store/joueur_features.parquet from train.parquet aggregates.

Plan 3 Pre-Task 1 fix : FeatureStore was stub (only returned 3 cols).
Model trained on 201 features → XGBoost rejected stub output.

This script:
1. Loads data/features/train.parquet (1.1M rows × 147 cols)
2. For each player, computes mean of his "as blanc" rows (blanc_* features)
3. Outputs data/feature_store/joueur_features.parquet (per-player features)

Training means (all players, all games) stored in same parquet under key 'TRAIN_MEAN_ALL'
→ used as fallback for unknown players at inference.

ISO 5259 : data lineage via SHA-256 of train.parquet.
ISO 42001 : model feature schema preserved (201 feature_names from LightGBM.txt).

Usage : python scripts/build_feature_store.py
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_PATH = Path("data/features/train.parquet")
LIGHTGBM_MODEL = Path("models/cache/LightGBM.txt")
OUT_DIR = Path("data/feature_store")
OUT_PATH = OUT_DIR / "joueur_features.parquet"
META_PATH = OUT_DIR / "lineage.json"

MEAN_ALL_KEY = "__TRAIN_MEAN_ALL__"


def extract_model_feature_names(lgb_path: Path) -> list[str]:
    """Read feature_names from LightGBM.txt model file."""
    with lgb_path.open() as f:
        for line in f:
            if line.startswith("feature_names="):
                return line[len("feature_names=") :].strip().split()
    raise RuntimeError(f"feature_names not found in {lgb_path}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_per_player_features(train_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Aggregate per-player feature means from training data.

    Strategy:
    - For each unique blanc_nom : compute mean of all "*_blanc" features from his rows
    - Strip "_blanc" suffix to get canonical feature name per player
    - Store in single DataFrame indexed by joueur_nom
    """
    # Cols with _blanc suffix in feature_names
    blanc_cols = [c for c in feature_names if c.endswith("_blanc")]

    # Group by blanc_nom
    available_blanc = [c for c in blanc_cols if c in train_df.columns]
    grouped = train_df.groupby("blanc_nom")[available_blanc].mean(numeric_only=True)

    # Rename cols : strip _blanc → canonical
    rename = {old: old.removesuffix("_blanc") for old in available_blanc}
    grouped = grouped.rename(columns=rename)
    grouped.index.name = "joueur_nom"
    logger.info(
        "Per-player features built: %d players, %d cols", len(grouped), len(grouped.columns)
    )
    return grouped


def build_training_mean_row(train_df: pd.DataFrame, feature_names: list[str]) -> pd.Series:
    """Training-time mean of each feature_name (fallback for unknowns).

    Numeric cols : mean from train.parquet.
    Missing cols (categoricals encoded OR derived) : 0.0 (equivalent encoded default).
    """
    available_numeric = [
        c
        for c in feature_names
        if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    means_numeric = train_df[available_numeric].mean(numeric_only=True)
    # Fill non-numeric or missing with 0.0 (encoded default)
    full_series: dict[str, float] = {}
    for c in feature_names:
        if c in means_numeric.index:
            full_series[c] = float(means_numeric[c])
        else:
            full_series[c] = 0.0
    return pd.Series(full_series)[feature_names]  # preserve order


def build_feature_store() -> None:
    """Main : build joueur_features.parquet + lineage.json."""
    logging.basicConfig(level=logging.INFO)

    if not TRAIN_PATH.exists():
        msg = f"{TRAIN_PATH} missing. Run data refresh pipeline first."
        raise FileNotFoundError(msg)
    if not LIGHTGBM_MODEL.exists():
        msg = f"{LIGHTGBM_MODEL} missing. Run model loader first (HF Hub)."
        raise FileNotFoundError(msg)

    feature_names = extract_model_feature_names(LIGHTGBM_MODEL)
    logger.info("Model expects %d features", len(feature_names))

    train_df = pd.read_parquet(TRAIN_PATH)
    logger.info("Loaded train : %d rows × %d cols", len(train_df), len(train_df.columns))

    player_features = build_per_player_features(train_df, feature_names)
    training_mean = build_training_mean_row(train_df, feature_names)

    # Store MEAN_ALL row as an additional row with index=MEAN_ALL_KEY
    mean_row_df = pd.DataFrame([training_mean.to_dict()])
    # Only keep cols that are per-player (canonical, not _blanc suffix)
    # For MEAN_ALL, we keep the full 201 feature_names
    mean_row_df.index = pd.Index([MEAN_ALL_KEY], name="joueur_nom")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save two files : player-level aggregate + training mean (full 201 cols)
    player_features.to_parquet(OUT_PATH)
    mean_row_df.to_parquet(OUT_DIR / "training_mean.parquet")

    # Lineage metadata
    import json

    lineage = {
        "train_parquet_sha256": _sha256(TRAIN_PATH),
        "lightgbm_sha256": _sha256(LIGHTGBM_MODEL),
        "n_players": len(player_features),
        "n_feature_names": len(feature_names),
        "n_player_cols": len(player_features.columns),
        "output_joueur_features": str(OUT_PATH),
        "output_training_mean": str(OUT_DIR / "training_mean.parquet"),
    }
    META_PATH.write_text(json.dumps(lineage, indent=2))

    logger.info(
        "Feature store built : %d players × %d cols, training_mean=%d cols",
        len(player_features),
        len(player_features.columns),
        len(training_mean),
    )


if __name__ == "__main__":
    build_feature_store()
