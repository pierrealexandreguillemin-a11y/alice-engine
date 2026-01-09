"""Constants for model evaluation.

This module contains feature definitions and configuration constants
used throughout the evaluation pipeline.
"""

from __future__ import annotations

from pathlib import Path

# Configuration paths
PROJECT_DIR = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "features"

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

# Model training defaults
DEFAULT_ITERATIONS = 500
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_DEPTH = 6
DEFAULT_EARLY_STOPPING = 50
DEFAULT_RANDOM_SEED = 42
