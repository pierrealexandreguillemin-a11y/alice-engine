"""Constants for model evaluation.

This module contains feature definitions and configuration constants
used throughout the evaluation pipeline.
"""

from __future__ import annotations

from pathlib import Path

# Features: canonical source is scripts.training.features (DRY - ISO 24027)
from scripts.training.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

__all__ = ["CATEGORICAL_FEATURES", "NUMERIC_FEATURES"]

# Configuration paths
PROJECT_DIR = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "features"

# Model training defaults
DEFAULT_ITERATIONS = 500
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_DEPTH = 6
DEFAULT_EARLY_STOPPING = 50
DEFAULT_RANDOM_SEED = 42
