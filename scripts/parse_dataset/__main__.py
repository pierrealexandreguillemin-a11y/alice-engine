"""Entry point for: python -m scripts.parse_dataset.

Parses raw FFE HTML data to Parquet and validates ISO 5259.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML
- ISO/IEC 5055:2021 - Code Quality (<50 lines)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from schemas.parsing_validation import validate_raw_echiquiers, validate_raw_joueurs
from scripts.parse_dataset.orchestration import parse_compositions, parse_joueurs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse HTML -> Parquet + validate ISO 5259."""
    data_dir = Path("dataset_alice")
    output_dir = Path("data")

    logger.info("Parsing compositions -> echiquiers.parquet")
    stats_ech = parse_compositions(data_dir, output_dir / "echiquiers.parquet")
    logger.info("Compositions: %s", stats_ech)

    logger.info("Parsing players -> joueurs.parquet")
    stats_jou = parse_joueurs(data_dir / "players_v2", output_dir / "joueurs.parquet")
    logger.info("Players: %s", stats_jou)

    logger.info("Validating ISO 5259")
    df_ech = pd.read_parquet(output_dir / "echiquiers.parquet")
    df_jou = pd.read_parquet(output_dir / "joueurs.parquet")
    validate_raw_echiquiers(df_ech, source_path=str(output_dir / "echiquiers.parquet"))
    validate_raw_joueurs(df_jou, source_path=str(output_dir / "joueurs.parquet"))

    logger.info("Parse + validation complete")


if __name__ == "__main__":
    main()
