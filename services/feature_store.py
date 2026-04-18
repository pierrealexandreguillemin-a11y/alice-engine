"""Feature Store — assemble features for inference (ISO 5259/42001).

Document ID: ALICE-FEATURE-STORE
Version: 1.0.0

Looks up pre-computed player features from parquets.
Unknown players get default features (ISO 24029 robustness).
SHA-256 lineage tracked per parquet (ISO 5259).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """Assemble features for inference from pre-computed parquets."""

    def __init__(self, store_path: Path) -> None:
        """Initialize the feature store with the path to the parquet directory."""
        self._path = Path(store_path)
        self._joueur_features: pd.DataFrame | None = None
        self._loaded = False
        self._load_time: datetime | None = None
        self._hashes: dict[str, str] = {}

    def load(self) -> None:
        """Load joueur_features.parquet and compute SHA-256 lineage hash."""
        joueur_path = self._path / "joueur_features.parquet"
        if joueur_path.exists():
            self._joueur_features = pd.read_parquet(joueur_path)
            self._hashes["joueur"] = self._sha256(joueur_path)
            logger.info(
                "Feature store loaded: %d players, hash=%s",
                len(self._joueur_features),
                self._hashes["joueur"],
            )
        else:
            self._joueur_features = pd.DataFrame()
            logger.warning("No joueur_features.parquet — empty feature store")
        self._loaded = True
        self._load_time = datetime.now(tz=UTC)

    @property
    def age_hours(self) -> float:
        """Return hours elapsed since the last load() call."""
        if self._load_time is None:
            return float("inf")
        delta = datetime.now(tz=UTC) - self._load_time
        return delta.total_seconds() / 3600

    def assemble(
        self,
        player_name: str,
        player_elo: int,
        opponent_elo: int,
        context: dict,
    ) -> pd.DataFrame:
        """Assemble a single-row feature DataFrame for inference.

        Looks up the player in the joueur features parquet.
        Unknown players fall back to default features.
        Context keys (division, ronde, …) are merged last.
        """
        if not self._loaded:
            raise RuntimeError("Feature store not loaded — call load() first")
        row = self._lookup_player(player_name)
        row["blanc_elo"] = player_elo
        row["noir_elo"] = opponent_elo
        row["diff_elo"] = player_elo - opponent_elo
        for key, value in context.items():
            row[key] = value
        return pd.DataFrame([row])

    def _lookup_player(self, name: str) -> dict[str, object]:
        """Return feature dict for player name, or defaults if not found."""
        if self._joueur_features is not None and len(self._joueur_features) > 0:
            if "joueur_nom" in self._joueur_features.columns:
                match = self._joueur_features[self._joueur_features["joueur_nom"] == name]
                if len(match) > 0:
                    return dict(match.iloc[0])
        logger.debug("Unknown player '%s' — using defaults", name)
        return {"blanc_elo": 1500, "noir_elo": 1500, "diff_elo": 0}

    @staticmethod
    def _sha256(path: Path) -> str:
        """Return first 16 hex chars of the SHA-256 digest for lineage tracking."""
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()[:16]
