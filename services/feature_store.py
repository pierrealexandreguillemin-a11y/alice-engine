"""Feature Store — assemble features for inference (ISO 5259/42001).

Document ID: ALICE-FEATURE-STORE
Version: 2.0.0

v2.0 (Plan 3 Pre-Task 1 fix) : assemble 201-col DataFrame matching
model feature_names (from LightGBM.txt training).

Sources:
- data/feature_store/training_mean.parquet : 201-col mean row (fallback)
- data/feature_store/joueur_features.parquet : per-player canonical aggregates
- Built by scripts/build_feature_store.py

ISO 5259 : SHA-256 lineage per parquet.
ISO 42001 : feature schema = LightGBM.txt feature_names (201 cols).
ISO 24029 : unknown players fall back to training_mean (robustness).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_BLANC_SUFFIX = "_blanc"
_NOIR_SUFFIX = "_noir"


class FeatureStore:
    """Assemble 201-col feature DataFrame for inference from pre-computed parquets."""

    def __init__(self, store_path: Path) -> None:
        """Initialize the feature store with the path to the parquet directory."""
        self._path = Path(store_path)
        self._player_features: pd.DataFrame | None = None  # per-player canonical aggregates
        self._training_mean: pd.Series | None = None  # 201-col fallback row
        self._feature_names: list[str] = []  # preserved order from training_mean
        self._loaded = False
        self._load_time: datetime | None = None
        self._hashes: dict[str, str] = {}

    def load(self) -> None:
        """Load training_mean.parquet + joueur_features.parquet."""
        mean_path = self._path / "training_mean.parquet"
        player_path = self._path / "joueur_features.parquet"

        if not mean_path.exists():
            msg = (
                f"training_mean.parquet missing in {self._path}. "
                "Run scripts/build_feature_store.py first."
            )
            raise FileNotFoundError(msg)

        mean_df = pd.read_parquet(mean_path)
        if len(mean_df) < 1:
            msg = f"training_mean.parquet empty at {mean_path}"
            raise RuntimeError(msg)
        self._training_mean = mean_df.iloc[0]  # single-row DataFrame → Series
        self._feature_names = list(mean_df.columns)
        self._hashes["training_mean"] = self._sha256(mean_path)

        if player_path.exists():
            self._player_features = pd.read_parquet(player_path)
            self._hashes["player_features"] = self._sha256(player_path)
            logger.info(
                "Feature store v2 loaded : %d players, %d feature_names, hash=%s",
                len(self._player_features),
                len(self._feature_names),
                self._hashes["training_mean"][:12],
            )
        else:
            logger.warning(
                "joueur_features.parquet absent — all inferences use training_mean fallback"
            )
            self._player_features = pd.DataFrame()

        self._loaded = True
        self._load_time = datetime.now(tz=UTC)

    @property
    def age_hours(self) -> float:
        """Return hours elapsed since the last load() call."""
        if self._load_time is None:
            return float("inf")
        delta = datetime.now(tz=UTC) - self._load_time
        return delta.total_seconds() / 3600

    @property
    def feature_names(self) -> list[str]:
        """201 feature names matching trained model (LightGBM.txt)."""
        return list(self._feature_names)

    def assemble(  # noqa: PLR0913
        self,
        player_name: str,
        player_elo: int,
        opponent_elo: int,
        context: dict[str, object],
        opponent_name: str | None = None,
    ) -> pd.DataFrame:
        """Assemble single-row 201-col feature DataFrame matching model schema.

        Strategy (ISO 24029 robustness + ISO 42001 explainability):
        1. Start from training_mean (201-col baseline)
        2. Override blanc_* cols with player_name aggregates (if known)
        3. Override noir_* cols with opponent_name aggregates (if known)
        4. Override blanc_elo, noir_elo, diff_elo with input
        5. Apply context (division, ronde, etc.) — numeric only
        """
        if not self._loaded:
            raise RuntimeError("Feature store not loaded — call load() first")

        # Start from training mean (copy to avoid mutation)
        assert self._training_mean is not None
        row = self._training_mean.copy()

        # Override blanc_* from player aggregates
        self._override_from_player(row, player_name, side=_BLANC_SUFFIX)

        # Override noir_* from opponent aggregates
        if opponent_name:
            self._override_from_player(row, opponent_name, side=_NOIR_SUFFIX)

        # Elo overrides (always from input)
        row["blanc_elo"] = float(player_elo)
        row["noir_elo"] = float(opponent_elo)
        if "diff_elo" in row.index:
            row["diff_elo"] = float(player_elo - opponent_elo)

        # Context overrides (numeric only)
        for key, value in context.items():
            if key in row.index and isinstance(value, int | float):
                row[key] = float(value)

        # Return single-row DataFrame with 201 cols in correct order
        return pd.DataFrame([row.to_dict()], columns=self._feature_names)

    def _override_from_player(self, row: pd.Series, player_name: str, side: str) -> None:
        """Override row's side_* cols from player's canonical aggregates.

        side = '_blanc' or '_noir'.
        """
        if self._player_features is None or len(self._player_features) == 0:
            return
        if player_name not in self._player_features.index:
            return
        player_row = self._player_features.loc[player_name]
        for canonical_col, value in player_row.items():
            sided_col = f"{canonical_col}{side}"
            if sided_col in row.index and pd.notna(value):
                row[sided_col] = float(value)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Return first 16 hex chars of the SHA-256 digest for lineage tracking."""
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()[:16]
