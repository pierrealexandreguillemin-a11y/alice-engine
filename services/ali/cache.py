"""ALIDataCache — in-RAM cache des parquets pour inference /compose.

ISO 5259 : SHA-256 lineage des parquets source (reproducibility).
ISO 25010 : performance (cache evite I/O par request).
ISO 5055 : SRP strict (I/O seulement, pas de logique metier).

Document ID: ALICE-ALI-CACHE
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ALIDataCache:
    """Cache in-RAM des parquets + indexes + SHA-256 lineage.

    Charge une fois au lifespan FastAPI (Plan 2).
    """

    joueurs_total: pd.DataFrame
    echiquiers_total: pd.DataFrame
    joueurs_by_club: dict[str, pd.DataFrame]
    echiquiers_by_player: dict[str, pd.DataFrame]
    parquet_sig_joueurs: str
    parquet_sig_echiquiers: str
    loaded_at: datetime

    @classmethod
    def load_from_parquets(
        cls,
        path_joueurs: Path,
        path_echiquiers: Path,
    ) -> ALIDataCache:
        """Charge parquets + calcule SHA-256 + construit indexes."""
        sig_j = hashlib.sha256(path_joueurs.read_bytes()).hexdigest()
        sig_e = hashlib.sha256(path_echiquiers.read_bytes()).hexdigest()
        df_j = pd.read_parquet(path_joueurs)
        df_e = pd.read_parquet(path_echiquiers)
        return cls(
            joueurs_total=df_j,
            echiquiers_total=df_e,
            joueurs_by_club=cls._index_by_club(df_j),
            echiquiers_by_player=cls._index_by_player(df_e),
            parquet_sig_joueurs=sig_j,
            parquet_sig_echiquiers=sig_e,
            loaded_at=datetime.now(UTC),
        )

    @staticmethod
    def _index_by_club(df_j: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Index joueurs par club (cle = str(club))."""
        return {str(club): group for club, group in df_j.groupby("club", dropna=False)}

    @staticmethod
    def _index_by_player(df_e: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Index echiquiers par joueur (union blanc_nom + noir_nom)."""
        idx: dict[str, pd.DataFrame] = {}
        for color in ("blanc", "noir"):
            col = f"{color}_nom"
            if col not in df_e.columns:
                continue
            for name, group in df_e.groupby(col, dropna=True):
                key = str(name)
                idx[key] = (
                    group if key not in idx else pd.concat([idx[key], group], ignore_index=True)
                )
        return idx

    def is_stale(self, max_age_days: int = 7) -> bool:
        """Return True si l'age du cache depasse max_age_days (UTC)."""
        age = (datetime.now(UTC) - self.loaded_at).total_seconds()
        return age > max_age_days * 86400

    def lineage_ok(self) -> bool:
        """Sanity check : signatures SHA-256 présentes et non-vides."""
        return bool(self.parquet_sig_joueurs) and bool(self.parquet_sig_echiquiers)

    def lookup_club(self, club_id: str) -> pd.DataFrame:
        """Return joueurs subset for a given club_id. Empty DataFrame if unknown."""
        return self.joueurs_by_club.get(
            str(club_id),
            self.joueurs_total.iloc[0:0],
        )

    def lookup_history(self, player_names: list[str]) -> pd.DataFrame:
        """Return echiquiers rows where blanc_nom OR noir_nom in player_names."""
        parts: list[pd.DataFrame] = []
        for name in player_names:
            df = self.echiquiers_by_player.get(str(name))
            if df is not None:
                parts.append(df)
        if not parts:
            return self.echiquiers_total.iloc[0:0]
        return pd.concat(parts, ignore_index=True).drop_duplicates()
