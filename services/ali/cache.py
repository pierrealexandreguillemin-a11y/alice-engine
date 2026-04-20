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
    team_to_club: dict[str, str]
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
            team_to_club=cls._build_team_to_club(df_j, df_e),
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

    @staticmethod
    def _build_team_to_club(df_j: pd.DataFrame, df_e: pd.DataFrame) -> dict[str, str]:
        """Map echiquiers team_name (equipe_dom/ext) -> joueurs club_name.

        Plan 3 T11 needs to lookup opponent's joueur pool from their echiquiers
        team_name (since echiquiers uses "Nancy-Metz Saint Leon IX" while
        joueurs.club uses "Nancy-Metz Saint Leon"). Strategy : majority vote
        over joueurs.club of players who played for each team.

        For each (blanc_nom, blanc_equipe) row : lookup blanc_nom in joueurs
        to get joueurs.club. Count (team, club) occurrences. Team -> club with
        highest count.

        ISO 5259 : derived lineage explicit. ISO 42010 : additive to cache API.
        """
        if "nom_complet" not in df_j.columns or "club" not in df_j.columns:
            return {}
        player_to_club = dict(
            zip(
                df_j["nom_complet"].astype(str),
                df_j["club"].astype(str),
                strict=True,
            )
        )

        counts: dict[str, dict[str, int]] = {}
        for color in ("blanc", "noir"):
            nom_col = f"{color}_nom"
            eq_col = f"{color}_equipe"
            if nom_col not in df_e.columns or eq_col not in df_e.columns:
                continue
            sub = df_e[[nom_col, eq_col]].dropna()
            for player_name, team in zip(
                sub[nom_col].astype(str), sub[eq_col].astype(str), strict=True
            ):
                club = player_to_club.get(player_name)
                if club is None or not team:
                    continue
                counts.setdefault(team, {})[club] = counts.setdefault(team, {}).get(club, 0) + 1

        return {team: max(clubs, key=lambda c: clubs[c]) for team, clubs in counts.items()}

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
