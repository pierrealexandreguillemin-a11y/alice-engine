"""Module: data_loader.py - Repository Layer (SRP).

Acces aux donnees MongoDB et fichiers locaux.
Seul module autorise a faire des I/O.

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (data loading, validation)
- ISO/IEC 25012 - Data Quality (completude, coherence)
- ISO/IEC 42010 - Architecture (Repository layer, SRP)
- ISO/IEC 27001 - Information Security (acces donnees, audit log)

@see CDC_ALICE.md - Data access patterns

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

import logging
import time
from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.audit.logger import AuditLogger

logger = logging.getLogger(__name__)


class DataLoader:
    """Repository pour l'acces aux donnees.

    Responsabilite unique: I/O (MongoDB, fichiers).
    Aucune logique metier ici.

    Methodes:
        - get_opponent_history: Historique compositions d'un club
        - get_club_players: Joueurs d'un club
        - load_training_data: Donnees pour entrainement
    """

    def __init__(
        self,
        mongodb_uri: str | None = None,
        dataset_path: Path | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialise le data loader.

        @param mongodb_uri: URI MongoDB (lecture seule)
        @param dataset_path: Chemin vers dataset local (Parquet)
        @param audit_logger: Logger d'audit (ISO 27001)
        """
        self.mongodb_uri = mongodb_uri
        self.dataset_path = dataset_path
        self.db = None
        self._audit = audit_logger

    async def connect(self) -> bool:
        """Connect to MongoDB.

        @returns: True si connecte
        """
        if not self.mongodb_uri:
            logger.warning("MongoDB URI non configure")
            return False

        try:
            from motor.motor_asyncio import AsyncIOMotorClient  # noqa: PLC0415

            client = AsyncIOMotorClient(self.mongodb_uri)
            self.db = client.get_default_database()
            # Test connexion
            await client.admin.command("ping")
            logger.info("MongoDB connecte")
            return True
        except Exception as e:
            logger.error("Erreur connexion MongoDB: %s", e)
            return False

    async def get_opponent_history(
        self,
        club_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Recupere l'historique des compositions d'un club.

        @param club_id: ID FFE du club
        @param limit: Nombre max de matchs

        @returns: Liste des compositions passees

        @see ISO 25019 - Multi-tenant (filtrage par clubId)
        """
        if not self.db:
            logger.warning("MongoDB non connecte, retourne liste vide")
            return []

        t0 = time.monotonic()
        try:
            query = {"clubId": club_id}
            cursor = (
                self.db.compositions.find(
                    query,
                    {"_id": 0, "roundNumber": 1, "players": 1, "date": 1},
                )
                .sort("date", -1)
                .limit(limit)
            )
            results = await cursor.to_list(length=limit)
            await self._audit_log("read", "compositions", query, len(results), t0)
            return results
        except Exception as e:
            await self._audit_log("read", "compositions", {"clubId": club_id}, 0, t0, error=str(e))
            logger.error("Erreur lecture historique: %s", e)
            return []

    async def get_club_players(
        self,
        club_id: str,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Recupere les joueurs d'un club.

        @param club_id: ID FFE du club
        @param active_only: Uniquement joueurs actifs

        @returns: Liste des joueurs avec Elo

        @see ISO 25019 - Multi-tenant
        """
        if not self.db:
            return []

        t0 = time.monotonic()
        try:
            query: dict[str, Any] = {"clubId": club_id}
            if active_only:
                query["isActive"] = True

            cursor = self.db.players.find(
                query,
                {
                    "_id": 0,
                    "ffeId": 1,
                    "firstName": 1,
                    "lastName": 1,
                    "elo": 1,
                    "licence": 1,
                    "category": 1,
                    "gender": 1,
                },
            ).sort("elo", -1)

            results = await cursor.to_list(length=200)
            await self._audit_log("read", "players", query, len(results), t0)
            return results
        except Exception as e:
            await self._audit_log("read", "players", {"clubId": club_id}, 0, t0, error=str(e))
            logger.error("Erreur lecture joueurs: %s", e)
            return []

    def load_training_data(
        self,
        seasons: list[int] | None = None,
    ) -> Any:
        """Charge les donnees d'entrainement depuis fichiers Parquet.

        @param seasons: Saisons a charger (None = toutes)

        @returns: DataFrame pandas avec les donnees

        @see ISO 25012 - Qualite donnees
        """
        if not self.dataset_path:
            logger.warning("Dataset path non configure")
            return None

        try:
            import pandas as pd  # noqa: PLC0415

            parquet_file = self.dataset_path / "echiquiers.parquet"
            if not parquet_file.exists():
                logger.error("Fichier non trouve: %s", parquet_file)
                return None

            df = pd.read_parquet(parquet_file)

            if seasons:
                df = df[df["saison"].isin(seasons)]

            logger.info("Charge %d lignes depuis %s", len(df), parquet_file)
            return df

        except Exception as e:
            logger.error("Erreur chargement donnees: %s", e)
            return None

    async def _audit_log(  # noqa: PLR0913
        self,
        op: str,
        collection: str,
        query: dict[str, Any],
        count: int,
        t0: float,
        *,
        error: str | None = None,
    ) -> None:
        """Enregistre une operation dans le log d'audit (ISO 27001)."""
        if self._audit is None:
            return
        from services.audit.types import OperationType  # noqa: PLC0415

        op_map = {
            "read": OperationType.READ,
            "write": OperationType.WRITE,
            "update": OperationType.UPDATE,
            "delete": OperationType.DELETE,
        }
        await self._audit.log(
            op_map.get(op, OperationType.READ),
            collection,
            query=query,
            result_count=count,
            duration_ms=(time.monotonic() - t0) * 1000,
            success=error is None,
            error_message=error,
        )
