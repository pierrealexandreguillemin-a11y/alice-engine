# services/data_loader.py
"""
Data Loader - Repository Layer (SRP)

Acces aux donnees MongoDB et fichiers locaux.
Seul module autorise a faire des I/O.

@description Charge les donnees historiques pour l'entrainement et l'inference
@see ISO 42010 - Repository layer
@see ISO 25012 - Qualite des donnees
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Repository pour l'acces aux donnees.

    Responsabilite unique: I/O (MongoDB, fichiers).
    Aucune logique metier ici.

    Methodes:
        - get_opponent_history: Historique compositions d'un club
        - get_club_players: Joueurs d'un club
        - load_training_data: Donnees pour entrainement
    """

    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        dataset_path: Optional[Path] = None,
    ):
        """
        Initialise le data loader.

        @param mongodb_uri: URI MongoDB (lecture seule)
        @param dataset_path: Chemin vers dataset local (Parquet)
        """
        self.mongodb_uri = mongodb_uri
        self.dataset_path = dataset_path
        self.db = None

    async def connect(self) -> bool:
        """
        Connecte a MongoDB.

        @returns: True si connecte
        """
        if not self.mongodb_uri:
            logger.warning("MongoDB URI non configure")
            return False

        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            client = AsyncIOMotorClient(self.mongodb_uri)
            self.db = client.get_default_database()
            # Test connexion
            await client.admin.command("ping")
            logger.info("MongoDB connecte")
            return True
        except Exception as e:
            logger.error(f"Erreur connexion MongoDB: {e}")
            return False

    async def get_opponent_history(
        self,
        club_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Recupere l'historique des compositions d'un club.

        @param club_id: ID FFE du club
        @param limit: Nombre max de matchs

        @returns: Liste des compositions passees

        @see ISO 25019 - Multi-tenant (filtrage par clubId)
        """
        if not self.db:
            logger.warning("MongoDB non connecte, retourne liste vide")
            return []

        try:
            cursor = self.db.compositions.find(
                {"clubId": club_id},
                {"_id": 0, "roundNumber": 1, "players": 1, "date": 1},
            ).sort("date", -1).limit(limit)

            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"Erreur lecture historique: {e}")
            return []

    async def get_club_players(
        self,
        club_id: str,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Recupere les joueurs d'un club.

        @param club_id: ID FFE du club
        @param active_only: Uniquement joueurs actifs

        @returns: Liste des joueurs avec Elo

        @see ISO 25019 - Multi-tenant
        """
        if not self.db:
            return []

        try:
            query = {"clubId": club_id}
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

            return await cursor.to_list(length=200)
        except Exception as e:
            logger.error(f"Erreur lecture joueurs: {e}")
            return []

    def load_training_data(
        self,
        seasons: Optional[List[int]] = None,
    ) -> Any:
        """
        Charge les donnees d'entrainement depuis fichiers Parquet.

        @param seasons: Saisons a charger (None = toutes)

        @returns: DataFrame pandas avec les donnees

        @see ISO 25012 - Qualite donnees
        """
        if not self.dataset_path:
            logger.warning("Dataset path non configure")
            return None

        try:
            import pandas as pd

            parquet_file = self.dataset_path / "echiquiers.parquet"
            if not parquet_file.exists():
                logger.error(f"Fichier non trouve: {parquet_file}")
                return None

            df = pd.read_parquet(parquet_file)

            if seasons:
                df = df[df["saison"].isin(seasons)]

            logger.info(f"Charge {len(df)} lignes depuis {parquet_file}")
            return df

        except Exception as e:
            logger.error(f"Erreur chargement donnees: {e}")
            return None
