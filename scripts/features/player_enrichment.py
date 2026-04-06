"""Player enrichment from joueurs.parquet - ISO 5055/5259.

Enrichit les echiquiers avec des donnees joueur depuis joueurs.parquet:
- elo_type (F/N/E): type de classement FIDE/National/Estime
- categorie: categorie d'age FFE (U8 -> S65)
- k_coefficient: coefficient K FIDE (10/20/40) selon FIDE 8.3.3

Conformite ISO/IEC:
- 5055: Module <300 lignes, SRP, fonctions <50 lignes
- 5259: Qualite donnees ML, enrichissement depuis source officielle
- 27034: Validation d'entree (Pydantic-style guards)

Document ID: ALICE-FEA-ENRICH-001
Version: 1.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# FFE age categories mapping to FIDE "young player" (K=40)
# FFE uses French abbreviations: PpoM=Petit Poussin Masculin (U8), PouM=Poussin (U10), etc.
# Covers U8 through U18 (< 18 years old at start of season)
_YOUNG_CATS = frozenset(
    {
        "PpoM",
        "PpoF",  # Petit Poussin = U8
        "PouM",
        "PouF",  # Poussin = U10
        "PupM",
        "PupF",  # Pupille = U12
        "BenM",
        "BenF",  # Benjamin = U14
        "MinM",
        "MinF",  # Minime = U16
        "CadM",
        "CadF",  # Cadet = U18
    }
)


def enrich_from_joueurs(df_split: pd.DataFrame, joueurs_path: Path) -> None:  # noqa: D417
    """Ajoute elo_type, categorie, k_coefficient depuis joueurs.parquet (in-place).

    Join par nom_complet <-> blanc_nom / noir_nom.
    Vectorise pour eviter apply(axis=1) sur 3.5M lignes.

    Args:
    ----
        df_split: DataFrame echiquiers (modifie in-place)
        joueurs_path: Path — chemin absolu vers joueurs.parquet

    ISO 5259: Enrichissement depuis source officielle FFE.
    """
    if not joueurs_path.exists():
        logger.warning("joueurs.parquet non trouve: %s — skip enrichissement", joueurs_path)
        return

    joueur_map = _load_joueur_map(joueurs_path)
    if joueur_map.empty:
        return

    for color in ("blanc", "noir"):
        _enrich_color(df_split, joueur_map, color)

    logger.info("  Enrichissement joueurs: elo_type, categorie, k_coefficient (blanc + noir)")


def _load_joueur_map(joueurs_path: Path) -> pd.DataFrame:
    """Charge le mapping joueur depuis joueurs.parquet."""
    try:
        joueurs = pd.read_parquet(
            joueurs_path,
            columns=["nom_complet", "elo_type", "categorie"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Erreur lecture joueurs.parquet: %s", exc)
        return pd.DataFrame()

    return joueurs.drop_duplicates("nom_complet").set_index("nom_complet")


def _enrich_color(
    df_split: pd.DataFrame,
    joueur_map: pd.DataFrame,
    color: str,
) -> None:
    """Enrichit les colonnes pour une couleur (blanc ou noir)."""
    nom_col = f"{color}_nom"
    elo_col = f"{color}_elo"

    if nom_col not in df_split.columns:
        return

    df_split[f"elo_type_{color}"] = df_split[nom_col].map(joueur_map["elo_type"]).fillna("")
    df_split[f"categorie_{color}"] = df_split[nom_col].map(joueur_map["categorie"]).fillna("")

    if elo_col not in df_split.columns:
        df_split[f"k_coefficient_{color}"] = 20
        return

    _compute_k_vectorized(df_split, color, elo_col)


def _compute_k_vectorized(
    df_split: pd.DataFrame,
    color: str,
    elo_col: str,
) -> None:
    """Calcule k_coefficient via operations vectorisees (FIDE 8.3.3).

    Ordre d'application:
    1. Defaut = 20
    2. Si elo >= 2400 → 10 (prend precedence sur jeune)
    3. Si categorie jeune ET elo < 2300 → 40
    """
    cat_col = f"categorie_{color}"
    k_col = f"k_coefficient_{color}"

    elo = df_split[elo_col].fillna(0)

    df_split[k_col] = 20
    # K=40: joueur jeune avec elo < 2300
    is_young = df_split[cat_col].isin(_YOUNG_CATS)
    df_split.loc[is_young & (elo < 2300), k_col] = 40
    # K=10: elite (prend precedence)
    df_split.loc[elo >= 2400, k_col] = 10
