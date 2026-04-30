"""D8 loader — match eligibility + SHA-256 lineage (ISO 5259).

Document ID: ALICE-D8-LOADER
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MatchSpec:
    """Match candidate eligible for D8 audit."""

    saison: int
    ronde: int
    equipe_dom: str
    equipe_ext: str
    niveau: str


_HASH_CHUNK = 65536


def compute_file_sha256(path: Path) -> str:
    """SHA-256 of file content (ISO 5259 lineage). Read in 65536-byte chunks.

    @raises FileNotFoundError if path missing.
    """
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def filter_eligible_matches(df: pd.DataFrame, saison: int) -> list[MatchSpec]:
    """Filter matches : ronde>=1, equipes valides, dedup unique pairs.

    @param df: echiquiers DataFrame (cols: saison, ronde, equipe_dom, equipe_ext, niveau)
    @param saison: integer year to filter
    @raises TypeError: if saison is not int
    """
    if not isinstance(saison, int):
        msg = f"saison must be int, got {type(saison).__name__}"
        raise TypeError(msg)
    sub = df[df["saison"] == saison]
    sub = sub[sub["ronde"] >= 1]
    sub = sub.dropna(subset=["equipe_dom", "equipe_ext"])
    pairs = sub[["saison", "ronde", "equipe_dom", "equipe_ext", "niveau"]].drop_duplicates()
    return [
        MatchSpec(
            saison=int(row["saison"]),
            ronde=int(row["ronde"]),
            equipe_dom=str(row["equipe_dom"]),
            equipe_ext=str(row["equipe_ext"]),
            niveau=str(row["niveau"]),
        )
        for _, row in pairs.iterrows()
    ]


def load_match_eligible(echiquiers_path: Path, saison: int) -> list[MatchSpec]:
    """Load echiquiers parquet + filter saison eligible matches.

    @raises FileNotFoundError if path missing.
    """
    if not echiquiers_path.exists():
        msg = f"echiquiers parquet not found: {echiquiers_path}"
        raise FileNotFoundError(msg)
    df = pd.read_parquet(echiquiers_path)
    return filter_eligible_matches(df, saison=saison)
