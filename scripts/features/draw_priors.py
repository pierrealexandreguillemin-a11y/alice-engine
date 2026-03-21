"""Draw priors — avg_elo, elo_proximity, draw_rate_prior - ISO 5055/5259.

Document ID: ALICE-FEAT-DRAW-PRIORS
Version: 1.0.0

Single responsibility: compute calibrated draw probability priors
derived from Elo band × diff band historical rates.

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (SRP, <300 lines)
- ISO/IEC 5259:2024 - Data Quality for ML (forfait exclusion, no leakage)
- ISO/IEC 42001:2023 - AI Management (traceable feature engineering)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.features.helpers import exclude_forfeits

logger = logging.getLogger(__name__)

# ELO_BINS and DIFF_BINS are PUBLIC — consumed by baselines.py (Plan B)
ELO_BINS: list[int] = [0, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 3500]
DIFF_BINS: list[int] = [0, 50, 100, 200, 400, 800]

# Minimum observations per cell for a reliable prior
_MIN_CELL_ROWS: int = 10


def build_draw_rate_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Build draw rate lookup table by (elo_band × diff_band).

    PUBLIC — used internally by compute_draw_priors and externally
    by baselines.py (Plan B).

    Args:
    ----
        df: Raw games DataFrame (forfeits excluded inside this function).

    Returns:
    -------
        DataFrame with columns: elo_band, diff_band, draw_rate_prior.
        Only cells with >= _MIN_CELL_ROWS observations are kept.
    """
    clean = exclude_forfeits(df)

    clean = _fill_elo_medians(clean)
    clean = _add_elo_bands(clean)

    clean["_is_draw"] = (clean["resultat_blanc"] == 0.5).astype(int)
    agg = (
        clean.groupby(["elo_band", "diff_band"])
        .agg(
            n=("resultat_blanc", "count"),
            n_draws=("_is_draw", "sum"),
        )
        .reset_index()
    )
    agg = agg[agg["n"] >= _MIN_CELL_ROWS].copy()
    agg["draw_rate_prior"] = agg["n_draws"] / agg["n"]

    return agg[["elo_band", "diff_band", "draw_rate_prior"]]


def compute_draw_priors(
    df_split: pd.DataFrame,
    df_history: pd.DataFrame,
) -> pd.DataFrame:
    """Add avg_elo, elo_proximity, and draw_rate_prior to df_split.

    Lookup table is built from df_history only (train set) — no leakage.

    Args:
    ----
        df_split: DataFrame for which to compute features (val/test split).
        df_history: Train-only DataFrame used to build the lookup table.

    Returns:
    -------
        df_split with three new columns added:
        - avg_elo: (blanc_elo + noir_elo) / 2
        - elo_proximity: 1 - min(|diff_elo|, 800) / 800, clipped [0, 1]
        - draw_rate_prior: historical draw rate from lookup, or global rate
    """
    result = df_split.copy()

    result = _fill_elo_medians(result)
    result = _add_elo_bands(result)

    result["avg_elo"] = (result["blanc_elo"] + result["noir_elo"]) / 2

    abs_diff = result["diff_elo"].abs()
    result["elo_proximity"] = (1 - (abs_diff.clip(upper=800) / 800)).clip(0.0, 1.0)

    lookup = build_draw_rate_lookup(df_history)
    global_rate = _global_draw_rate(df_history)

    result = result.merge(lookup, on=["elo_band", "diff_band"], how="left")
    result["draw_rate_prior"] = result["draw_rate_prior"].fillna(global_rate)

    # Drop helper band columns added internally
    result = result.drop(columns=["elo_band", "diff_band", "diff_elo"], errors="ignore")

    logger.info(
        "compute_draw_priors: %d rows, global_rate=%.3f, " "null_priors=%d",
        len(result),
        global_rate,
        result["draw_rate_prior"].isna().sum(),
    )
    return result


def compute_player_draw_rates(
    df: pd.DataFrame,
    min_games: int = 10,
) -> pd.DataFrame:
    """Compute per-player draw rate combining blanc + noir appearances.

    For noir perspective: draw stays 0.5, win/loss are flipped in value
    but draw detection uses resultat_blanc == 0.5 regardless.

    Args:
    ----
        df: Raw games DataFrame.
        min_games: Minimum combined games to include a player.

    Returns:
    -------
        DataFrame with columns: joueur_nom, draw_rate, n_games.
    """
    clean = exclude_forfeits(df)

    clean["_is_draw"] = (clean["resultat_blanc"] == 0.5).astype(int)

    blanc_draws = (
        clean.groupby("blanc_nom")
        .agg(
            n_games=("resultat_blanc", "count"),
            n_draws=("_is_draw", "sum"),
        )
        .reset_index()
        .rename(columns={"blanc_nom": "joueur_nom"})
    )

    noir_draws = (
        clean.groupby("noir_nom")
        .agg(
            n_games=("resultat_noir", "count"),
            n_draws=("_is_draw", "sum"),
        )
        .reset_index()
        .rename(columns={"noir_nom": "joueur_nom"})
    )

    combined = (
        pd.concat([blanc_draws, noir_draws])
        .groupby("joueur_nom")
        .agg(n_games=("n_games", "sum"), n_draws=("n_draws", "sum"))
        .reset_index()
    )

    combined = combined[combined["n_games"] >= min_games].copy()
    combined["draw_rate"] = combined["n_draws"] / combined["n_games"]

    return combined[["joueur_nom", "draw_rate", "n_games"]]


def compute_equipe_draw_rates(
    df: pd.DataFrame,
    min_games: int = 10,
) -> pd.DataFrame:
    """Compute per-team draw rate combining home + away perspectives.

    Args:
    ----
        df: Raw games DataFrame with equipe_dom and equipe_ext columns.
        min_games: Minimum combined games to include a team.

    Returns:
    -------
        DataFrame with columns: equipe, draw_rate, n_games.
    """
    clean = exclude_forfeits(df)

    is_draw = (clean["resultat_blanc"] == 0.5).astype(int)
    clean = clean.copy()
    clean["_is_draw"] = is_draw

    dom_draws = (
        clean.groupby("equipe_dom")
        .agg(n_games=("_is_draw", "count"), n_draws=("_is_draw", "sum"))
        .reset_index()
        .rename(columns={"equipe_dom": "equipe"})
    )

    ext_draws = (
        clean.groupby("equipe_ext")
        .agg(n_games=("_is_draw", "count"), n_draws=("_is_draw", "sum"))
        .reset_index()
        .rename(columns={"equipe_ext": "equipe"})
    )

    combined = (
        pd.concat([dom_draws, ext_draws])
        .groupby("equipe")
        .agg(n_games=("n_games", "sum"), n_draws=("n_draws", "sum"))
        .reset_index()
    )

    combined = combined[combined["n_games"] >= min_games].copy()
    combined["draw_rate"] = combined["n_draws"] / combined["n_games"]

    return combined[["equipe", "draw_rate", "n_games"]]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fill_elo_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN Elo values with per-column median (ISO 5259 graceful handling)."""
    result = df.copy()
    for col in ("blanc_elo", "noir_elo"):
        if col in result.columns:
            median_val = result[col].median()
            result[col] = result[col].fillna(median_val if not np.isnan(median_val) else 1500.0)
    return result


def _add_elo_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add elo_band and diff_elo / diff_band columns."""
    result = df.copy()
    result["avg_elo_tmp"] = (result["blanc_elo"] + result["noir_elo"]) / 2
    result["diff_elo"] = result["blanc_elo"] - result["noir_elo"]
    abs_diff = result["diff_elo"].abs()

    result["elo_band"] = pd.cut(
        result["avg_elo_tmp"],
        bins=ELO_BINS,
        labels=[f"{ELO_BINS[i]}-{ELO_BINS[i+1]}" for i in range(len(ELO_BINS) - 1)],
        right=False,
    ).astype(str)

    result["diff_band"] = pd.cut(
        abs_diff,
        bins=DIFF_BINS,
        labels=[f"{DIFF_BINS[i]}-{DIFF_BINS[i+1]}" for i in range(len(DIFF_BINS) - 1)],
        right=False,
    ).astype(str)

    return result.drop(columns=["avg_elo_tmp"])


def _global_draw_rate(df: pd.DataFrame) -> float:
    """Compute global draw rate from df (forfeits excluded) as fallback."""
    clean = exclude_forfeits(df)
    if clean.empty:
        return 0.15
    return float((clean["resultat_blanc"] == 0.5).mean())
