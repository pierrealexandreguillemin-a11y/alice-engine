"""Pandera schema validation for backtest inputs (ISO 27034 + 5259).

Plan 3 V2 T2b. Validation of user-supplied data before running backtest match.

ISO 27034 : input validation (rejects malformed data).
ISO 5259  : ensures schema consistency with training assumptions.

Document ID: ALICE-BACKTEST-SCHEMAS
Version: 1.0.0
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

_VALID_DIVISIONS = frozenset({"N1", "N2", "N3", "N4", "N5", "N6"})


class BacktestInputSchema(pa.DataFrameModel):
    """Pandera schema for backtest inputs (one row per match to backtest)."""

    club_id: Series[str] = pa.Field(str_length={"min_value": 1})
    opponent_club: Series[str] = pa.Field(str_length={"min_value": 1})
    saison: Series[int] = pa.Field(ge=2000, le=2100)
    ronde: Series[int] = pa.Field(ge=1, le=20)
    division: Series[str] = pa.Field(isin=_VALID_DIVISIONS)
    team_size: Series[int] = pa.Field(ge=4, le=16)
    nb_rondes_total: Series[int] = pa.Field(ge=1, le=20)


def validate_backtest_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input DataFrame.

    @raises pandera.errors.SchemaError: when the DataFrame violates the schema.
    """
    return BacktestInputSchema.validate(df)
