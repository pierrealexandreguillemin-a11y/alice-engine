"""Tests T2b Pandera schema validation (Plan 3 V2)."""

from __future__ import annotations

import pandas as pd
import pandera.errors as pa_errors
import pytest

from scripts.backtest.schemas import validate_backtest_inputs


def _good_df() -> pd.DataFrame:
    """Return a valid backtest-input DataFrame (single row)."""
    return pd.DataFrame(
        [
            {
                "club_id": "CLUB1",
                "opponent_club": "CLUB2",
                "saison": 2024,
                "ronde": 5,
                "division": "N3",
                "team_size": 8,
                "nb_rondes_total": 11,
            },
        ],
    )


def test_schema_accepts_valid_input() -> None:
    """Ligne valide → validation OK, shape conservée."""
    result = validate_backtest_inputs(_good_df())
    assert len(result) == 1
    assert list(result.columns) == list(_good_df().columns)


def test_schema_rejects_invalid_division() -> None:
    """Division hors enum N1-N6 → SchemaError."""
    df = _good_df()
    df["division"] = "INVALID"
    with pytest.raises(pa_errors.SchemaError):
        validate_backtest_inputs(df)


def test_schema_rejects_out_of_range_ronde() -> None:
    """Ronde > 20 → SchemaError (ISO 27034 input validation)."""
    df = _good_df()
    df["ronde"] = 50
    with pytest.raises(pa_errors.SchemaError):
        validate_backtest_inputs(df)
