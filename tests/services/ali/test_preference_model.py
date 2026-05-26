"""Tests for services.ali.preference_model PreferenceModel (Phase 4a T2).

ISO 29119 : class-based structured tests + bare functions, >= 10 cases.
ISO 5259  : Pandera DataFrameModel schema validation of echiquiers.parquet.
ISO 24027 : Bias gate test on synthetic gender-disparate data.
ISO 25059 : >= 90% coverage on services.ali.preference_model.
ISO 5055  : SRP per test class.

Tests cover :
 - TestSchema (T2.2)            : Pandera echiquiers.parquet validation.
 - TestPreferenceFeatures       : frozen dataclass invariant.
 - TestFitDeterminism (T2.4)    : signal recovery + same seed -> same SHA.
 - TestSerialization (T2.5)     : joblib roundtrip + inference < 100ms.
 - TestEdgeCases                : empty/single-team/missing-feature guards.
 - TestBiasGate (ISO 24027)     : recall gap > 0.10 -> BiasViolationError.

Document ID: ALICE-TEST-ALI-PREFERENCE-MODEL
Version: 1.0.0
Count: 12 test cases
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.typing import Series

from services.ali.preference_model import (
    BiasViolationError,
    PreferenceModel,
    load_artifact,
    save_artifact,
)
from services.ali.types import PreferenceFeatures


# ---------------------------------------------------------------------------
# ISO 5259 : Pandera schema for echiquiers.parquet (subset used by T2)
# ---------------------------------------------------------------------------
class EchiquiersSchema(pa.DataFrameModel):
    """ISO 5259 schema validation for echiquiers.parquet subset 2024.

    Range `ronde 1..18` and `echiquier 1..16` empirically verified on
    real parquet (2026-05-26). `blanc_elo >= 0` (0 = unrated scolaires,
    filtered upstream by PreferenceModel._build_features via > 0).
    """

    saison: Series[int] = pa.Field(ge=2000, le=2030)
    division: Series[str] = pa.Field(nullable=False)
    ronde: Series[int] = pa.Field(ge=1, le=20)
    echiquier: Series[int] = pa.Field(ge=1, le=16)
    equipe_dom: Series[str] = pa.Field(nullable=False)
    equipe_ext: Series[str] = pa.Field(nullable=False)
    blanc_equipe: Series[str] = pa.Field(nullable=False)
    blanc_elo: Series[int] = pa.Field(ge=0)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _synthetic_df(n_players: int = 200, saison: int = 2024) -> pd.DataFrame:
    """Synthetic echiquiers-shaped df : higher Elo -> lower team_rank."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_players):
        team_rank = i // 8  # 8 players per team
        elo = 2400 - i * 5 + int(rng.normal(0, 30))
        rows.append(
            {
                "saison": saison,
                "ronde": 5,
                "echiquier": (i % 8) + 1,
                "blanc_equipe": f"Team{team_rank}",
                "blanc_elo": max(elo, 100),
            }
        )
    return pd.DataFrame(rows)


def _synthetic_df_gender_biased() -> pd.DataFrame:
    """Synthetic df with gender disparity to trigger ISO 24027 bias gate.

    Male rows : Elo perfectly correlated with team_rank (clean signal).
    Female rows : team_rank randomised independently of Elo (label noise).
    -> Model learns Elo -> rank mapping which gives high recall for M
       and near-random recall for F, producing a recall gap > 0.10.
    """
    rng = np.random.default_rng(7)
    rows = []
    n_per_group = 300
    # Males : clean Elo->rank mapping (rank = (2400 - elo) // 100 clipped 0..3)
    for i in range(n_per_group):
        elo = 2400 - i  # 2400 down to 2101
        team_rank = min(i // 75, 3)  # 4 distinct ranks 0..3
        rows.append(
            {
                "saison": 2024,
                "ronde": 5,
                "echiquier": team_rank + 1,
                "blanc_equipe": f"TeamM{team_rank}",
                "blanc_elo": elo,
                "sexe": "M",
            }
        )
    # Females : same Elo range, but team_rank randomised (no signal -> low recall)
    for i in range(n_per_group):
        elo = 2400 - i
        team_rank = int(rng.integers(0, 4))
        rows.append(
            {
                "saison": 2024,
                "ronde": 5,
                "echiquier": team_rank + 1,
                "blanc_equipe": f"TeamF{team_rank}",
                "blanc_elo": elo,
                "sexe": "F",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T2.2 — Schema validation
# ---------------------------------------------------------------------------
class TestSchema:
    """T2.2 : Pandera ISO 5259 validation of echiquiers.parquet 2024 subset."""

    def test_echiquiers_schema_2024(self) -> None:
        parquet = Path("data/echiquiers.parquet")
        if not parquet.exists():  # pragma: no cover - CI safeguard
            pytest.skip("data/echiquiers.parquet not present")
        df = pd.read_parquet(parquet)
        df_2024 = df[df["saison"] == 2024].head(1000)
        # ISO 5259 : raises pa.errors.SchemaError on violation
        EchiquiersSchema.validate(df_2024)


# ---------------------------------------------------------------------------
# PreferenceFeatures dataclass
# ---------------------------------------------------------------------------
class TestPreferenceFeatures:
    """PreferenceFeatures frozen invariant (ISO 29119 testability)."""

    def test_frozen(self) -> None:
        f = PreferenceFeatures(
            player_nr_ffe="P00001",
            team_name="T0",
            elo=2400,
            recency_decay=0.5,
            streak_count=1,
            brule_count=0,
            historical_team_rank=0,
        )
        assert dataclasses.is_dataclass(f)
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.elo = 1500  # type: ignore[misc]


# ---------------------------------------------------------------------------
# T2.4 — Fit + determinism
# ---------------------------------------------------------------------------
class TestFitDeterminism:
    """T2.4 : signal recovery + same seed -> identical artifact SHA."""

    def test_fit_synthetic_recovers_signal(self) -> None:
        df = _synthetic_df()
        model = PreferenceModel(laplace_alpha=1.0, seed=42)
        artifact = model.fit(df, saison=2024)
        assert artifact.train_size > 0
        assert artifact.n_teams_max >= 1
        # High Elo -> team_rank 0
        feats_high = [
            PreferenceFeatures("P00001", "T0", 2400, 0.5, 1, 0, 0),
        ]
        proba_high = model.predict_proba(feats_high)
        assert proba_high[0].argmax() == 0

    def test_fit_synthetic_low_elo_higher_rank(self) -> None:
        df = _synthetic_df()
        model = PreferenceModel(seed=42).fit_with_self_return(df, 2024)
        feats_low = [PreferenceFeatures("P00200", "T9", 1500, 0.1, 1, 0, 5)]
        proba_low = model.predict_proba(feats_low)
        # Lowest Elo : argmax should NOT be rank 0
        assert proba_low[0].argmax() > 0

    def test_determinism_same_seed_same_artifact(self) -> None:
        df = _synthetic_df()
        a1 = PreferenceModel(seed=42).fit(df, 2024)
        a2 = PreferenceModel(seed=42).fit(df, 2024)
        assert a1.artifact_sha256 == a2.artifact_sha256
        assert a1.input_sha256 == a2.input_sha256

    def test_predict_before_fit_raises(self) -> None:
        model = PreferenceModel(seed=42)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba([PreferenceFeatures("P00001", "T0", 2400, 0.5, 1, 0, 0)])


# ---------------------------------------------------------------------------
# T2.5 — Serialization + inference perf
# ---------------------------------------------------------------------------
class TestSerialization:
    """T2.5 : joblib roundtrip + inference batch < 100ms."""

    def test_serialization_roundtrip(self, tmp_path: Path) -> None:
        df = _synthetic_df()
        m = PreferenceModel(seed=42)
        a = m.fit(df, 2024)
        p = tmp_path / "pref.joblib"
        save_artifact(a, p)
        loaded = load_artifact(p)
        assert loaded.artifact_sha256 == a.artifact_sha256
        assert loaded.train_size == a.train_size
        assert loaded.input_sha256 == a.input_sha256

    def test_inference_batch_under_100ms(self) -> None:
        df = _synthetic_df()
        model = PreferenceModel(seed=42)
        model.fit(df, 2024)
        feats = [
            PreferenceFeatures(f"P{i:05d}", "T0", 2400 - i * 5, 0.5, 1, 0, 0) for i in range(1000)
        ]
        start = time.perf_counter()
        proba = model.predict_proba(feats)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"Too slow: {elapsed_ms:.0f}ms for 1000 features"
        assert proba.shape[0] == 1000
        # Probabilities sum to 1.0 per row (sanity self-review checkpoint)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases / structural guards
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases : empty saison + missing required columns."""

    def test_fit_empty_saison_raises(self) -> None:
        df = _synthetic_df(saison=2024)
        model = PreferenceModel(seed=42)
        with pytest.raises(ValueError, match="no rows"):
            model.fit(df, saison=1999)

    def test_fit_missing_blanc_elo_raises(self) -> None:
        df = _synthetic_df().drop(columns=["blanc_elo"])
        model = PreferenceModel(seed=42)
        with pytest.raises((KeyError, ValueError)):
            model.fit(df, 2024)


# ---------------------------------------------------------------------------
# ISO 24027 — bias gate
# ---------------------------------------------------------------------------
class TestBiasGate:
    """ISO 24027 : per-gender recall gap > 0.10 -> BiasViolationError."""

    def test_bias_gate_fails_on_synthetic_gap(self) -> None:
        df = _synthetic_df_gender_biased()
        model = PreferenceModel(seed=42)
        with pytest.raises(BiasViolationError) as excinfo:
            model.fit(df, 2024)
        msg = str(excinfo.value)
        assert "M" in msg and "F" in msg

    def test_bias_gate_passes_without_sexe(self) -> None:
        """No `sexe` column -> bias gate gracefully skipped (backward-compat)."""
        df = _synthetic_df()
        model = PreferenceModel(seed=42)
        # Should NOT raise (no sexe column)
        artifact = model.fit(df, 2024)
        assert artifact.bias_gate_skipped is True
