"""Tests for feature store assembly (ISO 29119 / ISO 5259).

Document ID: ALICE-TEST-FEATURE-STORE
Version: 1.0.0
Count: 4
"""

import pandas as pd
import pytest

from services.feature_store import FeatureStore


class TestFeatureStore:
    """Test feature assembly from parquets."""

    def test_assemble_returns_dataframe(self, tmp_path):
        joueur_df = pd.DataFrame(
            {
                "joueur_nom": ["Player1"],
                "blanc_elo": [1800],
                "win_rate_normal_blanc": [0.45],
            }
        )
        joueur_df.to_parquet(tmp_path / "joueur_features.parquet")
        store = FeatureStore(tmp_path)
        store.load()
        result = store.assemble(
            player_name="Player1",
            player_elo=1800,
            opponent_elo=1750,
            context={"division": 3, "ronde": 5},
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.shape[1] > 0

    def test_unknown_player_returns_defaults(self, tmp_path):
        joueur_df = pd.DataFrame({"joueur_nom": ["Player1"], "blanc_elo": [1800]})
        joueur_df.to_parquet(tmp_path / "joueur_features.parquet")
        store = FeatureStore(tmp_path)
        store.load()
        result = store.assemble(
            player_name="UNKNOWN",
            player_elo=1500,
            opponent_elo=1500,
            context={},
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_age_hours(self, tmp_path):
        joueur_df = pd.DataFrame({"joueur_nom": []})
        joueur_df.to_parquet(tmp_path / "joueur_features.parquet")
        store = FeatureStore(tmp_path)
        store.load()
        assert store.age_hours < 0.01  # just loaded

    def test_not_loaded_raises(self, tmp_path):
        store = FeatureStore(tmp_path)
        with pytest.raises(RuntimeError, match="not loaded"):
            store.assemble("X", 1500, 1500, {})
