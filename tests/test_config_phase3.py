"""Tests for Phase 3 config settings.

ISO Compliance:
- ISO/IEC 27034 - Pydantic Settings validation
- ISO/IEC 29119 - Test coverage
"""

from app.config import get_settings


def test_phase3_settings_defaults() -> None:
    """Phase 3 Settings expose RuleEngine + ALI data defaults."""
    s = get_settings()
    assert s.ffe_rules_dir == "./config/ffe_rules"
    assert s.ali_cache_max_age_days == 7
    assert s.joueurs_parquet == "./data/joueurs.parquet"
    assert s.echiquiers_parquet == "./data/echiquiers.parquet"
    assert s.recency_decay_lambda == 0.9
    assert s.streak_lag_window == 3
