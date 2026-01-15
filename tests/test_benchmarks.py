"""Benchmarks Performance - ISO 25010 Performance Efficiency.

Document ID: ALICE-BENCH-PERFORMANCE
Version: 1.1.0
Benchmarks: 4

ISO Compliance:
- ISO/IEC 25010:2023 - Performance Efficiency
  - Time behavior: Response time thresholds enforced
  - Resource utilization: CPU, memory efficiency
- ISO/IEC 29119:2022 - Software Testing (benchmark methodology)

Thresholds (ISO 25010 Time Behavior):
- prepare_features (100 rows): < 50ms
- prepare_features (5000 rows): < 200ms
- compute_psi: < 10ms
- compute_metrics: < 100ms

Usage:
    pytest tests/test_benchmarks.py --benchmark-only
    pytest tests/test_benchmarks.py --benchmark-json=benchmark.json

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

import numpy as np
import pandas as pd
import pytest

# ISO 25010 Performance Thresholds (milliseconds)
THRESHOLD_PREPARE_FEATURES_SMALL_MS = 50
THRESHOLD_PREPARE_FEATURES_LARGE_MS = 200
THRESHOLD_COMPUTE_PSI_MS = 10
THRESHOLD_COMPUTE_METRICS_MS = 100


@pytest.fixture
def sample_dataframe_small() -> pd.DataFrame:
    """DataFrame 100 lignes pour benchmarks rapides."""
    n = 100
    return pd.DataFrame(
        {
            "blanc_elo": [1600] * n,
            "noir_elo": [1500] * n,
            "diff_elo": [100] * n,
            "echiquier": list(range(1, n + 1)),
            "niveau": [1] * n,
            "ronde": [1] * n,
            "type_competition": ["national"] * n,
            "division": ["N1"] * n,
            "ligue_code": ["IDF"] * n,
            "blanc_titre": [""] * n,
            "noir_titre": [""] * n,
            "jour_semaine": ["Samedi"] * n,
            "resultat_blanc": [1.0] * n,
        }
    )


@pytest.fixture
def sample_dataframe_large() -> pd.DataFrame:
    """DataFrame 5000 lignes pour benchmarks réalistes."""
    n = 5000
    return pd.DataFrame(
        {
            "blanc_elo": np.random.randint(1200, 2400, n),
            "noir_elo": np.random.randint(1200, 2400, n),
            "diff_elo": np.random.randint(-500, 500, n),
            "echiquier": np.tile(range(1, 9), n // 8 + 1)[:n],
            "niveau": np.random.randint(1, 5, n),
            "ronde": np.random.randint(1, 12, n),
            "type_competition": np.random.choice(["national", "regional"], n),
            "division": np.random.choice(["N1", "N2", "N3", "R1"], n),
            "ligue_code": np.random.choice(["IDF", "ARA", "OCC"], n),
            "blanc_titre": np.random.choice(["", "FM", "CM"], n),
            "noir_titre": np.random.choice(["", "FM", "CM"], n),
            "jour_semaine": np.random.choice(["Samedi", "Dimanche"], n),
            "resultat_blanc": np.random.choice([0.0, 0.5, 1.0], n),
        }
    )


class TestPrepareFeaturesBenchmark:
    """Benchmarks prepare_features (ISO 25010 Time Behavior)."""

    def test_benchmark_prepare_features_small(
        self, benchmark, sample_dataframe_small: pd.DataFrame
    ) -> None:
        """Benchmark prepare_features sur 100 lignes - seuil 50ms."""
        from scripts.training.features import prepare_features

        result = benchmark(prepare_features, sample_dataframe_small, fit_encoders=True)

        X, y, encoders = result
        assert len(X) == 100

        # ISO 25010 Time Behavior: enforce threshold
        mean_ms = benchmark.stats["mean"] * 1000
        assert (
            mean_ms < THRESHOLD_PREPARE_FEATURES_SMALL_MS
        ), f"Performance regression: {mean_ms:.1f}ms > {THRESHOLD_PREPARE_FEATURES_SMALL_MS}ms"

    def test_benchmark_prepare_features_large(
        self, benchmark, sample_dataframe_large: pd.DataFrame
    ) -> None:
        """Benchmark prepare_features sur 5000 lignes - seuil 200ms."""
        from scripts.training.features import prepare_features

        result = benchmark(prepare_features, sample_dataframe_large, fit_encoders=True)

        X, y, encoders = result
        assert len(X) == 5000

        # ISO 25010 Time Behavior: enforce threshold
        mean_ms = benchmark.stats["mean"] * 1000
        assert (
            mean_ms < THRESHOLD_PREPARE_FEATURES_LARGE_MS
        ), f"Performance regression: {mean_ms:.1f}ms > {THRESHOLD_PREPARE_FEATURES_LARGE_MS}ms"


class TestDriftMonitorBenchmark:
    """Benchmarks drift monitoring (ISO 25010 Time Behavior)."""

    def test_benchmark_compute_psi(self, benchmark) -> None:
        """Benchmark PSI calculation - seuil 10ms."""
        from scripts.model_registry.drift_monitor import compute_psi

        baseline = np.random.normal(1600, 200, 5000)
        current = np.random.normal(1650, 200, 5000)

        result = benchmark(compute_psi, baseline, current)

        assert result >= 0

        # ISO 25010 Time Behavior: enforce threshold
        mean_ms = benchmark.stats["mean"] * 1000
        assert (
            mean_ms < THRESHOLD_COMPUTE_PSI_MS
        ), f"Performance regression: {mean_ms:.1f}ms > {THRESHOLD_COMPUTE_PSI_MS}ms"


class TestMetricsBenchmark:
    """Benchmarks compute_all_metrics (ISO 25010 Time Behavior)."""

    def test_benchmark_compute_metrics(self, benchmark) -> None:
        """Benchmark metrics calculation - seuil 100ms."""
        from scripts.training.metrics import compute_all_metrics

        n = 10000
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        y_proba = np.random.rand(n)

        result = benchmark(compute_all_metrics, y_true, y_pred, y_proba)

        assert result.accuracy >= 0

        # ISO 25010 Time Behavior: enforce threshold
        mean_ms = benchmark.stats["mean"] * 1000
        assert (
            mean_ms < THRESHOLD_COMPUTE_METRICS_MS
        ), f"Performance regression: {mean_ms:.1f}ms > {THRESHOLD_COMPUTE_METRICS_MS}ms"
