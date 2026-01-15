"""Performance Profiler - ISO 25010 (Performance Efficiency).

Script de profiling pour identifier les goulots d'étranglement.

Usage:
    python scripts/profile_performance.py

ISO Compliance:
- ISO/IEC 25010:2023 - Performance Efficiency
- ISO/IEC 5055:2021 - Code Quality

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path
from pstats import SortKey

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def create_test_dataframe(n_rows: int = 1000) -> pd.DataFrame:
    """Crée un DataFrame de test."""
    return pd.DataFrame(
        {
            "blanc_elo": [1600] * n_rows,
            "noir_elo": [1500] * n_rows,
            "diff_elo": [100] * n_rows,
            "echiquier": list(range(1, n_rows + 1)),
            "niveau": [1] * n_rows,
            "ronde": [1] * n_rows,
            "type_competition": ["national"] * n_rows,
            "division": ["N1"] * n_rows,
            "ligue_code": ["IDF"] * n_rows,
            "blanc_titre": [""] * n_rows,
            "noir_titre": [""] * n_rows,
            "jour_semaine": ["Samedi"] * n_rows,
            "resultat_blanc": [1.0] * n_rows,
        }
    )


def profile_prepare_features() -> str:
    """Profile la fonction prepare_features."""
    from scripts.training.features import prepare_features

    df = create_test_dataframe(5000)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(10):
        prepare_features(df.copy(), fit_encoders=True)

    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
    return stream.getvalue()


def profile_drift_monitor() -> str:
    """Profile le monitoring de drift."""
    import numpy as np

    from scripts.model_registry.drift_monitor import monitor_drift

    # Créer données numériques seulement pour le drift
    n = 5000
    baseline = pd.DataFrame(
        {
            "blanc_elo": np.random.normal(1600, 200, n),
            "noir_elo": np.random.normal(1550, 200, n),
            "diff_elo": np.random.normal(50, 100, n),
        }
    )
    current = pd.DataFrame(
        {
            "blanc_elo": np.random.normal(1650, 200, n),  # Drift
            "noir_elo": np.random.normal(1550, 200, n),
            "diff_elo": np.random.normal(80, 100, n),  # Drift
        }
    )

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(5):
        monitor_drift(baseline, current, "v1.0.0")

    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
    return stream.getvalue()


def main() -> None:
    """Point d'entrée du profiler."""
    print("=" * 60)
    print("ALICE Engine Performance Profiler")
    print("=" * 60)

    print("\n[1/2] Profiling prepare_features...")
    result1 = profile_prepare_features()
    print(result1)

    print("\n[2/2] Profiling drift_monitor...")
    result2 = profile_drift_monitor()
    print(result2)

    print("\n" + "=" * 60)
    print("Profiling terminé.")
    print("=" * 60)


if __name__ == "__main__":
    main()
