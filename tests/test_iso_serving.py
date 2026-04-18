"""ISO compliance tests for serving pipeline (ISO 5055/24029/25010)."""

import time
from pathlib import Path

import numpy as np


class TestISO5055CodeQuality:
    """ISO 5055: file size and complexity limits."""

    def test_file_sizes_under_300_lines(self):
        """No Phase 2 file exceeds 300 lines."""
        phase2_files = [
            "scripts/serving/meta_features.py",
            "scripts/serving/model_loader.py",
            "services/feature_store.py",
            "services/inference.py",
            "services/ffe_rules.py",
        ]
        for f in phase2_files:
            p = Path(f)
            if p.exists():
                lines = len(p.read_text().splitlines())
                assert lines <= 300, f"{f} has {lines} lines (max 300, ISO 5055)"


class TestISO24029Robustness:
    """ISO 24029: degraded inputs produce valid output."""

    def test_meta_features_with_zeros(self):
        """Meta-features handles zero probabilities."""
        from scripts.serving.meta_features import build_meta_features

        p = np.array([[0.0, 0.0, 1.0]])
        result = build_meta_features(p, p, p)
        assert np.all(np.isfinite(result))

    def test_meta_features_with_uniform(self):
        """Meta-features handles uniform distribution."""
        from scripts.serving.meta_features import build_meta_features

        p = np.array([[1 / 3, 1 / 3, 1 / 3]])
        result = build_meta_features(p, p, p)
        assert result.shape == (1, 18)


class TestISO25010Latency:
    """ISO 25010: meta-feature computation under 10ms."""

    def test_meta_features_fast(self):
        from scripts.serving.meta_features import build_meta_features

        p = np.random.dirichlet([1, 1, 1], size=100)
        t0 = time.time()
        for _ in range(100):
            build_meta_features(p, p, p)
        elapsed = (time.time() - t0) / 100
        assert elapsed < 0.01, f"Meta-features took {elapsed:.4f}s (max 0.01)"
