"""Tests Architecture Metrics - ISO 29119/25010.

Document ID: ALICE-TEST-ARCH-METRICS
Version: 1.0.0
Tests: 9

Classes:
- TestCalculateCoupling: Tests couplage (3 tests)
- TestDetectCircularImports: Tests cycles (3 tests)
- TestCalculateHealthScore: Tests score santé (3 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 25010:2011 - System Quality (Architecture)
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from scripts.architecture.metrics import (
    calculate_coupling,
    calculate_health_score,
    detect_circular_imports,
)


class TestCalculateCoupling:
    """Tests pour calculate_coupling."""

    def test_empty_deps(self):
        """Gère deps vides."""
        result = calculate_coupling({})
        assert result == []

    def test_calculates_efferent(self):
        """Calcule couplage efférent."""
        deps = {"scripts/a.py": ["scripts", "pandas", "numpy"]}
        result = calculate_coupling(deps)

        assert len(result) == 1
        assert result[0]["efferent"] == 3

    def test_sorts_by_total(self):
        """Trie par couplage total décroissant."""
        deps = {
            "scripts/small.py": ["a"],
            "scripts/big.py": ["a", "b", "c", "d", "e"],
        }
        result = calculate_coupling(deps)

        assert result[0]["file"] == "scripts/big.py"


class TestDetectCircularImports:
    """Tests pour detect_circular_imports."""

    def test_no_cycles(self):
        """Détecte absence de cycles."""
        deps = {
            "app/main.py": ["services"],
            "services/core.py": [],
        }
        result = detect_circular_imports(deps)
        assert result == []

    def test_detects_cycle(self):
        """Détecte cycle simple."""
        deps = {
            "app/main.py": ["services"],
            "services/core.py": ["app"],
        }
        result = detect_circular_imports(deps)

        assert len(result) == 1
        assert set(result[0]) == {"app", "services"}

    def test_ignores_self_deps(self):
        """Ignore auto-dépendances."""
        deps = {
            "app/main.py": ["app"],  # Import interne
        }
        result = detect_circular_imports(deps)
        assert result == []


class TestCalculateHealthScore:
    """Tests pour calculate_health_score."""

    def test_perfect_score(self):
        """Score 100 pour architecture saine."""
        coupling = [{"file": "a.py", "total": 5}]
        score, issues = calculate_health_score(coupling, [], 3.0, 5)

        assert score == 100
        assert len(issues) == 0

    def test_penalizes_high_coupling(self):
        """Pénalise couplage élevé."""
        coupling = [
            {"file": "a.py", "total": 20},
            {"file": "b.py", "total": 18},
            {"file": "c.py", "total": 16},
            {"file": "d.py", "total": 17},
        ]
        score, issues = calculate_health_score(coupling, [], 5.0, 8)

        assert score < 100
        assert any("fortement couple" in i for i in issues)

    def test_penalizes_circular_deps(self):
        """Pénalise dépendances circulaires."""
        coupling = []
        circular = [["app", "services"], ["services", "core"]]

        score, issues = calculate_health_score(coupling, circular, 3.0, 5)

        assert score <= 80  # -10 per cycle
        assert any("circulaire" in i for i in issues)
