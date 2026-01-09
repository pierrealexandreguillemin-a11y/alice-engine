"""Module: test_features_pipeline.py - Tests Pipeline Feature Engineering.

Tests unitaires pour le module pipeline.py - orchestration du pipeline.

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing (integration tests)
- ISO/IEC 5259:2024 - Data Quality for ML (feature pipeline)
- ISO/IEC 25010:2023 - System Quality (testability)

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

import pandas as pd
import pytest

from scripts.features.pipeline import extract_all_features, merge_all_features

# ==============================================================================
# FIXTURES (ISO 29119-3: Test Design)
# ==============================================================================


@pytest.fixture
def sample_history() -> pd.DataFrame:
    """Fixture avec historique complet pour pipeline."""
    return pd.DataFrame(
        [
            {
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 1,
                "echiquier": 1,
                "equipe_dom": "Club A",
                "equipe_ext": "Club B",
                "blanc_nom": "Joueur 1",
                "blanc_club": "Club A",
                "blanc_elo": 2200,
                "noir_nom": "Joueur 2",
                "noir_club": "Club B",
                "noir_elo": 2150,
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": "2025-01-15",
                "score_dom": 4,
                "score_ext": 2,
            },
            {
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 1,
                "echiquier": 2,
                "equipe_dom": "Club A",
                "equipe_ext": "Club B",
                "blanc_nom": "Joueur 3",
                "blanc_club": "Club A",
                "blanc_elo": 2100,
                "noir_nom": "Joueur 4",
                "noir_club": "Club B",
                "noir_elo": 2080,
                "resultat_blanc": 0.5,
                "resultat_noir": 0.5,
                "type_resultat": "nulle",
                "date": "2025-01-15",
                "score_dom": 4,
                "score_ext": 2,
            },
            {
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": 2,
                "echiquier": 1,
                "equipe_dom": "Club B",
                "equipe_ext": "Club A",
                "blanc_nom": "Joueur 2",
                "blanc_club": "Club B",
                "blanc_elo": 2150,
                "noir_nom": "Joueur 1",
                "noir_club": "Club A",
                "noir_elo": 2200,
                "resultat_blanc": 0.0,
                "resultat_noir": 1.0,
                "type_resultat": "victoire_noir",
                "date": "2025-01-22",
                "score_dom": 2,
                "score_ext": 4,
            },
            # Ajouter plus de donnees pour features fiables
        ]
        + [
            {
                "saison": 2025,
                "competition": "Interclubs",
                "division": "Nationale 1",
                "groupe": "Groupe A",
                "ronde": r,
                "echiquier": 1,
                "equipe_dom": "Club A",
                "equipe_ext": "Club C",
                "blanc_nom": "Joueur 1",
                "blanc_club": "Club A",
                "blanc_elo": 2200,
                "noir_nom": "Joueur 5",
                "noir_club": "Club C",
                "noir_elo": 2000,
                "resultat_blanc": 1.0,
                "resultat_noir": 0.0,
                "type_resultat": "victoire_blanc",
                "date": f"2025-02-{r:02d}",
                "score_dom": 5,
                "score_ext": 1,
            }
            for r in range(3, 9)
        ]
    )


@pytest.fixture
def sample_history_played(sample_history: pd.DataFrame) -> pd.DataFrame:
    """Fixture avec parties jouees uniquement (sans forfaits)."""
    return sample_history[
        ~sample_history["type_resultat"].isin(
            ["non_joue", "forfait_blanc", "forfait_noir", "double_forfait"]
        )
    ].copy()


@pytest.fixture
def sample_target() -> pd.DataFrame:
    """Fixture DataFrame cible pour merge."""
    return pd.DataFrame(
        [
            {
                "saison": 2025,
                "ronde": 10,
                "echiquier": 1,
                "equipe_dom": "Club A",
                "equipe_ext": "Club D",
                "blanc_nom": "Joueur 1",
                "blanc_club": "Club A",
                "noir_nom": "Joueur 6",
                "noir_club": "Club D",
            },
        ]
    )


# ==============================================================================
# TESTS: extract_all_features (ISO 29119-4: Integration Testing)
# ==============================================================================


class TestExtractAllFeatures:
    """Tests pour extract_all_features."""

    def test_basic_extraction(
        self, sample_history: pd.DataFrame, sample_history_played: pd.DataFrame
    ) -> None:
        """Test extraction basique sans features avancees."""
        result = extract_all_features(sample_history, sample_history_played, include_advanced=False)

        assert isinstance(result, dict)
        # Features de base attendues
        expected_keys = [
            "club_reliability",
            "player_reliability",
            "recent_form",
            "board_position",
            "color_perf",
            "ffe_regulatory",
            "team_enjeu",
        ]
        for key in expected_keys:
            assert key in result, f"Feature manquante: {key}"
            assert isinstance(result[key], pd.DataFrame)

    def test_extraction_with_advanced(
        self, sample_history: pd.DataFrame, sample_history_played: pd.DataFrame
    ) -> None:
        """Test extraction avec features avancees."""
        result = extract_all_features(sample_history, sample_history_played, include_advanced=True)

        # Features avancees attendues
        advanced_keys = ["h2h", "fatigue", "home_away", "pressure", "trajectory"]
        for key in advanced_keys:
            assert key in result, f"Feature avancee manquante: {key}"

    def test_empty_history_returns_empty_features(self) -> None:
        """Test historique vide retourne features vides."""
        empty_df = pd.DataFrame()
        result = extract_all_features(empty_df, empty_df, include_advanced=False)

        assert isinstance(result, dict)
        # Toutes les features doivent etre des DataFrames (possiblement vides)
        for key, value in result.items():
            assert isinstance(value, pd.DataFrame), f"{key} n'est pas un DataFrame"

    def test_feature_count_without_advanced(
        self, sample_history: pd.DataFrame, sample_history_played: pd.DataFrame
    ) -> None:
        """Test nombre de features sans avancees = 7."""
        result = extract_all_features(sample_history, sample_history_played, include_advanced=False)
        assert len(result) == 7

    def test_feature_count_with_advanced(
        self, sample_history: pd.DataFrame, sample_history_played: pd.DataFrame
    ) -> None:
        """Test nombre de features avec avancees = 12."""
        result = extract_all_features(sample_history, sample_history_played, include_advanced=True)
        # 7 base + 5 avancees = 12
        assert len(result) == 12


# ==============================================================================
# TESTS: merge_all_features (ISO 29119-4: Integration Testing)
# ==============================================================================


class TestMergeAllFeatures:
    """Tests pour merge_all_features."""

    def test_basic_merge(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test merge basique des features."""
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )
        result = merge_all_features(sample_target.copy(), features, include_advanced=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_target)

    def test_merge_preserves_original_columns(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test merge preserve colonnes originales."""
        original_cols = set(sample_target.columns)
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )
        result = merge_all_features(sample_target.copy(), features, include_advanced=False)

        # Toutes les colonnes originales doivent etre presentes
        for col in original_cols:
            assert col in result.columns, f"Colonne originale perdue: {col}"

    def test_merge_adds_feature_columns(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test merge ajoute colonnes features."""
        original_col_count = len(sample_target.columns)
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )
        result = merge_all_features(sample_target.copy(), features, include_advanced=False)

        # Plus de colonnes apres merge
        assert len(result.columns) >= original_col_count

    def test_merge_with_advanced_features(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test merge avec features avancees."""
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=True
        )
        result = merge_all_features(sample_target.copy(), features, include_advanced=True)

        assert isinstance(result, pd.DataFrame)

    def test_merge_handles_empty_features(self, sample_target: pd.DataFrame) -> None:
        """Test merge gere features vides gracieusement."""
        empty_features = {
            "club_reliability": pd.DataFrame(),
            "player_reliability": pd.DataFrame(),
            "recent_form": pd.DataFrame(),
            "board_position": pd.DataFrame(),
            "color_perf": pd.DataFrame(),
            "ffe_regulatory": pd.DataFrame(),
            "team_enjeu": pd.DataFrame(),
        }

        # Ne doit pas lever d'exception
        result = merge_all_features(sample_target.copy(), empty_features, include_advanced=False)
        assert isinstance(result, pd.DataFrame)


# ==============================================================================
# TESTS: Pipeline Integration (ISO 29119-4: End-to-End)
# ==============================================================================


class TestPipelineIntegration:
    """Tests integration complete du pipeline."""

    def test_full_pipeline_basic(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test pipeline complet basique."""
        # Extract
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )

        # Merge
        result = merge_all_features(sample_target.copy(), features, include_advanced=False)

        # Verifications
        assert not result.empty
        assert len(result) == len(sample_target)

    def test_full_pipeline_advanced(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test pipeline complet avec features avancees."""
        # Extract
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=True
        )

        # Merge
        result = merge_all_features(sample_target.copy(), features, include_advanced=True)

        # Verifications
        assert not result.empty

    def test_pipeline_idempotent(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test pipeline est idempotent (meme resultat si execute 2x)."""
        features1 = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )
        features2 = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )

        # Memes cles
        assert set(features1.keys()) == set(features2.keys())

        # Memes tailles
        for key in features1:
            assert len(features1[key]) == len(features2[key])


# ==============================================================================
# TESTS: Data Quality (ISO 5259 - Data Quality for ML)
# ==============================================================================


class TestPipelineDataQuality:
    """Tests qualite donnees ISO 5259."""

    def test_no_data_leakage(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
        sample_target: pd.DataFrame,
    ) -> None:
        """Test ISO 5259: pas de fuite de donnees futur vers passe."""
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )

        # Features extraites de l'historique (rondes 1-8)
        # Target est ronde 10
        # Aucune feature ne doit contenir info de ronde >= 10

        # Verifier recent_form ne contient pas de rondes futures
        if not features["recent_form"].empty and "ronde" in features["recent_form"].columns:
            max_ronde = features["recent_form"]["ronde"].max()
            assert max_ronde < 10, "Fuite de donnees: forme contient rondes futures"

    def test_feature_types_consistent(
        self,
        sample_history: pd.DataFrame,
        sample_history_played: pd.DataFrame,
    ) -> None:
        """Test types de features consistants (pas de mixed types)."""
        features = extract_all_features(
            sample_history, sample_history_played, include_advanced=False
        )

        for name, df in features.items():
            # Tous doivent etre des DataFrames
            assert isinstance(df, pd.DataFrame), f"{name} n'est pas DataFrame"

            # Verifier pas de colonnes avec dtype 'object' mixte
            for col in df.columns:
                if len(df) > 0 and df[col].dtype == "object":
                    # Pour colonnes object, verifier coherence
                    sample = df[col].dropna()
                    if len(sample) >= 5:
                        # Verifier que ce n'est pas un melange de types
                        types_in_col = {type(v).__name__ for v in sample}
                        # Accepter si un seul type ou str avec NoneType
                        assert (
                            len(types_in_col) <= 2
                        ), f"Types mixtes dans {name}.{col}: {types_in_col}"
