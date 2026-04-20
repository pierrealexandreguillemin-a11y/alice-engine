"""Backtest harness orchestrator (walk-forward).

Plan 3 P3-Task 1. Bootstrap services Plan 1+2 + iterate matches échantillonnés.

Document ID: ALICE-BACKTEST-HARNESS
Version: 1.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import settings
from scripts.backtest.run_match import BacktestMatchResult, run_backtest_match
from scripts.serving.model_loader import load_models
from services.ali.cache import ALIDataCache
from services.ali.generator import ScenarioGenerator
from services.ali.history import HistoryEnricher
from services.ali.pool_loader import PlayerPoolLoader
from services.ali.verifiability import VerifiabilityClassifier
from services.feature_store import FeatureStore
from services.ffe.rule_engine import RuleEngine
from services.inference import StackingInferenceService

logger = logging.getLogger(__name__)


class BacktestHarness:
    """Bootstrap services Plan 1+2 pour backtest.

    Réutilise même chargement que app/main.py::lifespan mais sans FastAPI.
    """

    def __init__(self) -> None:
        """Initialise attributs vides — setup() fait le chargement."""
        self.cache: ALIDataCache | None = None
        self.rule_engine: RuleEngine | None = None
        self.classifier: VerifiabilityClassifier | None = None
        self.scenario_generator: ScenarioGenerator | None = None
        self.inference: StackingInferenceService | None = None
        self.feature_store: FeatureStore | None = None

    def setup(self) -> None:
        """Charge tous services. Raise si données absentes (fail-fast)."""
        self.cache = ALIDataCache.load_from_parquets(
            Path(settings.joueurs_parquet),
            Path(settings.echiquiers_parquet),
        )
        self.rule_engine = RuleEngine.from_json_file(
            Path(settings.ffe_rules_dir) / "a02.json",
        )
        self.classifier = VerifiabilityClassifier.from_json_file(
            Path(settings.ffe_rules_dir) / "alice_verifiability.json",
        )
        pool_loader = PlayerPoolLoader(self.cache)
        history_enricher = HistoryEnricher(
            self.cache,
            decay_lambda=settings.recency_decay_lambda,
        )
        self.scenario_generator = ScenarioGenerator(
            engine=self.rule_engine,
            classifier=self.classifier,
            cache=self.cache,
            pool_loader=pool_loader,
            history_enricher=history_enricher,
            decay_lambda=settings.recency_decay_lambda,
        )

        bundle = load_models(
            cache_dir=Path(settings.model_cache_dir),
            hf_repo_id=settings.hf_repo_id,
            download=False,
        )
        self.inference = StackingInferenceService(bundle)

        # Feature store MUST be loaded for strict-mode backtest (ISO 42001).
        # Run scripts/build_feature_store.py if training_mean.parquet absent.
        fs = FeatureStore(Path(settings.feature_store_path))
        fs.load()  # raises if training_mean.parquet missing
        self.feature_store = fs

    def run_match(  # noqa: PLR0913
        self,
        *,
        user_club_id: str,
        opponent_club_id: str,
        saison: int,
        ronde: int,
        nb_rondes_total: int,
        division: str,
        team_size: int,
        user_lineup: list[dict[str, Any]],
        seed: int = 42,
        strict: bool = True,
    ) -> BacktestMatchResult:
        """Run a single backtest match (services pre-setup)."""
        if self.scenario_generator is None or self.inference is None:
            raise RuntimeError("BacktestHarness not setup() yet")
        return run_backtest_match(
            user_club_id=user_club_id,
            opponent_club_id=opponent_club_id,
            saison=saison,
            ronde=ronde,
            nb_rondes_total=nb_rondes_total,
            division=division,
            team_size=team_size,
            user_lineup=user_lineup,
            scenario_generator=self.scenario_generator,
            inference=self.inference,
            feature_store=self.feature_store,
            seed=seed,
            strict=strict,
        )
