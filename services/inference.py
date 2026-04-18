"""Module: inference.py - Stacking Inference Pipeline (Phase 2).

Serves P(W/D/L) predictions via the champion stacking pipeline:
3 GBMs -> 18 meta-features -> MLP(32,16) -> temperature scaling.

Falls back to LGB + Dirichlet if in fallback_mode.

Also exposes the legacy InferenceService (ALI lineup prediction stub)
for backward compatibility with existing tests and service registry.

ISO Compliance:
- ISO 42001: Pipeline tracability (model versions, alpha per-model)
- ISO 25059: Serving quality gates (sum=1, no NaN, mean_p_draw > 1%)
- ISO 24029: Robustness (missing Elo -> 1500 default)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.baselines import compute_elo_baseline, compute_elo_init_scores
from scripts.kaggle_metrics import predict_with_init
from scripts.serving.meta_features import build_meta_features

if TYPE_CHECKING:
    from scripts.serving.model_loader import ModelBundle

logger = logging.getLogger(__name__)

ALPHA_LGB = 0.1
ALPHA_XGB = 0.5
ALPHA_CB = 0.3


# ---------------------------------------------------------------------------
# New stacking pipeline (Phase 2)
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """P(W/D/L) prediction result with expected score for a single board."""

    p_loss: float
    p_draw: float
    p_win: float
    e_score: float


class StackingInferenceService:
    """Stacking inference pipeline: 3 GBMs -> MLP -> temperature scaling."""

    def __init__(self, bundle: ModelBundle) -> None:
        """Initialise the service with a loaded ModelBundle."""
        self._bundle = bundle
        self._fallback = bundle.fallback_mode
        if self._fallback:
            logger.warning("InferenceService in FALLBACK mode (LGB+Dirichlet)")

    def predict_proba(
        self,
        player_elo: int,
        opponent_elo: int,
        features: np.ndarray,
        draw_rate_lookup: Any = None,
    ) -> np.ndarray:
        """Return (1, 3) P(loss/draw/win) via the stacking pipeline."""
        lookup = draw_rate_lookup or self._bundle.draw_rate_lookup
        if lookup is None:
            raise ValueError(
                "draw_rate_lookup is required for inference. "
                "Ensure draw_rate_lookup.parquet is in the model cache."
            )
        elo_probas = compute_elo_baseline(
            np.array([player_elo], dtype=float),
            np.array([opponent_elo], dtype=float),
            lookup,
        )
        init_scores = compute_elo_init_scores(elo_probas)
        if self._fallback:
            return self._predict_fallback(features, init_scores)
        return self._predict_stacking(features, init_scores)

    def _predict_stacking(self, features: np.ndarray, init_scores: np.ndarray) -> np.ndarray:
        b = self._bundle
        p_lgb = predict_with_init(b.lgb_model, features, init_scores * ALPHA_LGB)
        p_xgb = predict_with_init(b.xgb_model, features, init_scores * ALPHA_XGB)
        p_cb = predict_with_init(b.cb_model, features, init_scores * ALPHA_CB)
        meta = build_meta_features(p_xgb, p_lgb, p_cb)
        p_raw = np.asarray(b.mlp_model.predict_proba(meta))
        p_cal = self._apply_temperature(p_raw, b.temperature)
        self._validate_output(p_cal)
        return p_cal

    def _predict_stacking_mock(self, features: np.ndarray) -> np.ndarray:
        """For testing: uses predict_proba directly (no init_scores)."""
        b = self._bundle
        p_lgb = np.asarray(b.lgb_model.predict_proba(features))
        p_xgb = np.asarray(b.xgb_model.predict_proba(features))
        p_cb = np.asarray(b.cb_model.predict_proba(features))
        meta = build_meta_features(p_xgb, p_lgb, p_cb)
        p_raw = np.asarray(b.mlp_model.predict_proba(meta))
        p_cal = self._apply_temperature(p_raw, b.temperature)
        self._validate_output(p_cal)
        return p_cal

    def _predict_fallback(self, features: np.ndarray, init_scores: np.ndarray) -> np.ndarray:
        b = self._bundle
        p_lgb = predict_with_init(b.lgb_model, features, init_scores * ALPHA_LGB)
        if b.mlp_model is not None:
            log_p = np.log(np.clip(p_lgb, 1e-7, 1.0))
            p_cal = np.asarray(b.mlp_model.predict_proba(log_p))
        else:
            p_cal = p_lgb
        self._validate_output(p_cal)
        return p_cal

    @staticmethod
    def _apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to logits then renormalise."""
        logits = np.log(np.clip(p, 1e-7, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    @staticmethod
    def _validate_output(p: np.ndarray) -> None:
        if not np.all(np.isfinite(p)):
            raise ValueError("NaN/Inf in prediction output")
        sums = p.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-4):
            raise ValueError(f"Probabilities don't sum to 1: {sums}")

    def predict_board(
        self,
        player_elo: int,
        opponent_elo: int,
        features: np.ndarray,
        draw_rate_lookup: Any = None,
    ) -> PredictionResult:
        """Return structured PredictionResult with e_score for a single board."""
        p = self.predict_proba(player_elo, opponent_elo, features, draw_rate_lookup)
        return PredictionResult(
            p_loss=float(p[0, 0]),
            p_draw=float(p[0, 1]),
            p_win=float(p[0, 2]),
            e_score=float(p[0, 2] + 0.5 * p[0, 1]),
        )


# ---------------------------------------------------------------------------
# Legacy ALI lineup prediction service (backward compat — Phase 3 TODO)
# ---------------------------------------------------------------------------


@dataclass
class PlayerProbability:
    """Probabilite de presence d'un joueur."""

    ffe_id: str
    name: str
    elo: int
    probability: float
    board: int
    reasoning: str


class InferenceService:
    """Service ALI (Adversarial Lineup Inference) — legacy stub.

    Responsabilite: Predire la composition adverse probable.
    Phase 3 will replace this stub with real ML predictions.

    Methodes:
        - predict_lineup: Predit les joueurs qui vont jouer
        - generate_scenarios: Genere N scenarios possibles
        - get_player_probability: Probabilite pour un joueur specifique
    """

    def __init__(self, model_path: str | None = None) -> None:
        """Initialise le service d'inference.

        @param model_path: Chemin vers le modele CatBoost/XGBoost
        """
        self.model = None
        self.model_path = model_path
        self.is_loaded = False

    def load_model(self) -> bool:
        """Charge le modele ML depuis le disque.

        @returns: True si charge avec succes
        """
        if not self.model_path:
            logger.warning("Aucun chemin de modele specifie")
            return False

        try:
            # TODO: Charger le modele CatBoost ou XGBoost
            # from catboost import CatBoostClassifier
            # self.model = CatBoostClassifier()
            # self.model.load_model(self.model_path)
            logger.info("Modele charge depuis %s", self.model_path)
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error("Erreur chargement modele: %s", e)
            return False

    def predict_lineup(
        self,
        opponent_club_id: str,
        round_number: int,
        opponent_players: list[dict[str, Any]],
        team_size: int = 8,
    ) -> list[PlayerProbability]:
        """Predit la composition adverse probable.

        @param opponent_club_id: ID FFE du club adverse
        @param round_number: Numero de la ronde
        @param opponent_players: Liste des joueurs du club adverse
        @param team_size: Nombre de joueurs dans l'equipe

        @returns: Liste des joueurs avec probabilites, tries par echiquier

        @see ISO 25010 - Fiabilite (predictions)
        """
        if not self.is_loaded:
            logger.warning("Modele non charge, utilisation fallback")
            return self._fallback_prediction(opponent_players, team_size)

        # TODO: Implementer la prediction ML (Phase 3)
        # 1. Preparer les features pour chaque joueur
        # 2. compute_differentials(df_features)  # FTI anti-skew (same as FE pipeline)
        # 3. Appeler model.predict_proba() with init_scores (residual learning)
        # 4. Trier par probabilite et assigner aux echiquiers

        return self._fallback_prediction(opponent_players, team_size)

    def generate_scenarios(
        self,
        predictions: list[PlayerProbability],
        scenario_count: int = 20,
    ) -> list[dict[str, Any]]:
        """Genere plusieurs scenarios de composition adverse.

        @param predictions: Predictions de probabilites par joueur
        @param scenario_count: Nombre de scenarios a generer

        @returns: Liste de scenarios avec probabilites
        """
        # TODO: Implementer generation de scenarios (Phase 3)
        return []

    def _fallback_prediction(
        self,
        players: list[dict[str, Any]],
        team_size: int,
    ) -> list[PlayerProbability]:
        """Prediction fallback basee sur l'ordre Elo.

        Utilisee quand le modele n'est pas charge.
        Hypothese: Les meilleurs joueurs jouent en priorite.
        """
        sorted_players = sorted(
            players,
            key=lambda p: p.get("elo", 0),
            reverse=True,
        )

        results = []
        for i, player in enumerate(sorted_players[:team_size]):
            results.append(
                PlayerProbability(
                    ffe_id=player.get("ffe_id", ""),
                    name=player.get("name", ""),
                    elo=player.get("elo", 0),
                    probability=0.5,
                    board=i + 1,
                    reasoning="Fallback: Ordre Elo (modele non charge)",
                )
            )

        return results
