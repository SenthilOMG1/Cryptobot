"""
EnsembleEvaluator — Dynamically adjusts model weights based on live performance.

Scores each model using profit-weighted accuracy on recent predictions,
then converts scores to weights via softmax with temperature control.
Uses EMA blending to prevent weight oscillation.

Key concepts:
- Profit-weighted scoring: A model that's right on big moves gets more credit
- Exponential decay: Recent predictions matter more than old ones
- Temperature: Controls how aggressively weights diverge (high=conservative)
- EMA blending: Smooths weight transitions (prevents whiplash)
- Bounds: No model goes below 0.10 or above 0.60
"""

import json
import logging
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"xgb": 0.45, "rl": 0.25, "lstm": 0.30}
WEIGHT_FLOOR = 0.10
WEIGHT_CEILING = 0.60
MIN_OUTCOMES_FOR_UPDATE = 15


class EnsembleEvaluator:

    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        temperature: float = 3.0,
        ema_alpha: float = 0.3,
        decay_rate: float = 0.95,
        state_path: str = "models/evaluator_state.json",
    ):
        self.weights = dict(initial_weights or DEFAULT_WEIGHTS)
        self.temperature = temperature
        self.ema_alpha = ema_alpha       # 0.3 = 30% new, 70% old
        self.decay_rate = decay_rate     # Per-prediction decay for recency weighting
        self.state_path = state_path
        self.weight_history: List[Dict] = []
        self.update_count = 0

        # Try to load saved state
        self._load_state()

    def update_weights(self, recent_outcomes: List[Dict]) -> Dict[str, float]:
        """
        Recalculate model weights from recent prediction outcomes.

        Args:
            recent_outcomes: List of dicts from PredictionTracker.get_recent_outcomes()
                Each dict has: xgb_action, xgb_conf, rl_action, rl_conf,
                lstm_action, lstm_conf, actual_return_6h, actual_label, etc.

        Returns:
            Updated weights dict
        """
        if len(recent_outcomes) < MIN_OUTCOMES_FOR_UPDATE:
            logger.info(
                f"Only {len(recent_outcomes)} outcomes, need {MIN_OUTCOMES_FOR_UPDATE}. "
                f"Keeping current weights."
            )
            return self.weights

        # Score each model
        xgb_score = self._score_model("xgb", recent_outcomes)
        rl_score = self._score_model("rl", recent_outcomes)
        lstm_score = self._score_model("lstm", recent_outcomes)

        scores = {"xgb": xgb_score, "rl": rl_score, "lstm": lstm_score}
        logger.info(f"Model scores: {scores}")

        # Convert scores to weights via softmax with temperature
        raw_weights = self._softmax_weights(scores)

        # Clip to bounds
        clipped = self._clip_weights(raw_weights)

        # EMA blend with current weights
        new_weights = {}
        for model in self.weights:
            new_weights[model] = round(
                (1 - self.ema_alpha) * self.weights[model]
                + self.ema_alpha * clipped[model],
                4,
            )

        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}

        # Log the change
        old_weights = dict(self.weights)
        self.weights = new_weights
        self.update_count += 1

        self.weight_history.append({
            "timestamp": datetime.now().isoformat(),
            "update_num": self.update_count,
            "scores": scores,
            "old_weights": old_weights,
            "new_weights": new_weights,
            "outcomes_used": len(recent_outcomes),
        })

        # Keep only last 50 history entries
        if len(self.weight_history) > 50:
            self.weight_history = self.weight_history[-50:]

        self._save_state()

        logger.info(f"Weights updated: {old_weights} → {new_weights}")
        return new_weights

    def _score_model(self, model_name: str, outcomes: List[Dict]) -> float:
        """
        Calculate profit-weighted performance score for one model.

        Score = sum(actual_return * model_confidence * direction_agreement * decay)

        A model gets positive score when:
        - It predicted the RIGHT direction AND the move was large
        - It was confident AND right

        A model gets negative score when:
        - It predicted the WRONG direction with high confidence
        """
        action_key = f"{model_name}_action"
        conf_key = f"{model_name}_conf"

        total_score = 0.0
        n = len(outcomes)

        for i, outcome in enumerate(outcomes):
            model_action = outcome.get(action_key, 0)
            model_conf = outcome.get(conf_key, 0.5) or 0.5
            actual_return = outcome.get("actual_return_6h", 0) or 0
            actual_label = outcome.get("actual_label", 0)

            if model_action == 0:
                # Model said HOLD
                # Small positive score if market was flat (HOLD was correct)
                # Small negative if market moved and model missed it
                if actual_label == 0:
                    score = 0.002 * model_conf  # Correct hold
                else:
                    score = -abs(actual_return) * 0.3 * model_conf  # Missed opportunity
            else:
                # Model said BUY or SELL
                if model_action == actual_label:
                    # Correct direction — reward proportional to move size
                    score = abs(actual_return) * model_conf
                elif actual_label == 0:
                    # Model predicted direction but market was flat
                    # Small penalty for unnecessary trade
                    score = -0.001 * model_conf
                else:
                    # Wrong direction — penalty proportional to move and confidence
                    score = -abs(actual_return) * model_conf * 1.5  # Extra penalty for being wrong

            # Apply recency decay (most recent = full weight, older = less)
            decay = self.decay_rate ** i
            total_score += score * decay

        return round(total_score, 6)

    def _softmax_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert raw scores to weights using softmax with temperature."""
        models = list(scores.keys())
        values = [scores[m] / self.temperature for m in models]

        # Numerical stability: subtract max
        max_val = max(values)
        exp_values = [math.exp(v - max_val) for v in values]
        total = sum(exp_values)

        weights = {m: exp_values[i] / total for i, m in enumerate(models)}
        return weights

    def _clip_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Clip weights to [FLOOR, CEILING] and renormalize."""
        clipped = {
            k: max(WEIGHT_FLOOR, min(WEIGHT_CEILING, v))
            for k, v in weights.items()
        }
        total = sum(clipped.values())
        return {k: v / total for k, v in clipped.items()}

    def get_weights(self) -> Dict[str, float]:
        """Return current model weights."""
        return dict(self.weights)

    def get_weight_history(self) -> List[Dict]:
        """Return history of weight changes."""
        return list(self.weight_history)

    def get_status(self) -> Dict:
        """Return evaluator status for monitoring."""
        return {
            "current_weights": self.weights,
            "update_count": self.update_count,
            "temperature": self.temperature,
            "ema_alpha": self.ema_alpha,
            "last_update": (
                self.weight_history[-1]["timestamp"]
                if self.weight_history else "never"
            ),
        }

    def _save_state(self):
        """Persist evaluator state to disk."""
        state = {
            "weights": self.weights,
            "update_count": self.update_count,
            "temperature": self.temperature,
            "ema_alpha": self.ema_alpha,
            "weight_history": self.weight_history[-20:],  # Last 20 entries
            "saved_at": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Restore evaluator state from disk."""
        if not os.path.exists(self.state_path):
            logger.info("No saved evaluator state, using defaults")
            return

        try:
            with open(self.state_path, "r") as f:
                state = json.load(f)
            self.weights = state.get("weights", self.weights)
            self.update_count = state.get("update_count", 0)
            self.weight_history = state.get("weight_history", [])
            logger.info(
                f"Loaded evaluator state: weights={self.weights}, "
                f"updates={self.update_count}"
            )
        except Exception as e:
            logger.warning(f"Failed to load evaluator state: {e}")
