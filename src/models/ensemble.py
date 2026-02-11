"""
Ensemble Decision Engine
========================
Combines XGBoost and RL agent predictions through voting.
Only trades when both models agree with high confidence.
"""

import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Trading actions."""
    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass
class TradeDecision:
    """
    Trading decision from ensemble.

    Attributes:
        action: BUY (1), SELL (-1), or HOLD (0)
        confidence: Combined confidence score (0.0 to 1.0)
        xgb_action: XGBoost's vote
        xgb_confidence: XGBoost's confidence
        rl_action: RL agent's vote
        rl_confidence: RL agent's confidence
        reasoning: Explanation of the decision
    """
    action: int
    confidence: float
    xgb_action: int
    xgb_confidence: float
    rl_action: int
    rl_confidence: float
    reasoning: str
    pair: str = ""
    suggested_size_pct: float = 0.0


class EnsembleDecider:
    """
    Ensemble voting system for trading decisions.

    Rules:
    1. Both models must agree on action (BUY/SELL)
    2. Combined confidence must exceed threshold
    3. If models disagree -> HOLD
    4. Higher confidence = larger suggested position size

    This conservative approach prevents bad trades from
    a single model's mistake.
    """

    def __init__(
        self,
        xgb_model,
        rl_agent,
        min_confidence: float = 0.70,
        agreement_required: bool = True
    ):
        """
        Initialize ensemble.

        Args:
            xgb_model: Trained XGBoostPredictor instance
            rl_agent: Trained RLTradingAgent instance
            min_confidence: Minimum confidence to trade (0.0 to 1.0)
            agreement_required: If True, both models must agree to trade
        """
        self.xgb_model = xgb_model
        self.rl_agent = rl_agent
        self.min_confidence = min_confidence
        self.agreement_required = agreement_required

        # Weight for each model (can be adjusted based on performance)
        self.xgb_weight = 0.5
        self.rl_weight = 0.5

        # Track recent accuracy for dynamic weighting
        self.xgb_recent_accuracy = []
        self.rl_recent_accuracy = []

    def get_decision(
        self,
        features: pd.DataFrame,
        portfolio_state: Dict[str, Any],
        pair: str = ""
    ) -> TradeDecision:
        """
        Get ensemble trading decision.

        Args:
            features: DataFrame with calculated features (single row)
            portfolio_state: Dict with balance, position, entry_price, current_price
            pair: Trading pair name for logging

        Returns:
            TradeDecision with action and confidence
        """
        # Get XGBoost prediction
        try:
            xgb_action, xgb_conf = self.xgb_model.predict(features)
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            xgb_action, xgb_conf = Action.HOLD, 0.5

        # Get RL agent prediction
        try:
            # Extract features as numpy array for RL
            feature_cols = self.xgb_model.feature_names
            feature_array = features[feature_cols].values.flatten().astype(np.float32)
            rl_action, rl_conf = self.rl_agent.decide(feature_array, portfolio_state)
        except Exception as e:
            logger.error(f"RL prediction failed: {e}")
            rl_action, rl_conf = Action.HOLD, 0.5

        # Ensemble voting logic
        decision = self._vote(
            xgb_action=xgb_action,
            xgb_conf=xgb_conf,
            rl_action=rl_action,
            rl_conf=rl_conf,
            portfolio_state=portfolio_state
        )

        decision.pair = pair

        # Log decision
        action_name = {Action.BUY: "BUY", Action.SELL: "SELL", Action.HOLD: "HOLD"}
        logger.info(
            f"[{pair}] Ensemble: {action_name.get(decision.action, 'HOLD')} "
            f"(conf: {decision.confidence:.2f}) - "
            f"XGB: {action_name.get(xgb_action, 'HOLD')} ({xgb_conf:.2f}), "
            f"RL: {action_name.get(rl_action, 'HOLD')} ({rl_conf:.2f})"
        )

        return decision

    def _vote(
        self,
        xgb_action: int,
        xgb_conf: float,
        rl_action: int,
        rl_conf: float,
        portfolio_state: Dict[str, Any]
    ) -> TradeDecision:
        """
        Voting logic to combine model predictions.

        Returns:
            TradeDecision
        """
        # Calculate weighted confidence
        weighted_conf = (
            xgb_conf * self.xgb_weight +
            rl_conf * self.rl_weight
        )

        # Case 1: Both agree on BUY
        if xgb_action == Action.BUY and rl_action == Action.BUY:
            if weighted_conf >= self.min_confidence:
                return TradeDecision(
                    action=Action.BUY,
                    confidence=weighted_conf,
                    xgb_action=xgb_action,
                    xgb_confidence=xgb_conf,
                    rl_action=rl_action,
                    rl_confidence=rl_conf,
                    reasoning="Both models agree: BUY signal with high confidence",
                    suggested_size_pct=self._calculate_position_size(weighted_conf)
                )
            else:
                return self._hold_decision(
                    xgb_action, xgb_conf, rl_action, rl_conf,
                    "Both agree BUY but confidence too low"
                )

        # Case 2: Both agree on SELL
        if xgb_action == Action.SELL and rl_action == Action.SELL:
            if weighted_conf >= self.min_confidence:
                return TradeDecision(
                    action=Action.SELL,
                    confidence=weighted_conf,
                    xgb_action=xgb_action,
                    xgb_confidence=xgb_conf,
                    rl_action=rl_action,
                    rl_confidence=rl_conf,
                    reasoning="Both models agree: SELL signal with high confidence",
                    suggested_size_pct=100.0  # Sell entire position
                )
            else:
                return self._hold_decision(
                    xgb_action, xgb_conf, rl_action, rl_conf,
                    "Both agree SELL but confidence too low"
                )

        # Case 3: One model is very confident (>80%) - allow single model override
        if xgb_action != Action.HOLD and xgb_conf > 0.80:
            return TradeDecision(
                action=xgb_action,
                confidence=xgb_conf * 0.8,  # Reduce confidence for single model
                xgb_action=xgb_action,
                xgb_confidence=xgb_conf,
                rl_action=rl_action,
                rl_confidence=rl_conf,
                reasoning=f"XGBoost high-confidence signal (RL disagrees)",
                suggested_size_pct=self._calculate_position_size(xgb_conf * 0.8)
            )

        if rl_action != Action.HOLD and rl_conf > 0.80:
            return TradeDecision(
                action=rl_action,
                confidence=rl_conf * 0.8,
                xgb_action=xgb_action,
                xgb_confidence=xgb_conf,
                rl_action=rl_action,
                rl_confidence=rl_conf,
                reasoning=f"RL high-confidence signal (XGBoost disagrees)",
                suggested_size_pct=self._calculate_position_size(rl_conf * 0.8)
            )

        # Case 4: Disagreement or low confidence -> HOLD
        reason = "Models disagree" if xgb_action != rl_action else "Both models suggest HOLD"
        return self._hold_decision(xgb_action, xgb_conf, rl_action, rl_conf, reason)

    def _hold_decision(
        self,
        xgb_action: int,
        xgb_conf: float,
        rl_action: int,
        rl_conf: float,
        reason: str
    ) -> TradeDecision:
        """Create a HOLD decision."""
        return TradeDecision(
            action=Action.HOLD,
            confidence=0.0,
            xgb_action=xgb_action,
            xgb_confidence=xgb_conf,
            rl_action=rl_action,
            rl_confidence=rl_conf,
            reasoning=reason,
            suggested_size_pct=0.0
        )

    def _calculate_position_size(self, confidence: float) -> float:
        """
        Calculate suggested position size based on confidence.

        Higher confidence = larger position (within limits)
        Range: 10% to 25% of portfolio
        """
        min_size = 10.0
        max_size = 25.0

        # Linear scaling: 70% conf -> 10%, 100% conf -> 25%
        confidence_range = 1.0 - self.min_confidence  # e.g., 0.3 if min is 0.7
        scaled = (confidence - self.min_confidence) / confidence_range
        scaled = max(0, min(1, scaled))  # Clamp to [0, 1]

        size = min_size + (max_size - min_size) * scaled
        return round(size, 1)

    def update_weights(self, xgb_correct: bool, rl_correct: bool):
        """
        Update model weights based on recent accuracy.

        Call this after each trade to track which model performs better.
        """
        # Track recent accuracy (rolling window of 20)
        self.xgb_recent_accuracy.append(1 if xgb_correct else 0)
        self.rl_recent_accuracy.append(1 if rl_correct else 0)

        if len(self.xgb_recent_accuracy) > 20:
            self.xgb_recent_accuracy.pop(0)
        if len(self.rl_recent_accuracy) > 20:
            self.rl_recent_accuracy.pop(0)

        # Update weights based on recent accuracy
        if len(self.xgb_recent_accuracy) >= 10:
            xgb_acc = sum(self.xgb_recent_accuracy) / len(self.xgb_recent_accuracy)
            rl_acc = sum(self.rl_recent_accuracy) / len(self.rl_recent_accuracy)

            total = xgb_acc + rl_acc
            if total > 0:
                self.xgb_weight = xgb_acc / total
                self.rl_weight = rl_acc / total

                logger.info(
                    f"Updated weights - XGB: {self.xgb_weight:.2f}, RL: {self.rl_weight:.2f}"
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            "xgb_weight": self.xgb_weight,
            "rl_weight": self.rl_weight,
            "min_confidence": self.min_confidence,
            "xgb_recent_accuracy": (
                sum(self.xgb_recent_accuracy) / len(self.xgb_recent_accuracy)
                if self.xgb_recent_accuracy else 0
            ),
            "rl_recent_accuracy": (
                sum(self.rl_recent_accuracy) / len(self.rl_recent_accuracy)
                if self.rl_recent_accuracy else 0
            ),
        }
