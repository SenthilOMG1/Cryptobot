"""
Ensemble Decision Engine
========================
Combines XGBoost, RL agent, and LSTM predictions through voting.

V2: MetaLearnerEnsemble with 3-model voting + LogisticRegression meta-learner.
Original EnsembleDecider kept as fallback.
"""

import os
import logging
import pickle
from typing import Tuple, Optional, Dict, Any, List
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
    """Trading decision from ensemble."""
    action: int
    confidence: float
    xgb_action: int
    xgb_confidence: float
    rl_action: int
    rl_confidence: float
    reasoning: str
    pair: str = ""
    suggested_size_pct: float = 0.0
    lstm_action: int = 0
    lstm_confidence: float = 0.0
    regime: str = ""


class EnsembleDecider:
    """
    Original 2-model ensemble voting system (kept as fallback).

    Rules:
    1. Both models must agree on action (BUY/SELL)
    2. Combined confidence must exceed threshold
    3. If models disagree -> HOLD
    4. Higher confidence = larger suggested position size
    """

    def __init__(
        self,
        xgb_model,
        rl_agent,
        min_confidence: float = 0.70,
        agreement_required: bool = True
    ):
        self.xgb_model = xgb_model
        self.rl_agent = rl_agent
        self.min_confidence = min_confidence
        self.agreement_required = agreement_required

        self.xgb_weight = 0.5
        self.rl_weight = 0.5

        self.xgb_recent_accuracy = []
        self.rl_recent_accuracy = []

    def get_decision(
        self,
        features: pd.DataFrame,
        portfolio_state: Dict[str, Any],
        pair: str = ""
    ) -> TradeDecision:
        """Get ensemble trading decision."""
        try:
            xgb_action, xgb_conf = self.xgb_model.predict(features)
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            xgb_action, xgb_conf = Action.HOLD, 0.5

        try:
            feature_cols = self.xgb_model.feature_names
            feature_array = features[feature_cols].values.flatten().astype(np.float32)
            rl_action, rl_conf = self.rl_agent.decide(feature_array, portfolio_state)
        except Exception as e:
            logger.error(f"RL prediction failed: {e}")
            rl_action, rl_conf = Action.HOLD, 0.5

        decision = self._vote(
            xgb_action=xgb_action,
            xgb_conf=xgb_conf,
            rl_action=rl_action,
            rl_conf=rl_conf,
            portfolio_state=portfolio_state
        )

        decision.pair = pair

        action_name = {Action.BUY: "BUY", Action.SELL: "SELL", Action.HOLD: "HOLD"}
        logger.info(
            f"[{pair}] Ensemble: {action_name.get(decision.action, 'HOLD')} "
            f"(conf: {decision.confidence:.2f}) - "
            f"XGB: {action_name.get(xgb_action, 'HOLD')} ({xgb_conf:.2f}), "
            f"RL: {action_name.get(rl_action, 'HOLD')} ({rl_conf:.2f})"
        )

        return decision

    def _vote(self, xgb_action, xgb_conf, rl_action, rl_conf, portfolio_state):
        weighted_conf = xgb_conf * self.xgb_weight + rl_conf * self.rl_weight

        if xgb_action == Action.BUY and rl_action == Action.BUY:
            if weighted_conf >= self.min_confidence:
                return TradeDecision(
                    action=Action.BUY, confidence=weighted_conf,
                    xgb_action=xgb_action, xgb_confidence=xgb_conf,
                    rl_action=rl_action, rl_confidence=rl_conf,
                    reasoning="Both models agree: BUY signal with high confidence",
                    suggested_size_pct=self._calculate_position_size(weighted_conf)
                )
            return self._hold_decision(xgb_action, xgb_conf, rl_action, rl_conf, "Both agree BUY but confidence too low")

        if xgb_action == Action.SELL and rl_action == Action.SELL:
            if weighted_conf >= self.min_confidence:
                return TradeDecision(
                    action=Action.SELL, confidence=weighted_conf,
                    xgb_action=xgb_action, xgb_confidence=xgb_conf,
                    rl_action=rl_action, rl_confidence=rl_conf,
                    reasoning="Both models agree: SELL signal with high confidence",
                    suggested_size_pct=self._calculate_position_size(weighted_conf)
                )
            return self._hold_decision(xgb_action, xgb_conf, rl_action, rl_conf, "Both agree SELL but confidence too low")

        if xgb_action != Action.HOLD and xgb_conf > 0.80:
            return TradeDecision(
                action=xgb_action, confidence=xgb_conf * 0.8,
                xgb_action=xgb_action, xgb_confidence=xgb_conf,
                rl_action=rl_action, rl_confidence=rl_conf,
                reasoning="XGBoost high-confidence signal (RL disagrees)",
                suggested_size_pct=self._calculate_position_size(xgb_conf * 0.8)
            )

        if rl_action != Action.HOLD and rl_conf > 0.80:
            return TradeDecision(
                action=rl_action, confidence=rl_conf * 0.8,
                xgb_action=xgb_action, xgb_confidence=xgb_conf,
                rl_action=rl_action, rl_confidence=rl_conf,
                reasoning="RL high-confidence signal (XGBoost disagrees)",
                suggested_size_pct=self._calculate_position_size(rl_conf * 0.8)
            )

        reason = "Models disagree" if xgb_action != rl_action else "Both models suggest HOLD"
        return self._hold_decision(xgb_action, xgb_conf, rl_action, rl_conf, reason)

    def _hold_decision(self, xgb_action, xgb_conf, rl_action, rl_conf, reason):
        return TradeDecision(
            action=Action.HOLD, confidence=0.0,
            xgb_action=xgb_action, xgb_confidence=xgb_conf,
            rl_action=rl_action, rl_confidence=rl_conf,
            reasoning=reason, suggested_size_pct=0.0
        )

    def _calculate_position_size(self, confidence: float) -> float:
        if confidence >= 0.75:
            return 20.0
        elif confidence >= 0.65:
            return 15.0
        elif confidence >= 0.55:
            return 10.0
        else:
            return 5.0

    def update_weights(self, xgb_correct: bool, rl_correct: bool):
        self.xgb_recent_accuracy.append(1 if xgb_correct else 0)
        self.rl_recent_accuracy.append(1 if rl_correct else 0)
        if len(self.xgb_recent_accuracy) > 20:
            self.xgb_recent_accuracy.pop(0)
        if len(self.rl_recent_accuracy) > 20:
            self.rl_recent_accuracy.pop(0)
        if len(self.xgb_recent_accuracy) >= 10:
            xgb_acc = sum(self.xgb_recent_accuracy) / len(self.xgb_recent_accuracy)
            rl_acc = sum(self.rl_recent_accuracy) / len(self.rl_recent_accuracy)
            total = xgb_acc + rl_acc
            if total > 0:
                self.xgb_weight = xgb_acc / total
                self.rl_weight = rl_acc / total

    def get_stats(self):
        return {
            "xgb_weight": self.xgb_weight,
            "rl_weight": self.rl_weight,
            "min_confidence": self.min_confidence,
        }


class MetaLearnerEnsemble:
    """
    3-model ensemble with optional meta-learner.

    Models: XGBoost (0.35), RL (0.30), LSTM (0.35)
    Once 50+ outcomes are recorded, trains a LogisticRegression meta-learner
    that learns which model combinations produce winning trades.
    """

    META_LEARNER_PATH = "models/meta_learner.pkl"

    def __init__(
        self,
        xgb_model,
        rl_agent,
        lstm_model=None,
        min_confidence: float = 0.50
    ):
        self.xgb_model = xgb_model
        self.rl_agent = rl_agent
        self.lstm_model = lstm_model
        self.min_confidence = min_confidence

        # Equal model weights — all 3 retrained with balanced data
        self.xgb_weight = 0.35
        self.rl_weight = 0.30
        self.lstm_weight = 0.35

        # Meta-learner
        self.meta_model = None
        self.outcome_buffer = []  # List of (meta_features, outcome)
        self._load_meta_learner()

    def get_decision(
        self,
        features: pd.DataFrame,
        portfolio_state: Dict[str, Any],
        pair: str = "",
        regime: str = "",
        volatility: float = 0.0,
        trend_strength: float = 0.0
    ) -> TradeDecision:
        """
        Get 3-model ensemble trading decision.

        Args:
            features: DataFrame with calculated features (single row)
            portfolio_state: Dict with balance, position, entry_price, current_price
            pair: Trading pair name
            regime: Market regime string (for meta-features)
            volatility: Current volatility level (for meta-features)
            trend_strength: Current trend score (for meta-features)
        """
        # Get XGBoost prediction
        try:
            xgb_action, xgb_conf = self.xgb_model.predict(features)
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            xgb_action, xgb_conf = Action.HOLD, 0.5

        # Get RL prediction
        try:
            feature_cols = self.xgb_model.feature_names
            feature_array = features[feature_cols].values.flatten().astype(np.float32)
            rl_action, rl_conf = self.rl_agent.decide(feature_array, portfolio_state)
        except Exception as e:
            logger.error(f"RL prediction failed: {e}")
            rl_action, rl_conf = Action.HOLD, 0.5

        # Get LSTM prediction
        lstm_action, lstm_conf = Action.HOLD, 0.5
        if self.lstm_model is not None and self.lstm_model.is_trained:
            try:
                lstm_action, lstm_conf = self.lstm_model.predict(features)
            except Exception as e:
                logger.error(f"LSTM prediction failed: {e}")

        # Build meta-features
        regime_num = {"trending_up": 1, "trending_down": -1, "ranging": 0, "high_volatility": 2}.get(regime, 0)
        meta_features = [
            xgb_action, xgb_conf,
            rl_action, rl_conf,
            lstm_action, lstm_conf,
            regime_num, volatility, trend_strength
        ]

        # Try meta-learner first
        if self.meta_model is not None:
            try:
                decision = self._meta_learner_decision(
                    meta_features, xgb_action, xgb_conf, rl_action, rl_conf,
                    lstm_action, lstm_conf, pair, regime
                )
                if decision is not None:
                    return decision
            except Exception as e:
                logger.debug(f"Meta-learner failed, falling back to voting: {e}")

        # Fallback: weighted 3-model voting
        decision = self._weighted_vote(
            xgb_action, xgb_conf,
            rl_action, rl_conf,
            lstm_action, lstm_conf,
            pair, regime
        )

        # Log
        action_name = {Action.BUY: "BUY", Action.SELL: "SELL", Action.HOLD: "HOLD"}
        logger.info(
            f"[{pair}] MetaEnsemble: {action_name.get(decision.action, 'HOLD')} "
            f"(conf: {decision.confidence:.2f}) - "
            f"XGB: {action_name.get(xgb_action, 'HOLD')} ({xgb_conf:.2f}), "
            f"RL: {action_name.get(rl_action, 'HOLD')} ({rl_conf:.2f}), "
            f"LSTM: {action_name.get(lstm_action, 'HOLD')} ({lstm_conf:.2f})"
        )

        return decision

    def _weighted_vote(
        self,
        xgb_action, xgb_conf,
        rl_action, rl_conf,
        lstm_action, lstm_conf,
        pair, regime
    ) -> TradeDecision:
        """Enhanced 3-model weighted voting with directional balance."""
        models = [
            (xgb_action, xgb_conf, self.xgb_weight),
            (rl_action, rl_conf, self.rl_weight),
            (lstm_action, lstm_conf, self.lstm_weight)
        ]

        # Calculate weighted scores for each action
        buy_score = sum(w * c for a, c, w in models if a == Action.BUY)
        sell_score = sum(w * c for a, c, w in models if a == Action.SELL)
        hold_score = sum(w * c for a, c, w in models if a == Action.HOLD)

        # Count agreements
        buy_votes = sum(1 for a, c, w in models if a == Action.BUY)
        sell_votes = sum(1 for a, c, w in models if a == Action.SELL)

        # Decision logic
        # Symmetric rules: BUY and SELL use identical thresholds
        # 2/3 agreement = strong signal
        # XGB+RL agreement without LSTM = still valid (LSTM has known SELL bias)
        for action_type, action_score, action_votes, label in [
            (Action.BUY, buy_score, buy_votes, "BUY"),
            (Action.SELL, sell_score, sell_votes, "SELL"),
        ]:
            opposite_score = sell_score if action_type == Action.BUY else buy_score
            if action_score > opposite_score and action_score > hold_score and action_votes >= 2:
                agreeing_weight = sum(w for a, c, w in models if a == action_type)
                confidence = action_score / agreeing_weight if agreeing_weight > 0 else 0
                return TradeDecision(
                    action=action_type, confidence=confidence,
                    xgb_action=xgb_action, xgb_confidence=xgb_conf,
                    rl_action=rl_action, rl_confidence=rl_conf,
                    lstm_action=lstm_action, lstm_confidence=lstm_conf,
                    reasoning=f"{action_votes}/3 models agree: {label}",
                    pair=pair, regime=regime,
                    suggested_size_pct=self._calculate_position_size(confidence)
                )

        # XGB + RL agree but LSTM dissents (common for BUY since LSTM has SELL bias)
        # Allow if both have decent confidence (>0.55)
        if xgb_action == rl_action and xgb_action != Action.HOLD:
            if xgb_conf > 0.55 and rl_conf > 0.55:
                action_type = xgb_action
                label = "BUY" if action_type == Action.BUY else "SELL"
                # Use XGB+RL weighted confidence with slight penalty for missing LSTM
                agreeing_weight = self.xgb_weight + self.rl_weight
                raw_conf = (self.xgb_weight * xgb_conf + self.rl_weight * rl_conf) / agreeing_weight
                confidence = raw_conf * 0.90  # 10% penalty for 2-model only
                return TradeDecision(
                    action=action_type, confidence=confidence,
                    xgb_action=xgb_action, xgb_confidence=xgb_conf,
                    rl_action=rl_action, rl_confidence=rl_conf,
                    lstm_action=lstm_action, lstm_confidence=lstm_conf,
                    reasoning=f"XGB+RL agree: {label} (LSTM dissent, conf penalty)",
                    pair=pair, regime=regime,
                    suggested_size_pct=self._calculate_position_size(confidence)
                )

        # Single model high-confidence override (>80%)
        # Exclude LSTM from solo override - it has known directional bias
        for name, action, conf in [("XGB", xgb_action, xgb_conf), ("RL", rl_action, rl_conf)]:
            if action != Action.HOLD and conf > 0.80:
                adj_conf = conf * 0.75  # Penalty for single model
                return TradeDecision(
                    action=action, confidence=adj_conf,
                    xgb_action=xgb_action, xgb_confidence=xgb_conf,
                    rl_action=rl_action, rl_confidence=rl_conf,
                    lstm_action=lstm_action, lstm_confidence=lstm_conf,
                    reasoning=f"{name} high-confidence override ({conf:.2f})",
                    pair=pair, regime=regime,
                    suggested_size_pct=self._calculate_position_size(adj_conf)
                )

        # HOLD
        return TradeDecision(
            action=Action.HOLD, confidence=0.0,
            xgb_action=xgb_action, xgb_confidence=xgb_conf,
            rl_action=rl_action, rl_confidence=rl_conf,
            lstm_action=lstm_action, lstm_confidence=lstm_conf,
            reasoning="No clear signal from 3-model vote",
            pair=pair, regime=regime, suggested_size_pct=0.0
        )

    def _meta_learner_decision(self, meta_features, xgb_action, xgb_conf, rl_action, rl_conf, lstm_action, lstm_conf, pair, regime):
        """Use meta-learner to make decision."""
        features_arr = np.array(meta_features).reshape(1, -1)
        pred = self.meta_model.predict(features_arr)[0]
        proba = self.meta_model.predict_proba(features_arr)[0]
        confidence = float(np.max(proba))

        # Map: 0=SELL, 1=HOLD, 2=BUY
        label_map = {0: Action.SELL, 1: Action.HOLD, 2: Action.BUY}
        action = label_map.get(pred, Action.HOLD)

        if action == Action.HOLD:
            return None  # Let voting handle HOLD

        return TradeDecision(
            action=action, confidence=confidence,
            xgb_action=xgb_action, xgb_confidence=xgb_conf,
            rl_action=rl_action, rl_confidence=rl_conf,
            lstm_action=lstm_action, lstm_confidence=lstm_conf,
            reasoning=f"Meta-learner decision (conf: {confidence:.2f})",
            pair=pair, regime=regime,
            suggested_size_pct=self._calculate_position_size(confidence)
        )

    def record_outcome(self, meta_features: list, outcome: int):
        """
        Record a trade outcome for meta-learner training.

        Args:
            meta_features: [xgb_action, xgb_conf, rl_action, rl_conf, lstm_action, lstm_conf, regime, vol, trend]
            outcome: 2=BUY was right, 0=SELL was right, 1=HOLD was right
        """
        self.outcome_buffer.append((meta_features, outcome))

        # Auto-train meta-learner when we have enough data
        if len(self.outcome_buffer) >= 50 and len(self.outcome_buffer) % 10 == 0:
            self._train_meta_learner()

    def _train_meta_learner(self):
        """Train LogisticRegression meta-learner on outcome buffer."""
        try:
            from sklearn.linear_model import LogisticRegression

            X = np.array([f for f, _ in self.outcome_buffer])
            y = np.array([o for _, o in self.outcome_buffer])

            self.meta_model = LogisticRegression(max_iter=500, multi_class="multinomial")
            self.meta_model.fit(X, y)

            self._save_meta_learner()
            logger.info(f"Meta-learner trained on {len(self.outcome_buffer)} outcomes")
        except Exception as e:
            logger.warning(f"Meta-learner training failed: {e}")

    def _save_meta_learner(self):
        """Save meta-learner and outcome buffer."""
        os.makedirs(os.path.dirname(self.META_LEARNER_PATH), exist_ok=True)
        with open(self.META_LEARNER_PATH, "wb") as f:
            pickle.dump({
                "meta_model": self.meta_model,
                "outcome_buffer": self.outcome_buffer
            }, f)

    def _load_meta_learner(self):
        """Load meta-learner and outcome buffer."""
        if os.path.exists(self.META_LEARNER_PATH):
            try:
                with open(self.META_LEARNER_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.meta_model = data.get("meta_model")
                    self.outcome_buffer = data.get("outcome_buffer", [])
                    logger.info(f"Meta-learner loaded ({len(self.outcome_buffer)} outcomes)")
            except Exception as e:
                logger.warning(f"Failed to load meta-learner: {e}")

    @staticmethod
    def _calculate_position_size(confidence: float) -> float:
        if confidence >= 0.75:
            return 20.0
        elif confidence >= 0.65:
            return 15.0
        elif confidence >= 0.55:
            return 10.0
        else:
            return 5.0

    def get_stats(self):
        return {
            "xgb_weight": self.xgb_weight,
            "rl_weight": self.rl_weight,
            "lstm_weight": self.lstm_weight,
            "min_confidence": self.min_confidence,
            "meta_learner_trained": self.meta_model is not None,
            "outcome_buffer_size": len(self.outcome_buffer),
        }
