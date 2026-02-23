"""
AdaptiveTriggerEngine — Decides WHEN to retrain models.

Two trigger mechanisms:
1. Performance Degradation: Models are performing poorly on recent predictions
2. Regime Shift: Market conditions have fundamentally changed

Severity levels:
- LOW: Just update ensemble weights (instant, no training needed)
- MEDIUM: Fine-tune existing models with recent data (30-60 min)
- HIGH: Full retrain from scratch (2-4 hours)

Cooldown prevents thrashing: no retrain within 48h of last retrain
(unless emergency performance degradation).
"""

import json
import logging
import math
import os
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AdaptiveTriggerEngine:

    def __init__(
        self,
        perf_threshold_medium: float = 0.40,   # Below this → medium retrain
        perf_threshold_high: float = 0.30,      # Below this → full retrain
        regime_z_threshold: float = 2.5,        # Z-score for regime shift
        cooldown_hours: int = 48,               # Hours between retrains
        emergency_cooldown_hours: int = 12,     # Emergency override cooldown
        state_path: str = "models/trigger_state.json",
    ):
        self.perf_threshold_medium = perf_threshold_medium
        self.perf_threshold_high = perf_threshold_high
        self.regime_z_threshold = regime_z_threshold
        self.cooldown_hours = cooldown_hours
        self.emergency_cooldown_hours = emergency_cooldown_hours
        self.state_path = state_path

        # Internal state
        self.last_retrain_time: Optional[datetime] = None
        self.last_check_result: Optional[Dict] = None
        self.trigger_history: List[Dict] = []

        # Rolling market statistics for regime detection
        self._vol_history = deque(maxlen=720)    # ~30 days of hourly data
        self._volume_history = deque(maxlen=720)
        self._trend_history = deque(maxlen=720)
        self._perf_history = deque(maxlen=100)

        # Performance metrics cache
        self._current_metrics: Optional[Dict] = None

        self._load_state()

    def update_metrics(
        self,
        performance_metrics: Dict,
        market_data: Optional[Dict] = None,
    ):
        """
        Called every cycle. Updates internal metrics.

        Args:
            performance_metrics: From PredictionTracker.get_performance_metrics()
            market_data: Optional dict with current volatility, volume, trend
                         e.g. {"volatility": 0.02, "volume_ratio": 1.5, "adx": 28}
        """
        self._current_metrics = performance_metrics

        if performance_metrics.get("data_ready"):
            self._perf_history.append({
                "timestamp": datetime.now().isoformat(),
                "ensemble_accuracy": performance_metrics["ensemble_accuracy"],
                "rolling_pnl": performance_metrics["rolling_pnl"],
            })

        if market_data:
            if "volatility" in market_data:
                self._vol_history.append(market_data["volatility"])
            if "volume_ratio" in market_data:
                self._volume_history.append(market_data["volume_ratio"])
            if "adx" in market_data:
                self._trend_history.append(market_data["adx"])

    def should_retrain(self) -> bool:
        """
        Check if any retrain trigger is active.
        Returns True if retraining should be initiated.
        """
        if self._current_metrics is None:
            return False

        if not self._current_metrics.get("data_ready"):
            return False

        # Check triggers
        perf_trigger = self._check_performance_degradation()
        regime_trigger = self._check_regime_shift()

        if perf_trigger is None and regime_trigger is None:
            self.last_check_result = {"triggered": False, "reason": "all_clear"}
            return False

        # Determine severity
        trigger = perf_trigger or regime_trigger
        severity = trigger.get("severity", "medium")

        # Check cooldown
        if self._is_in_cooldown(severity):
            self.last_check_result = {
                "triggered": False,
                "reason": "cooldown",
                "potential_trigger": trigger,
            }
            logger.info(
                f"Retrain trigger suppressed by cooldown: {trigger['reason']}"
            )
            return False

        self.last_check_result = {
            "triggered": True,
            **trigger,
        }
        return True

    def get_trigger_info(self) -> Dict:
        """
        Get details about what triggered retraining.
        Call this after should_retrain() returns True.
        """
        if self.last_check_result and self.last_check_result.get("triggered"):
            return self.last_check_result
        return {"triggered": False, "reason": "no_active_trigger"}

    def record_retrain_completed(self, severity: str = "medium"):
        """Called after retrain finishes to reset cooldown timer."""
        now = datetime.now()
        self.last_retrain_time = now
        self.trigger_history.append({
            "timestamp": now.isoformat(),
            "trigger": self.last_check_result,
            "severity": severity,
        })
        # Keep last 20 trigger events
        if len(self.trigger_history) > 20:
            self.trigger_history = self.trigger_history[-20:]
        self._save_state()
        logger.info(f"Retrain completed (severity={severity}), cooldown started")

    def _check_performance_degradation(self) -> Optional[Dict]:
        """
        Check if model performance has dropped below threshold.

        Performance score combines:
        - Ensemble accuracy (50% weight)
        - Average model accuracy (30% weight)
        - PnL factor (20% weight)
        """
        m = self._current_metrics
        if m["total_predictions"] < 15:
            return None

        # Calculate composite performance score
        ens_acc = m["ensemble_accuracy"]
        avg_model_acc = (m["xgb_accuracy"] + m["rl_accuracy"] + m["lstm_accuracy"]) / 3

        # PnL factor: sigmoid of normalized rolling PnL
        trade_count = max(m["trade_count"], 1)
        pnl_per_trade = m["rolling_pnl"] / trade_count if trade_count > 0 else 0
        pnl_factor = 1 / (1 + math.exp(-pnl_per_trade * 500))  # Sigmoid

        perf_score = (
            0.50 * ens_acc
            + 0.30 * avg_model_acc
            + 0.20 * pnl_factor
        )

        # Check confidence miscalibration
        # High confidence + low accuracy = confidently wrong = very bad
        conf_miscalibration = m["avg_confidence"] - ens_acc
        if conf_miscalibration > 0.25:
            # Models are confidently wrong
            perf_score *= 0.8  # Penalize further

        logger.debug(
            f"Performance score: {perf_score:.3f} "
            f"(ens={ens_acc:.3f}, models={avg_model_acc:.3f}, pnl={pnl_factor:.3f})"
        )

        if perf_score < self.perf_threshold_high:
            return {
                "reason": "performance_degradation",
                "severity": "high",
                "perf_score": round(perf_score, 3),
                "ensemble_accuracy": ens_acc,
                "avg_model_accuracy": round(avg_model_acc, 3),
                "pnl_factor": round(pnl_factor, 3),
                "conf_miscalibration": round(conf_miscalibration, 3),
                "recommended_action": "full_retrain",
            }
        elif perf_score < self.perf_threshold_medium:
            return {
                "reason": "performance_degradation",
                "severity": "medium",
                "perf_score": round(perf_score, 3),
                "ensemble_accuracy": ens_acc,
                "avg_model_accuracy": round(avg_model_acc, 3),
                "pnl_factor": round(pnl_factor, 3),
                "recommended_action": "fine_tune",
            }

        return None

    def _check_regime_shift(self) -> Optional[Dict]:
        """
        Detect significant market regime changes using z-scores.

        Monitors volatility, volume, and trend strength for sudden shifts.
        """
        shifts_detected = []

        # Volatility regime shift
        if len(self._vol_history) >= 30:
            z = self._calc_z_score(self._vol_history)
            if abs(z) > self.regime_z_threshold:
                shifts_detected.append(
                    f"volatility_z={z:.2f}"
                )

        # Volume regime shift
        if len(self._volume_history) >= 30:
            z = self._calc_z_score(self._volume_history)
            if abs(z) > 3.0:  # Volume needs higher threshold (spikier)
                shifts_detected.append(
                    f"volume_z={z:.2f}"
                )

        # Trend regime shift (ADX crossing thresholds)
        if len(self._trend_history) >= 30:
            recent_adx = list(self._trend_history)[-1]
            avg_adx = sum(self._trend_history) / len(self._trend_history)
            # Detect transition between trending (>25) and ranging (<20)
            if (recent_adx > 35 and avg_adx < 22) or (recent_adx < 15 and avg_adx > 28):
                shifts_detected.append(
                    f"trend_shift=adx_{recent_adx:.0f}_vs_avg_{avg_adx:.0f}"
                )

        if not shifts_detected:
            return None

        return {
            "reason": "regime_shift",
            "severity": "medium",
            "shifts": shifts_detected,
            "recommended_action": "fine_tune",
        }

    def _calc_z_score(self, data: deque) -> float:
        """Calculate z-score of the most recent value vs rolling statistics."""
        values = list(data)
        if len(values) < 10:
            return 0.0

        recent = values[-1]
        # Use all but last 5 as the "baseline"
        baseline = values[:-5]
        mean = sum(baseline) / len(baseline)
        variance = sum((x - mean) ** 2 for x in baseline) / len(baseline)
        std = math.sqrt(variance) if variance > 0 else 1e-8

        return (recent - mean) / std

    def _is_in_cooldown(self, severity: str = "medium") -> bool:
        """Check if we're still in post-retrain cooldown."""
        if self.last_retrain_time is None:
            return False

        elapsed = datetime.now() - self.last_retrain_time

        # Emergency (high severity) has shorter cooldown
        if severity == "high":
            return elapsed < timedelta(hours=self.emergency_cooldown_hours)

        return elapsed < timedelta(hours=self.cooldown_hours)

    def get_status(self) -> Dict:
        """Return trigger engine status for monitoring."""
        cooldown_remaining = None
        if self.last_retrain_time:
            elapsed = datetime.now() - self.last_retrain_time
            remaining = timedelta(hours=self.cooldown_hours) - elapsed
            if remaining.total_seconds() > 0:
                cooldown_remaining = str(remaining).split(".")[0]

        return {
            "last_retrain": (
                self.last_retrain_time.isoformat()
                if self.last_retrain_time else "never"
            ),
            "cooldown_remaining": cooldown_remaining,
            "last_check": self.last_check_result,
            "current_metrics": self._current_metrics,
            "vol_samples": len(self._vol_history),
            "perf_samples": len(self._perf_history),
            "trigger_count": len(self.trigger_history),
        }

    def _save_state(self):
        """Persist trigger engine state."""
        state = {
            "last_retrain_time": (
                self.last_retrain_time.isoformat()
                if self.last_retrain_time else None
            ),
            "trigger_history": self.trigger_history[-10:],
            "saved_at": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Restore trigger engine state."""
        if not os.path.exists(self.state_path):
            return

        try:
            with open(self.state_path, "r") as f:
                state = json.load(f)
            if state.get("last_retrain_time"):
                self.last_retrain_time = datetime.fromisoformat(
                    state["last_retrain_time"]
                )
            self.trigger_history = state.get("trigger_history", [])
            logger.info(
                f"Loaded trigger state: last_retrain={self.last_retrain_time}"
            )
        except Exception as e:
            logger.warning(f"Failed to load trigger state: {e}")
