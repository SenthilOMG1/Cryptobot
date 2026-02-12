"""
Trend Filter
============
Blocks trades against the macro trend to prevent counter-trend losses.
Uses multi-timeframe features to determine the dominant market direction.
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Trend(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TrendFilter:
    """
    Determines the macro trend and blocks counter-trend trades.

    Scoring system combines:
    - ADX trend strength
    - Price vs SMA50 / SMA200 / EMA21
    - MACD histogram direction
    - 1D timeframe features for macro view (falls back to 1H)
    """

    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: Score magnitude required to declare bullish/bearish.
                       Score in [-1, 1]. Above threshold = bullish, below -threshold = bearish.
        """
        self.threshold = threshold

    def get_trend(self, features_row: pd.Series) -> Trend:
        """
        Determine the macro trend from a single row of features.

        Args:
            features_row: Series containing calculated features (1H + multi-TF)

        Returns:
            Trend enum: BULLISH, BEARISH, or NEUTRAL
        """
        score = self._calculate_trend_score(features_row)

        if score >= self.threshold:
            return Trend.BULLISH
        elif score <= -self.threshold:
            return Trend.BEARISH
        else:
            return Trend.NEUTRAL

    def _calculate_trend_score(self, row: pd.Series) -> float:
        """
        Calculate a trend score from -1 (strong bearish) to +1 (strong bullish).

        Components (each contributes roughly equal weight):
        1. ADX trend strength + direction
        2. Price vs SMA50
        3. Price vs SMA200
        4. Price vs EMA21
        5. MACD histogram
        """
        scores = []
        weights = []

        # 1. ADX trend direction (use 1D first, fallback to 1H)
        adx = self._get_feature(row, ["tf1d_adx", "adx"])
        adx_pos = self._get_feature(row, ["adx_pos"])
        adx_neg = self._get_feature(row, ["adx_neg"])

        if adx is not None and adx_pos is not None and adx_neg is not None:
            if adx > 20:  # Trend is meaningful
                # Direction from DI+ vs DI-
                direction = 1.0 if adx_pos > adx_neg else -1.0
                # Strength scales with ADX (20-50 range mapped to 0.3-1.0)
                strength = min(1.0, max(0.3, (adx - 20) / 30))
                scores.append(direction * strength)
                weights.append(2.0)  # ADX gets double weight
            else:
                scores.append(0.0)
                weights.append(1.0)

        # 2. Price vs SMA50 (use 1D first, fallback to 1H)
        pvs50 = self._get_feature(row, ["tf1d_price_vs_sma50", "price_vs_sma50"])
        if pvs50 is not None:
            # Clamp to [-0.10, 0.10] and normalize to [-1, 1]
            scores.append(np.clip(pvs50 / 0.05, -1.0, 1.0))
            weights.append(1.5)

        # 3. Price vs SMA200
        pvs200 = self._get_feature(row, ["price_vs_sma200"])
        if pvs200 is not None:
            scores.append(np.clip(pvs200 / 0.08, -1.0, 1.0))
            weights.append(1.5)

        # 4. Price vs EMA21 (use 1H for shorter-term context)
        pve21 = self._get_feature(row, ["price_vs_ema21"])
        if pve21 is not None:
            scores.append(np.clip(pve21 / 0.03, -1.0, 1.0))
            weights.append(1.0)

        # 5. MACD histogram (use 1D first, fallback to 1H)
        macd_hist = self._get_feature(row, ["tf1d_macd_hist", "macd_hist"])
        if macd_hist is not None:
            # Normalize by recent close price (rough scaling)
            close = self._get_feature(row, ["close"])
            if close and close > 0:
                normalized = macd_hist / (close * 0.001)
                scores.append(np.clip(normalized, -1.0, 1.0))
            else:
                scores.append(np.clip(macd_hist * 100, -1.0, 1.0))
            weights.append(1.0)

        if not scores:
            logger.warning("No trend features available - returning NEUTRAL")
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return float(np.clip(weighted_score, -1.0, 1.0))

    def filter_decision(self, trend: Trend, action: int) -> bool:
        """
        Check if a trade should be allowed based on trend.

        Args:
            trend: Current macro trend
            action: 1 = BUY (long), -1 = SELL (short), 0 = HOLD

        Returns:
            True if trade is allowed, False if blocked
        """
        if action == 0:
            return True  # HOLD is always allowed

        if trend == Trend.BEARISH and action == 1:
            return False  # Block longs in bearish trend

        if trend == Trend.BULLISH and action == -1:
            return False  # Block shorts in bullish trend

        return True

    @staticmethod
    def _get_feature(row: pd.Series, names: list) -> Optional[float]:
        """Try to get a feature value from a list of possible names."""
        for name in names:
            if name in row.index:
                val = row[name]
                if pd.notna(val) and np.isfinite(val):
                    return float(val)
        return None
