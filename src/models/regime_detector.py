"""
Market Regime Detector
======================
Detects the current market regime (trending, ranging, high volatility)
and provides per-regime parameter adjustments for risk management.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class RegimeParameters:
    """Per-regime adjustments applied to trading parameters."""
    regime: MarketRegime
    trailing_stop_mult: float   # Multiplier for trailing stop distance
    position_size_mult: float   # Multiplier for position size
    confidence_offset: float    # Added to min_confidence threshold (positive = harder entry)
    description: str

    @property
    def name(self) -> str:
        return self.regime.value


# Pre-defined regime parameter sets
REGIME_PARAMS = {
    MarketRegime.TRENDING_UP: RegimeParameters(
        regime=MarketRegime.TRENDING_UP,
        trailing_stop_mult=1.3,    # Wider stops - let trends run
        position_size_mult=1.2,    # Slightly larger positions
        confidence_offset=-0.03,   # Easier entry - trends are predictable
        description="Strong uptrend - wider stops, larger positions, easier entry"
    ),
    MarketRegime.TRENDING_DOWN: RegimeParameters(
        regime=MarketRegime.TRENDING_DOWN,
        trailing_stop_mult=1.3,    # Wider stops for shorts too
        position_size_mult=1.2,
        confidence_offset=-0.03,
        description="Strong downtrend - wider stops, larger positions, easier entry"
    ),
    MarketRegime.RANGING: RegimeParameters(
        regime=MarketRegime.RANGING,
        trailing_stop_mult=0.8,    # Tighter stops - no momentum
        position_size_mult=0.8,    # Smaller positions
        confidence_offset=0.02,    # Slightly harder entry - but don't block everything
        description="Range-bound - tighter stops, smaller positions, harder entry"
    ),
    MarketRegime.HIGH_VOLATILITY: RegimeParameters(
        regime=MarketRegime.HIGH_VOLATILITY,
        trailing_stop_mult=1.5,    # Much wider stops - volatile swings
        position_size_mult=0.5,    # Half positions - high risk
        confidence_offset=0.10,    # Much harder entry - wild markets
        description="High volatility - wide stops, half positions, very hard entry"
    ),
}


class MarketRegimeDetector:
    """
    Detects market regime from technical features.

    Uses:
    - ADX (>25 = trending) for trend detection
    - ADX_pos vs ADX_neg for trend direction
    - ATR percentile for volatility detection
    """

    def __init__(self, atr_lookback: int = 50, high_vol_percentile: float = 80.0):
        """
        Args:
            atr_lookback: Number of periods to calculate ATR percentile
            high_vol_percentile: ATR percentile above which is HIGH_VOLATILITY
        """
        self.atr_lookback = atr_lookback
        self.high_vol_percentile = high_vol_percentile
        self._atr_history = []  # Rolling ATR values for percentile calculation

    def detect_regime(self, features_row: pd.Series, features_df: Optional[pd.DataFrame] = None) -> MarketRegime:
        """
        Detect the current market regime.

        Args:
            features_row: Single row of features (latest candle)
            features_df: Full features DataFrame for ATR history (optional)

        Returns:
            MarketRegime enum
        """
        # Extract key indicators
        adx = self._get_val(features_row, ["adx"])
        adx_pos = self._get_val(features_row, ["adx_pos"])
        adx_neg = self._get_val(features_row, ["adx_neg"])
        atr_14 = self._get_val(features_row, ["atr_14"])
        close = self._get_val(features_row, ["close"])

        # Update ATR history for percentile calculation
        if atr_14 is not None and close is not None and close > 0:
            atr_pct = atr_14 / close * 100  # ATR as % of price
            self._atr_history.append(atr_pct)
            if len(self._atr_history) > 200:
                self._atr_history = self._atr_history[-200:]

        # If we have a full DataFrame, compute ATR percentile from it
        if features_df is not None and "atr_14" in features_df.columns and "close" in features_df.columns:
            recent = features_df.tail(self.atr_lookback)
            atr_series = recent["atr_14"] / recent["close"] * 100
            atr_series = atr_series.dropna()
            if len(atr_series) >= 10 and atr_pct is not None:
                atr_percentile = (atr_series < atr_pct).mean() * 100
            else:
                atr_percentile = 50.0
        elif len(self._atr_history) >= 10 and atr_14 is not None:
            atr_pct_val = self._atr_history[-1]
            atr_percentile = sum(1 for v in self._atr_history if v < atr_pct_val) / len(self._atr_history) * 100
        else:
            atr_percentile = 50.0

        # Step 1: Check for high volatility first (overrides trend)
        if atr_percentile >= self.high_vol_percentile:
            logger.debug(f"HIGH_VOLATILITY regime (ATR percentile: {atr_percentile:.0f}%)")
            return MarketRegime.HIGH_VOLATILITY

        # Step 2: Check for trending regime
        if adx is not None and adx > 25:
            if adx_pos is not None and adx_neg is not None:
                if adx_pos > adx_neg:
                    logger.debug(f"TRENDING_UP regime (ADX: {adx:.1f}, DI+: {adx_pos:.1f} > DI-: {adx_neg:.1f})")
                    return MarketRegime.TRENDING_UP
                else:
                    logger.debug(f"TRENDING_DOWN regime (ADX: {adx:.1f}, DI-: {adx_neg:.1f} > DI+: {adx_pos:.1f})")
                    return MarketRegime.TRENDING_DOWN
            # ADX high but no direction info - still trending
            price_vs_sma50 = self._get_val(features_row, ["price_vs_sma50"])
            if price_vs_sma50 is not None:
                if price_vs_sma50 > 0:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            return MarketRegime.TRENDING_UP  # Default to up if no other info

        # Step 3: Low ADX = ranging
        logger.debug(f"RANGING regime (ADX: {adx if adx else 'N/A'})")
        return MarketRegime.RANGING

    def get_regime_params(self, regime: MarketRegime) -> RegimeParameters:
        """Get the parameter adjustments for a given regime."""
        return REGIME_PARAMS[regime]

    def detect_and_get_params(self, features_row: pd.Series, features_df: Optional[pd.DataFrame] = None) -> RegimeParameters:
        """Convenience: detect regime and return its parameters in one call."""
        regime = self.detect_regime(features_row, features_df)
        return self.get_regime_params(regime)

    @staticmethod
    def _get_val(row: pd.Series, names: list) -> Optional[float]:
        """Try to get a feature value from a list of possible column names."""
        for name in names:
            if name in row.index:
                val = row[name]
                if pd.notna(val) and np.isfinite(val):
                    return float(val)
        return None
