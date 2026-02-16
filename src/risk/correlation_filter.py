"""
Correlation Filter
==================
Blocks new positions that are too correlated with existing positions
in the same direction, preventing concentration risk.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationFilter:
    """
    Prevents opening correlated positions in the same direction.

    Rules:
    - >0.85 correlation with same-direction position = blocked
    - Max 3 correlated (>0.70) same-direction positions allowed
    - Correlation calculated on 72h of hourly returns
    """

    def __init__(
        self,
        data_collector,
        max_correlation: float = 0.85,
        max_correlated_positions: int = 3,
        correlated_threshold: float = 0.70,
        lookback_hours: int = 72,
        cache_ttl: int = 900  # 15 min cache
    ):
        """
        Args:
            data_collector: DataCollector instance for fetching candles
            max_correlation: Block if correlation exceeds this with same-direction pos
            max_correlated_positions: Max positions correlated above correlated_threshold
            correlated_threshold: Threshold for counting as "correlated"
            lookback_hours: Hours of price data for correlation calculation
            cache_ttl: Cache time-to-live in seconds
        """
        self.collector = data_collector
        self.max_correlation = max_correlation
        self.max_correlated_positions = max_correlated_positions
        self.correlated_threshold = correlated_threshold
        self.lookback_hours = lookback_hours
        self.cache_ttl = cache_ttl

        # Cache: {pair: (returns_array, timestamp)}
        self._returns_cache: Dict[str, Tuple[np.ndarray, float]] = {}

    def should_allow_trade(
        self,
        pair: str,
        direction: int,
        existing_positions: list
    ) -> Tuple[bool, str]:
        """
        Check if a new trade should be allowed based on correlation with existing positions.

        Args:
            pair: The pair we want to trade (e.g., "SOL-USDT")
            direction: 1 for long, -1 for short
            existing_positions: List of position objects with .pair and .side attributes

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if not existing_positions:
            return True, "No existing positions"

        # Get same-direction positions
        same_dir_positions = []
        for pos in existing_positions:
            pos_dir = 1 if getattr(pos, "side", "long") == "long" else -1
            if pos_dir == direction and pos.pair != pair:
                same_dir_positions.append(pos)

        if not same_dir_positions:
            return True, "No same-direction positions"

        # Get returns for the new pair
        new_returns = self._get_returns(pair)
        if new_returns is None:
            return True, "Could not fetch returns data"

        high_corr_count = 0
        blocked_by = None

        for pos in same_dir_positions:
            pos_returns = self._get_returns(pos.pair)
            if pos_returns is None:
                continue

            corr = self._calculate_correlation(new_returns, pos_returns)
            if corr is None:
                continue

            # Check hard block
            if corr > self.max_correlation:
                blocked_by = pos.pair
                reason = (
                    f"Correlation {corr:.2f} with {pos.pair} "
                    f"(same direction: {'long' if direction == 1 else 'short'}) "
                    f"exceeds {self.max_correlation}"
                )
                logger.info(f"[{pair}] Correlation filter blocked: {reason}")
                return False, reason

            # Count correlated positions
            if corr > self.correlated_threshold:
                high_corr_count += 1

        # Check max correlated positions
        if high_corr_count >= self.max_correlated_positions:
            reason = (
                f"{high_corr_count} correlated same-direction positions "
                f"(max {self.max_correlated_positions})"
            )
            logger.info(f"[{pair}] Correlation filter blocked: {reason}")
            return False, reason

        return True, f"Passed (max corr with same-dir: {high_corr_count} correlated)"

    def _get_returns(self, pair: str) -> Optional[np.ndarray]:
        """Get hourly returns for a pair, with caching."""
        now = time.time()

        # Check cache
        if pair in self._returns_cache:
            cached_returns, cached_time = self._returns_cache[pair]
            if now - cached_time < self.cache_ttl:
                return cached_returns

        # Fetch fresh data
        try:
            candles = self.collector.get_candles(pair, "1h", min(self.lookback_hours, 300))
            if candles is None or len(candles) < 20:
                return None

            close = candles["close"].values.astype(float)
            returns = np.diff(close) / close[:-1]

            self._returns_cache[pair] = (returns, now)
            return returns

        except Exception as e:
            logger.debug(f"Failed to get returns for {pair}: {e}")
            return None

    @staticmethod
    def _calculate_correlation(returns_a: np.ndarray, returns_b: np.ndarray) -> Optional[float]:
        """Calculate Pearson correlation between two return series."""
        # Align lengths (use the shorter one from the end)
        min_len = min(len(returns_a), len(returns_b))
        if min_len < 20:
            return None

        a = returns_a[-min_len:]
        b = returns_b[-min_len:]

        # Remove NaN/Inf
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]

        if len(a) < 20:
            return None

        # Pearson correlation
        try:
            corr = np.corrcoef(a, b)[0, 1]
            if np.isfinite(corr):
                return float(corr)
        except Exception:
            pass

        return None

    def clear_cache(self):
        """Clear the returns cache (call at start of each trading cycle)."""
        self._returns_cache.clear()
