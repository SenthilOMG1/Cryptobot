"""
Funding Rate Filter
===================
Uses OKX funding rates as contrarian signals.
Extreme funding rates indicate crowded positioning.
"""

import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class FundingRateFilter:
    """
    Contrarian signals from OKX funding rates.

    - Extreme positive funding (>0.05%) = crowded longs = boost short confidence
    - Extreme negative funding (<-0.05%) = crowded shorts = boost long confidence
    """

    def __init__(
        self,
        okx_client,
        extreme_threshold: float = 0.0005,   # 0.05% funding rate
        confidence_boost: float = 0.08,       # How much to boost/reduce confidence
        cache_ttl: int = 1800                 # 30 min cache
    ):
        """
        Args:
            okx_client: SecureOKXClient instance
            extreme_threshold: Funding rate above which is considered extreme
            confidence_boost: Amount to adjust confidence by
            cache_ttl: Cache time-to-live in seconds
        """
        self.okx = okx_client
        self.extreme_threshold = extreme_threshold
        self.confidence_boost = confidence_boost
        self.cache_ttl = cache_ttl

        # Cache: {pair: (funding_rate, timestamp)}
        self._cache: Dict[str, Tuple[float, float]] = {}

    def get_funding_rate(self, pair: str) -> Optional[float]:
        """
        Get current funding rate for a pair.

        Args:
            pair: Trading pair (e.g., "SOL-USDT")

        Returns:
            Funding rate as float (e.g., 0.0001 = 0.01%), or None if unavailable
        """
        now = time.time()

        # Check cache
        if pair in self._cache:
            cached_rate, cached_time = self._cache[pair]
            if now - cached_time < self.cache_ttl:
                return cached_rate

        # Fetch from OKX
        try:
            data = self.okx.get_funding_rate(pair)
            if data and "fundingRate" in data:
                rate = float(data["fundingRate"])
                self._cache[pair] = (rate, now)
                return rate
        except Exception as e:
            logger.debug(f"Failed to get funding rate for {pair}: {e}")

        return None

    def adjust_decision_confidence(
        self,
        pair: str,
        action: int,
        confidence: float
    ) -> Tuple[float, str]:
        """
        Adjust confidence based on funding rate contrarian signal.

        Args:
            pair: Trading pair
            action: 1 = BUY (long), -1 = SELL (short)
            confidence: Original confidence score

        Returns:
            Tuple of (adjusted_confidence, reason_string)
        """
        if action == 0:
            return confidence, ""

        rate = self.get_funding_rate(pair)
        if rate is None:
            return confidence, ""

        adjustment = 0.0
        reason = ""

        if rate > self.extreme_threshold:
            # Crowded longs - boost shorts, reduce longs
            if action == -1:  # SHORT
                adjustment = self.confidence_boost
                reason = f"Funding +{rate*100:.3f}% (crowded longs) - boost short"
            elif action == 1:  # LONG
                adjustment = -self.confidence_boost * 0.5  # Smaller penalty for longs
                reason = f"Funding +{rate*100:.3f}% (crowded longs) - reduce long"

        elif rate < -self.extreme_threshold:
            # Crowded shorts - boost longs, reduce shorts
            if action == 1:  # LONG
                adjustment = self.confidence_boost
                reason = f"Funding {rate*100:.3f}% (crowded shorts) - boost long"
            elif action == -1:  # SHORT
                adjustment = -self.confidence_boost * 0.5
                reason = f"Funding {rate*100:.3f}% (crowded shorts) - reduce short"

        adjusted = max(0.0, min(1.0, confidence + adjustment))

        if adjustment != 0:
            logger.info(f"[{pair}] Funding adjustment: {confidence:.2f} -> {adjusted:.2f} ({reason})")

        return adjusted, reason

    def clear_cache(self):
        """Clear the funding rate cache."""
        self._cache.clear()
