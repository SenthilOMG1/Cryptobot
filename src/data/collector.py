"""
Data Collector Module
=====================
Fetches market data (OHLCV) from OKX exchange.
Handles caching and rate limiting.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects market data from OKX exchange.

    Features:
    - Fetches OHLCV candlestick data
    - Caches data to reduce API calls
    - Handles rate limiting
    - Supports multiple timeframes
    """

    # OKX timeframe mappings
    TIMEFRAMES = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
    }

    def __init__(self, okx_client):
        """
        Initialize data collector.

        Args:
            okx_client: Initialized OKX client for API calls
        """
        self.okx = okx_client
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl_minutes = 5  # Cache expires after 5 minutes

    def get_candles(
        self,
        pair: str,
        timeframe: str = "1h",
        limit: int = 100,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data for a trading pair.

        Args:
            pair: Trading pair (e.g., "BTC-USDT")
            timeframe: Candle timeframe ("1m", "5m", "15m", "30m", "1h", "4h", "1d")
            limit: Number of candles to fetch (max 300)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        cache_key = f"{pair}_{timeframe}_{limit}"

        # Check cache
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Using cached data for {cache_key}")
            return self._cache[cache_key].copy()

        # Validate timeframe
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Invalid timeframe. Choose from: {list(self.TIMEFRAMES.keys())}")

        okx_timeframe = self.TIMEFRAMES[timeframe]

        try:
            # Fetch from OKX
            logger.debug(f"Fetching {limit} candles for {pair} ({timeframe})")
            candles = self.okx.get_candles(
                inst_id=pair,
                bar=okx_timeframe,
                limit=min(limit, 300)  # OKX max is 300
            )

            # Convert to DataFrame
            df = self._candles_to_dataframe(candles)

            # Cache the result
            self._cache[cache_key] = df.copy()
            self._cache_timestamps[cache_key] = datetime.now()

            logger.info(f"Fetched {len(df)} candles for {pair}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch candles for {pair}: {e}")
            # Return cached data if available (even if expired)
            if cache_key in self._cache:
                logger.warning(f"Returning expired cache for {cache_key}")
                return self._cache[cache_key].copy()
            raise

    def _candles_to_dataframe(self, candles: List) -> pd.DataFrame:
        """
        Convert OKX candle data to DataFrame.

        OKX returns: [timestamp, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
        """
        if not candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low", "close",
            "volume", "volume_ccy", "volume_quote", "confirm"
        ])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Sort by timestamp ascending
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Keep only essential columns
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False

        cache_time = self._cache_timestamps.get(cache_key)
        if not cache_time:
            return False

        age = datetime.now() - cache_time
        return age.total_seconds() < (self._cache_ttl_minutes * 60)

    def get_current_price(self, pair: str) -> float:
        """
        Get the current price for a trading pair.

        Args:
            pair: Trading pair (e.g., "BTC-USDT")

        Returns:
            Current price as float
        """
        try:
            ticker = self.okx.get_ticker(inst_id=pair)
            return float(ticker.get("last", 0))
        except Exception as e:
            logger.error(f"Failed to get price for {pair}: {e}")
            # Fallback to latest candle close price
            candles = self.get_candles(pair, "1m", 1)
            if not candles.empty:
                return candles["close"].iloc[-1]
            raise

    def get_ticker(self, pair: str) -> Dict[str, Any]:
        """
        Get full ticker data for a trading pair.

        Returns:
            Dict with: last, bid, ask, high24h, low24h, vol24h, etc.
        """
        try:
            return self.okx.get_ticker(inst_id=pair)
        except Exception as e:
            logger.error(f"Failed to get ticker for {pair}: {e}")
            raise

    def get_orderbook(self, pair: str, depth: int = 20) -> Dict[str, List]:
        """
        Get order book data.

        Args:
            pair: Trading pair
            depth: Number of levels to fetch (max 400)

        Returns:
            Dict with 'bids' and 'asks' lists
        """
        try:
            return self.okx.get_orderbook(inst_id=pair, sz=depth)
        except Exception as e:
            logger.error(f"Failed to get orderbook for {pair}: {e}")
            raise

    def get_historical_data(
        self,
        pair: str,
        days: int = 90,
        timeframe: str = "1h"
    ) -> pd.DataFrame:
        """
        Fetch historical data for model training.

        Args:
            pair: Trading pair
            days: Number of days of history
            timeframe: Candle timeframe

        Returns:
            DataFrame with historical OHLCV data
        """
        all_candles = []
        end_time = None

        # Calculate how many candles we need
        timeframe_hours = {
            "1m": 1/60, "5m": 5/60, "15m": 15/60, "30m": 30/60,
            "1h": 1, "4h": 4, "1d": 24
        }
        hours_per_candle = timeframe_hours.get(timeframe, 1)
        total_candles = int((days * 24) / hours_per_candle)

        logger.info(f"Fetching {total_candles} candles ({days} days) for {pair}")

        # Fetch in batches of 300 (OKX limit)
        while len(all_candles) < total_candles:
            try:
                candles = self.okx.get_candles(
                    inst_id=pair,
                    bar=self.TIMEFRAMES[timeframe],
                    limit=300,
                    after=end_time
                )

                if not candles:
                    break

                all_candles.extend(candles)
                end_time = candles[-1][0]  # Timestamp of oldest candle

                # Rate limiting
                time.sleep(0.1)

                logger.debug(f"Fetched {len(all_candles)}/{total_candles} candles")

            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                break

        if not all_candles:
            raise ValueError(f"No historical data available for {pair}")

        df = self._candles_to_dataframe(all_candles)
        logger.info(f"Fetched {len(df)} historical candles for {pair}")

        return df

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cleared data cache")
