"""
Feature Engineering Module
==========================
Calculates 50+ technical indicators for ML model input.
These features are what the AI learns patterns from.
"""

import logging
import numpy as np
import pandas as pd
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Calculates technical analysis features for ML models.

    Features include:
    - Trend indicators (SMA, EMA, MACD, ADX)
    - Momentum indicators (RSI, Stochastic, Williams %R)
    - Volatility indicators (Bollinger Bands, ATR)
    - Volume indicators (OBV, VWAP)
    - Price patterns and derived features
    """

    def __init__(self):
        """Initialize feature engine."""
        self.feature_names = []

    def calculate_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators from OHLCV data.

        Args:
            candles: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            DataFrame with 50+ feature columns
        """
        if candles.empty:
            raise ValueError("Cannot calculate features from empty data")

        if len(candles) < 50:
            logger.warning(f"Only {len(candles)} candles - some features may be NaN")

        df = candles.copy()

        # Ensure proper column names
        df.columns = [c.lower() for c in df.columns]

        logger.debug(f"Calculating features for {len(df)} candles")

        # Calculate all feature groups
        df = self._add_trend_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        df = self._add_price_features(df)
        df = self._add_pattern_features(df)

        # Drop rows with NaN (from indicator warm-up period)
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.debug(f"Dropped {dropped} rows with NaN values")

        # Store feature names (excluding price columns)
        price_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        self.feature_names = [c for c in df.columns if c not in price_cols]

        logger.info(f"Calculated {len(self.feature_names)} features")
        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Simple Moving Averages
        for period in [7, 14, 21, 50, 100, 200]:
            if len(df) >= period:
                df[f"sma_{period}"] = SMAIndicator(close, window=period).sma_indicator()

        # Exponential Moving Averages
        for period in [9, 12, 21, 26, 50]:
            if len(df) >= period:
                df[f"ema_{period}"] = EMAIndicator(close, window=period).ema_indicator()

        # MACD
        if len(df) >= 26:
            macd = MACD(close)
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_hist"] = macd.macd_diff()

        # ADX (Average Directional Index) - trend strength
        if len(df) >= 14:
            adx = ADXIndicator(high, low, close)
            df["adx"] = adx.adx()
            df["adx_pos"] = adx.adx_pos()
            df["adx_neg"] = adx.adx_neg()

        # Price vs Moving Averages (important signals!)
        if "sma_50" in df.columns:
            df["price_vs_sma50"] = (close - df["sma_50"]) / df["sma_50"]
        if "sma_200" in df.columns:
            df["price_vs_sma200"] = (close - df["sma_200"]) / df["sma_200"]
        if "ema_21" in df.columns:
            df["price_vs_ema21"] = (close - df["ema_21"]) / df["ema_21"]

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            if len(df) >= period:
                df[f"rsi_{period}"] = RSIIndicator(close, window=period).rsi()

        # Stochastic Oscillator
        if len(df) >= 14:
            stoch = StochasticOscillator(high, low, close)
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()

        # Williams %R
        if len(df) >= 14:
            df["williams_r"] = WilliamsRIndicator(high, low, close).williams_r()

        # Rate of Change (momentum)
        for period in [5, 10, 20]:
            if len(df) >= period:
                df[f"roc_{period}"] = close.pct_change(periods=period) * 100

        # Momentum
        for period in [10, 20]:
            if len(df) >= period:
                df[f"momentum_{period}"] = close - close.shift(period)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Bollinger Bands
        if len(df) >= 20:
            bb = BollingerBands(close)
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_middle"] = bb.bollinger_mavg()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Average True Range (ATR)
        for period in [7, 14, 21]:
            if len(df) >= period:
                df[f"atr_{period}"] = AverageTrueRange(high, low, close, window=period).average_true_range()

        # Volatility (standard deviation of returns)
        for period in [7, 14, 30]:
            if len(df) >= period:
                df[f"volatility_{period}"] = close.pct_change().rolling(window=period).std() * np.sqrt(period)

        # High-Low Range
        df["hl_range"] = (high - low) / close
        df["hl_range_avg"] = df["hl_range"].rolling(window=14).mean()

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # On-Balance Volume (OBV)
        df["obv"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        # OBV trend
        if len(df) >= 20:
            df["obv_sma"] = df["obv"].rolling(window=20).mean()
            df["obv_trend"] = (df["obv"] - df["obv_sma"]) / df["obv_sma"].abs().replace(0, 1)

        # Volume Moving Averages
        for period in [10, 20, 50]:
            if len(df) >= period:
                df[f"volume_sma_{period}"] = volume.rolling(window=period).mean()

        # Volume relative to average
        if "volume_sma_20" in df.columns:
            df["volume_ratio"] = volume / df["volume_sma_20"].replace(0, 1)

        # VWAP (Volume Weighted Average Price)
        if len(df) >= 14:
            try:
                df["vwap"] = VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
                df["price_vs_vwap"] = (close - df["vwap"]) / df["vwap"]
            except:
                pass  # VWAP can fail with certain data

        # Money Flow
        df["money_flow"] = close * volume
        if len(df) >= 14:
            df["money_flow_avg"] = df["money_flow"].rolling(window=14).mean()

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-derived features."""
        close = df["close"]
        open_price = df["open"]
        high = df["high"]
        low = df["low"]

        # Returns at different periods
        for period in [1, 3, 5, 10, 20]:
            df[f"return_{period}"] = close.pct_change(periods=period)

        # Cumulative returns
        df["cum_return_5"] = (1 + df["return_1"]).rolling(window=5).apply(lambda x: x.prod() - 1, raw=True)
        df["cum_return_20"] = (1 + df["return_1"]).rolling(window=20).apply(lambda x: x.prod() - 1, raw=True)

        # Price position within candle
        df["candle_body"] = (close - open_price) / close
        df["candle_upper_shadow"] = (high - close.combine(open_price, max)) / close
        df["candle_lower_shadow"] = (close.combine(open_price, min) - low) / close

        # Higher highs / Lower lows
        df["higher_high"] = (high > high.shift(1)).astype(int)
        df["lower_low"] = (low < low.shift(1)).astype(int)

        # Distance from recent high/low
        if len(df) >= 20:
            df["dist_from_20h"] = (close - high.rolling(20).max()) / close
            df["dist_from_20l"] = (close - low.rolling(20).min()) / close

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        close = df["close"]
        open_price = df["open"]
        high = df["high"]
        low = df["low"]

        # Consecutive up/down days
        df["up_day"] = (close > close.shift(1)).astype(int)
        df["down_day"] = (close < close.shift(1)).astype(int)

        # Streak counting
        df["up_streak"] = df["up_day"].groupby((df["up_day"] != df["up_day"].shift()).cumsum()).cumsum()
        df["down_streak"] = df["down_day"].groupby((df["down_day"] != df["down_day"].shift()).cumsum()).cumsum()

        # Doji detection (small body candles)
        body_size = abs(close - open_price)
        candle_range = high - low
        df["is_doji"] = (body_size < candle_range * 0.1).astype(int)

        # Gap detection
        df["gap_up"] = (open_price > high.shift(1)).astype(int)
        df["gap_down"] = (open_price < low.shift(1)).astype(int)

        # Support/Resistance proximity
        if len(df) >= 50:
            df["near_resistance"] = (high >= high.rolling(50).max() * 0.99).astype(int)
            df["near_support"] = (low <= low.rolling(50).min() * 1.01).astype(int)

        return df

    def get_feature_names(self) -> list:
        """Return list of calculated feature names."""
        return self.feature_names

    def get_feature_importance_columns(self) -> list:
        """
        Return the most important features based on trading research.
        These should be prioritized in ML models.
        """
        return [
            # Top trend indicators
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "price_vs_sma50", "price_vs_ema21", "adx",

            # Top volatility indicators
            "bb_position", "bb_width", "atr_14", "volatility_14",

            # Top volume indicators
            "volume_ratio", "obv_trend", "price_vs_vwap",

            # Top momentum indicators
            "roc_10", "momentum_10", "stoch_k",

            # Top price features
            "return_1", "return_5", "cum_return_5",

            # Top pattern features
            "up_streak", "down_streak", "near_resistance", "near_support"
        ]


def create_target_labels(df: pd.DataFrame, lookahead: int = 1, threshold: float = 0.02) -> pd.Series:
    """
    Create target labels for ML training.

    Labels:
    - 1 (BUY): Price goes up more than threshold
    - -1 (SELL): Price goes down more than threshold
    - 0 (HOLD): Price stays within threshold

    Args:
        df: DataFrame with 'close' column
        lookahead: How many periods ahead to look
        threshold: Minimum price change to trigger signal (0.02 = 2%)

    Returns:
        Series with labels
    """
    future_return = df["close"].pct_change(periods=lookahead).shift(-lookahead)

    labels = pd.Series(0, index=df.index)
    labels[future_return > threshold] = 1   # BUY signal
    labels[future_return < -threshold] = -1  # SELL signal

    return labels
