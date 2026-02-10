"""
Configuration Module
====================
Centralized configuration for the autonomous trading bot.
Loads settings from environment variables with secure defaults.
"""

import os
import logging
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Load .env file if exists (for local development)
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Trading strategy parameters."""

    # Trading pairs to monitor
    trading_pairs: List[str]

    # Position sizing
    max_position_percent: float = 25.0  # Max 25% of balance per trade
    min_trade_usdt: float = 5.0  # Minimum trade size in USDT

    # Risk management
    stop_loss_percent: float = 3.0  # Exit if down 3%
    take_profit_percent: float = 20.0  # Exit if up 20%
    trailing_stop_percent: float = 3.0  # Trail 3% below peak price
    daily_loss_limit_percent: float = 12.0  # Pause if daily loss > 12%
    max_open_positions: int = 3  # Maximum concurrent positions

    # AI confidence thresholds
    min_confidence_to_trade: float = 0.70  # Only trade if confidence > 70%
    min_confidence_for_large_trade: float = 0.85  # Large trades need 85%+

    # Timing
    analysis_interval_minutes: int = 30  # Analyze every 30 minutes
    model_retrain_days: int = 7  # Retrain models every 7 days


@dataclass
class ExchangeConfig:
    """OKX exchange configuration."""

    # API endpoints
    base_url: str = "https://www.okx.com"
    ws_public_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    ws_private_url: str = "wss://ws.okx.com:8443/ws/v5/private"

    # Demo trading (set to True for paper trading)
    demo_mode: bool = False

    # Rate limiting
    max_requests_per_second: int = 10
    order_rate_limit: int = 60  # Orders per minute


@dataclass
class ModelConfig:
    """ML model configuration."""

    # XGBoost parameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1

    # PPO parameters
    ppo_learning_rate: float = 0.0003
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64

    # Training
    lookback_days: int = 90  # Days of historical data for training
    train_test_split: float = 0.2  # 20% for validation


@dataclass
class FuturesConfig:
    """Futures/perpetual swap configuration."""

    enabled: bool = False
    leverage: int = 2
    margin_mode: str = "cross"  # "cross" or "isolated"
    futures_pairs: List[str] = None  # Pairs enabled for futures trading

    def __post_init__(self):
        if self.futures_pairs is None:
            self.futures_pairs = []


class Config:
    """
    Main configuration class.
    Loads all settings and validates them.
    """

    def __init__(self):
        """Initialize configuration from environment."""
        self._load_trading_config()
        self._load_exchange_config()
        self._load_model_config()
        self._load_futures_config()
        self._validate()

    def _load_trading_config(self):
        """Load trading parameters."""
        # Parse trading pairs from environment or use defaults
        pairs_str = os.environ.get("TRADING_PAIRS", "BTC-USDT,ETH-USDT,SOL-USDT")
        pairs = [p.strip() for p in pairs_str.split(",")]

        self.trading = TradingConfig(
            trading_pairs=pairs,
            max_position_percent=float(os.environ.get("MAX_POSITION_PERCENT", "25")),
            stop_loss_percent=float(os.environ.get("STOP_LOSS_PERCENT", "3")),
            take_profit_percent=float(os.environ.get("TAKE_PROFIT_PERCENT", "20")),
            trailing_stop_percent=float(os.environ.get("TRAILING_STOP_PERCENT", "3")),
            daily_loss_limit_percent=float(os.environ.get("DAILY_LOSS_LIMIT", "12")),
            max_open_positions=int(os.environ.get("MAX_OPEN_POSITIONS", "3")),
            min_confidence_to_trade=float(os.environ.get("MIN_CONFIDENCE", "0.70")),
            analysis_interval_minutes=int(os.environ.get("ANALYSIS_INTERVAL", "30")),
        )

    def _load_exchange_config(self):
        """Load exchange parameters."""
        demo_mode = os.environ.get("TRADING_MODE", "live").lower() == "demo"

        self.exchange = ExchangeConfig(
            demo_mode=demo_mode,
        )

        if demo_mode:
            logger.info("Running in DEMO MODE - no real trades will be made")

    def _load_model_config(self):
        """Load ML model parameters."""
        self.model = ModelConfig(
            lookback_days=int(os.environ.get("LOOKBACK_DAYS", "90")),
        )

    def _load_futures_config(self):
        """Load futures trading parameters."""
        futures_pairs_str = os.environ.get("FUTURES_PAIRS", "")
        futures_pairs = [p.strip() for p in futures_pairs_str.split(",") if p.strip()]

        self.futures = FuturesConfig(
            enabled=os.environ.get("FUTURES_ENABLED", "false").lower() == "true",
            leverage=int(os.environ.get("FUTURES_LEVERAGE", "2")),
            margin_mode=os.environ.get("FUTURES_MARGIN_MODE", "cross").lower(),
            futures_pairs=futures_pairs,
        )

    def _validate(self):
        """Validate configuration values."""
        # Validate trading pairs
        if not self.trading.trading_pairs:
            raise ValueError("At least one trading pair must be configured")

        # Validate percentages
        if not (0 < self.trading.max_position_percent <= 100):
            raise ValueError("max_position_percent must be between 0 and 100")

        if not (0 < self.trading.stop_loss_percent <= 50):
            raise ValueError("stop_loss_percent must be between 0 and 50")

        if not (0 < self.trading.take_profit_percent <= 100):
            raise ValueError("take_profit_percent must be between 0 and 100")

        # Validate confidence thresholds
        if not (0.5 <= self.trading.min_confidence_to_trade <= 1.0):
            raise ValueError("min_confidence_to_trade must be between 0.5 and 1.0")

        # Validate futures config
        if self.futures.enabled:
            if not (1 <= self.futures.leverage <= 3):
                raise ValueError("futures leverage must be between 1 and 3 (safety limit)")
            if self.futures.margin_mode not in ("cross", "isolated"):
                raise ValueError("futures margin_mode must be 'cross' or 'isolated'")
            if not self.futures.futures_pairs:
                raise ValueError("FUTURES_PAIRS must be set when FUTURES_ENABLED=true")
            for pair in self.futures.futures_pairs:
                if pair not in self.trading.trading_pairs:
                    raise ValueError(f"Futures pair {pair} must also be in TRADING_PAIRS")

        logger.info("Configuration validated successfully")
        logger.info(f"Trading pairs: {self.trading.trading_pairs}")
        logger.info(f"Demo mode: {self.exchange.demo_mode}")
        if self.futures.enabled:
            logger.info(f"Futures ENABLED: {self.futures.futures_pairs}, leverage={self.futures.leverage}x, margin={self.futures.margin_mode}")

    def __repr__(self):
        return (
            f"Config(\n"
            f"  trading_pairs={self.trading.trading_pairs},\n"
            f"  max_position={self.trading.max_position_percent}%,\n"
            f"  stop_loss={self.trading.stop_loss_percent}%,\n"
            f"  take_profit={self.trading.take_profit_percent}%,\n"
            f"  demo_mode={self.exchange.demo_mode}\n"
            f")"
        )


# Global config instance
_config = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
