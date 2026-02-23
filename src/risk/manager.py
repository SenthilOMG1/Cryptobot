"""
Risk Manager
============
The guardian of your capital.
Enforces risk rules and prevents catastrophic losses.
"""

import logging
from datetime import datetime, date
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_position_pct: float = 25.0       # Max 25% of portfolio per position
    stop_loss_pct: float = 5.0           # Exit if down 5%
    take_profit_pct: float = 20.0        # Exit if up 20%
    trailing_stop_pct: float = 3.0       # Trail 3% below peak price
    daily_loss_limit_pct: float = 12.0   # Pause if daily loss > 12%
    max_open_positions: int = 3          # Maximum concurrent positions
    min_confidence: float = 0.70         # Minimum confidence to trade
    max_trade_per_hour: int = 4          # Max trades per hour (prevent overtrading)


class RiskManager:
    """
    Risk management system.

    Protects capital through:
    - Position sizing based on confidence
    - Stop-loss enforcement
    - Take-profit triggers
    - Daily loss circuit breaker
    - Trade frequency limits
    """

    def __init__(
        self,
        position_tracker,
        limits: Optional[RiskLimits] = None
    ):
        """
        Initialize risk manager.

        Args:
            position_tracker: PositionTracker instance
            limits: RiskLimits configuration
        """
        self.positions = position_tracker
        self.limits = limits or RiskLimits()

        # Daily tracking
        self._daily_starting_value: float = 0.0
        self._daily_date: date = date.today()
        self._daily_trades: int = 0
        self._hourly_trades: Dict[int, int] = {}  # hour -> count

        # State
        self._paused = False
        self._pause_reason = ""

        logger.info(f"Risk manager initialized with limits: {self.limits}")

    def validate_trade(self, decision) -> bool:
        """
        Validate a trade decision against risk rules.

        Args:
            decision: TradeDecision from ensemble

        Returns:
            True if trade is allowed, False otherwise
        """
        # Check if trading is paused
        if self._paused:
            logger.warning(f"Trading paused: {self._pause_reason}")
            return False

        # Check minimum confidence
        if decision.confidence < self.limits.min_confidence:
            logger.info(f"Trade rejected: confidence {decision.confidence:.2f} < {self.limits.min_confidence}")
            return False

        # Check position count (for BUY and short-open orders)
        if decision.action in (1, -1):  # BUY or SELL (potential short open)
            current_positions = len(self.positions.get_all_positions())
            if current_positions >= self.limits.max_open_positions:
                logger.warning(f"Trade rejected: max positions reached ({current_positions})")
                return False

        # Check trade frequency
        current_hour = datetime.now().hour
        hourly_count = self._hourly_trades.get(current_hour, 0)
        if hourly_count >= self.limits.max_trade_per_hour:
            logger.warning(f"Trade rejected: hourly limit reached ({hourly_count})")
            return False

        # All checks passed
        logger.info(f"Trade validated: {decision.pair} {decision.action}")
        return True

    def calculate_position_size(
        self,
        balance: float,
        confidence: float,
        current_positions: int,
        volatility_scale: float = 1.0
    ) -> float:
        """
        Calculate safe position size based on confidence, portfolio state, and volatility.

        Args:
            balance: Available USDT balance
            confidence: Model confidence (0.0 to 1.0)
            current_positions: Number of existing positions
            volatility_scale: Multiplier from regime detector (1.0=normal, 0.4=high vol)

        Returns:
            Maximum trade amount in USDT
        """
        # Base size is max_position_pct
        max_pct = self.limits.max_position_pct

        # Scale by confidence (higher confidence = larger position)
        # Range: 40% to 100% of max_pct based on confidence
        confidence_scale = 0.4 + (confidence - self.limits.min_confidence) * 2
        confidence_scale = min(1.0, max(0.4, confidence_scale))

        # Reduce size if we have existing positions (diversification)
        position_scale = 1.0 - (current_positions * 0.15)  # -15% per existing position
        position_scale = max(0.5, position_scale)

        # Apply volatility scaling (high vol = smaller positions, not blocked trades)
        vol_scale = max(0.0, min(1.0, volatility_scale))

        # Final percentage
        final_pct = max_pct * confidence_scale * position_scale * vol_scale

        # Calculate actual amount
        trade_amount = balance * (final_pct / 100)

        logger.debug(
            f"Position size: ${trade_amount:.2f} "
            f"({final_pct:.1f}% of ${balance:.2f}, vol_scale={vol_scale:.2f})"
        )

        return trade_amount

    def check_stop_loss(self, position, atr_pct: float = 0.0) -> bool:
        """
        Check if position should be stopped out.
        Uses adaptive stop: max(configured_stop, 1.5 * ATR%) to avoid
        getting stopped by normal volatility.

        Args:
            position: Position object
            atr_pct: Current ATR as percentage of price (0-100 scale)

        Returns:
            True if stop-loss triggered
        """
        # Adaptive stop loss: max(configured, 1.5 * ATR%)
        adaptive_stop = self.limits.stop_loss_pct
        if atr_pct > 0:
            atr_stop = 1.5 * atr_pct
            adaptive_stop = max(self.limits.stop_loss_pct, atr_stop)
            # Cap at 10% to prevent runaway stops
            adaptive_stop = min(adaptive_stop, 10.0)

        if position.unrealized_pnl_pct <= -adaptive_stop:
            logger.warning(
                f"STOP-LOSS triggered for {position.pair}: "
                f"{position.unrealized_pnl_pct:.2f}% < -{adaptive_stop:.1f}% "
                f"(base: {self.limits.stop_loss_pct}%, ATR: {atr_pct:.2f}%)"
            )
            return True
        return False

    def check_trailing_stop(self, position) -> bool:
        """
        Check if trailing stop is triggered.

        For longs: activates after 1%+ profit, triggers when price drops from peak.
        For shorts: activates after 1%+ profit, triggers when price rises from trough.

        Args:
            position: Position object with highest_price/lowest_price tracked

        Returns:
            True if trailing stop triggered
        """
        if getattr(position, 'side', 'long') == "short":
            # SHORT: trailing stop tracks the trough (lowest price)
            # Activate after position has been at least 1% profitable (price dropped 1%+)
            if position.lowest_price > position.entry_price * 0.99:
                return False

            # Calculate how far price has risen from trough
            if position.lowest_price <= 0:
                return False
            rise_from_trough_pct = (
                (position.current_price - position.lowest_price) / position.lowest_price
            ) * 100

            if rise_from_trough_pct >= self.limits.trailing_stop_pct:
                trailing_stop_price = position.lowest_price * (1 + self.limits.trailing_stop_pct / 100)
                locked_profit_pct = ((position.entry_price - trailing_stop_price) / position.entry_price) * 100

                logger.info(
                    f"TRAILING STOP triggered for {position.pair} (SHORT): "
                    f"trough ${position.lowest_price:.2f} → now ${position.current_price:.2f} "
                    f"(rose {rise_from_trough_pct:.1f}% from trough, locking ~{locked_profit_pct:.1f}% profit)"
                )
                return True

            return False
        else:
            # LONG: trailing stop tracks the peak (highest price)
            # Only activate after position has been at least 1% profitable
            if position.highest_price < position.entry_price * 1.01:
                return False

            # Calculate how far price has dropped from peak
            drop_from_peak_pct = (
                (position.highest_price - position.current_price) / position.highest_price
            ) * 100

            if drop_from_peak_pct >= self.limits.trailing_stop_pct:
                trailing_stop_price = position.highest_price * (1 - self.limits.trailing_stop_pct / 100)
                locked_profit_pct = ((trailing_stop_price - position.entry_price) / position.entry_price) * 100

                logger.info(
                    f"TRAILING STOP triggered for {position.pair}: "
                    f"peak ${position.highest_price:.2f} → now ${position.current_price:.2f} "
                    f"(dropped {drop_from_peak_pct:.1f}% from peak, locking ~{locked_profit_pct:.1f}% profit)"
                )
                return True

            return False

    def check_take_profit(self, position) -> bool:
        """
        Check if position should take profit.

        Args:
            position: Position object

        Returns:
            True if take-profit triggered
        """
        if position.unrealized_pnl_pct >= self.limits.take_profit_pct:
            logger.info(
                f"TAKE-PROFIT triggered for {position.pair}: "
                f"{position.unrealized_pnl_pct:.2f}% >= {self.limits.take_profit_pct}%"
            )
            return True
        return False

    def should_pause_trading(self) -> bool:
        """
        Check if trading should be paused (circuit breaker).

        Returns:
            True if trading should stop
        """
        # Reset daily tracking if new day
        today = date.today()
        if today != self._daily_date:
            self._reset_daily_stats()

        # Get current portfolio value
        summary = self.positions.get_portfolio_summary()
        current_value = summary["total_value"]

        # Initialize starting value if not set
        if self._daily_starting_value == 0:
            self._daily_starting_value = current_value
            return False

        # Calculate daily P&L
        daily_pnl_pct = (
            (current_value - self._daily_starting_value) / self._daily_starting_value
        ) * 100

        # Check if daily loss limit exceeded
        if daily_pnl_pct <= -self.limits.daily_loss_limit_pct:
            self._paused = True
            self._pause_reason = f"Daily loss limit exceeded: {daily_pnl_pct:.2f}%"
            logger.error(f"CIRCUIT BREAKER: {self._pause_reason}")
            return True

        return False

    def record_trade(self):
        """Record that a trade was made (for frequency tracking)."""
        self._daily_trades += 1

        current_hour = datetime.now().hour
        self._hourly_trades[current_hour] = self._hourly_trades.get(current_hour, 0) + 1

    def _reset_daily_stats(self):
        """Reset daily statistics."""
        logger.info("Resetting daily risk statistics")
        self._daily_date = date.today()
        self._daily_trades = 0
        self._hourly_trades = {}
        self._paused = False
        self._pause_reason = ""

        # Set new starting value
        summary = self.positions.get_portfolio_summary()
        self._daily_starting_value = summary["total_value"]

    def resume_trading(self):
        """Manually resume trading after pause."""
        self._paused = False
        self._pause_reason = ""
        logger.info("Trading resumed")

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        summary = self.positions.get_portfolio_summary()
        current_value = summary["total_value"]

        daily_pnl = 0
        daily_pnl_pct = 0
        if self._daily_starting_value > 0:
            daily_pnl = current_value - self._daily_starting_value
            daily_pnl_pct = (daily_pnl / self._daily_starting_value) * 100

        return {
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "daily_starting_value": self._daily_starting_value,
            "current_value": current_value,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "daily_trades": self._daily_trades,
            "open_positions": len(self.positions.get_all_positions()),
            "max_positions": self.limits.max_open_positions,
            "limits": {
                "stop_loss_pct": self.limits.stop_loss_pct,
                "take_profit_pct": self.limits.take_profit_pct,
                "daily_loss_limit_pct": self.limits.daily_loss_limit_pct,
                "min_confidence": self.limits.min_confidence,
            }
        }

    def evaluate_market_conditions(self, volatility: float) -> Dict[str, Any]:
        """
        Evaluate if market conditions are suitable for trading.

        Args:
            volatility: Current market volatility (e.g., from ATR)

        Returns:
            Dict with recommendation
        """
        # High volatility warning
        if volatility > 0.1:  # 10%+ volatility
            return {
                "suitable": False,
                "reason": "High volatility - reduce position sizes",
                "suggested_position_scale": 0.5
            }

        # Extreme volatility - stop trading
        if volatility > 0.15:
            return {
                "suitable": False,
                "reason": "Extreme volatility - trading suspended",
                "suggested_position_scale": 0
            }

        return {
            "suitable": True,
            "reason": "Normal market conditions",
            "suggested_position_scale": 1.0
        }
