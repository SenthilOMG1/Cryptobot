"""
Position Tracker
================
Tracks open positions and their P&L.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open trading position."""
    pair: str
    side: str  # "long" for bought crypto
    amount: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0

    def update_price(self, price: float):
        """Update position with current price."""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.amount
        self.unrealized_pnl_pct = ((price - self.entry_price) / self.entry_price) * 100

        # Track highest/lowest for trailing stops
        if price > self.highest_price:
            self.highest_price = price
        if self.lowest_price == 0 or price < self.lowest_price:
            self.lowest_price = price


class PositionTracker:
    """
    Tracks all open positions and their performance.

    Features:
    - Real-time P&L tracking
    - Position history
    - Support for multiple positions
    """

    def __init__(self, okx_client):
        """
        Initialize position tracker.

        Args:
            okx_client: SecureOKXClient instance
        """
        self.okx = okx_client
        self.positions: Dict[str, Position] = {}

    def sync_from_exchange(self):
        """
        Sync positions from exchange.

        This recovers positions after restart.
        """
        try:
            # Get account balances
            balances = self.okx.get_balance()

            # Filter out USDT and small dust balances
            for currency, amount in balances.items():
                if currency == "USDT":
                    continue

                pair = f"{currency}-USDT"

                # Skip tiny balances (dust)
                try:
                    current_price = float(self.okx.get_ticker(pair).get("last", 0))
                    value = amount * current_price
                    if value < 1:  # Less than $1
                        continue
                except:
                    continue

                # Create position if not already tracked
                if pair not in self.positions:
                    # We don't know the entry price after restart
                    # Use current price as estimate
                    self.positions[pair] = Position(
                        pair=pair,
                        side="long",
                        amount=amount,
                        entry_price=current_price,  # Estimate
                        entry_time=datetime.now(),
                        current_price=current_price,
                        highest_price=current_price,
                        lowest_price=current_price
                    )
                    logger.info(f"Recovered position: {pair}, amount: {amount}")

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")

    def add_position(
        self,
        pair: str,
        amount: float,
        entry_price: float
    ) -> Position:
        """
        Add a new position after a buy.

        Args:
            pair: Trading pair
            amount: Amount of crypto bought
            entry_price: Entry price

        Returns:
            Position object
        """
        position = Position(
            pair=pair,
            side="long",
            amount=amount,
            entry_price=entry_price,
            entry_time=datetime.now(),
            current_price=entry_price,
            highest_price=entry_price,
            lowest_price=entry_price
        )

        self.positions[pair] = position
        logger.info(f"Position opened: {pair}, amount: {amount}, entry: ${entry_price}")

        return position

    def remove_position(self, pair: str) -> Optional[Position]:
        """
        Remove a position after selling.

        Returns:
            The closed position (for P&L tracking)
        """
        if pair in self.positions:
            position = self.positions.pop(pair)
            logger.info(
                f"Position closed: {pair}, P&L: ${position.unrealized_pnl:.2f} "
                f"({position.unrealized_pnl_pct:.2f}%)"
            )
            return position
        return None

    def update_prices(self):
        """Update all positions with current prices."""
        for pair, position in self.positions.items():
            try:
                ticker = self.okx.get_ticker(pair)
                current_price = float(ticker.get("last", 0))
                position.update_price(current_price)
            except Exception as e:
                logger.error(f"Failed to update price for {pair}: {e}")

    def get_position(self, pair: str) -> Optional[Position]:
        """Get a specific position."""
        return self.positions.get(pair)

    def has_position(self, pair: str) -> bool:
        """Check if we have a position in this pair."""
        return pair in self.positions

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_total_value(self) -> float:
        """Get total value of all positions in USDT."""
        total = 0.0
        for position in self.positions.values():
            total += position.amount * position.current_price
        return total

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        total = 0.0
        for position in self.positions.values():
            total += position.unrealized_pnl
        return total

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        self.update_prices()

        usdt_balance = self.okx.get_usdt_balance()
        positions_value = self.get_total_value()
        total_value = usdt_balance + positions_value
        unrealized_pnl = self.get_total_unrealized_pnl()

        return {
            "usdt_balance": usdt_balance,
            "positions_value": positions_value,
            "total_value": total_value,
            "unrealized_pnl": unrealized_pnl,
            "num_positions": len(self.positions),
            "positions": [
                {
                    "pair": p.pair,
                    "amount": p.amount,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "pnl": p.unrealized_pnl,
                    "pnl_pct": p.unrealized_pnl_pct
                }
                for p in self.positions.values()
            ]
        }

    def check_stop_loss(self, pair: str, stop_loss_pct: float) -> bool:
        """
        Check if position has hit stop-loss.

        Args:
            pair: Trading pair
            stop_loss_pct: Stop loss percentage (e.g., 8 for 8%)

        Returns:
            True if stop-loss triggered
        """
        position = self.positions.get(pair)
        if not position:
            return False

        return position.unrealized_pnl_pct <= -stop_loss_pct

    def check_take_profit(self, pair: str, take_profit_pct: float) -> bool:
        """
        Check if position has hit take-profit.

        Args:
            pair: Trading pair
            take_profit_pct: Take profit percentage (e.g., 20 for 20%)

        Returns:
            True if take-profit triggered
        """
        position = self.positions.get(pair)
        if not position:
            return False

        return position.unrealized_pnl_pct >= take_profit_pct

    def get_position_age_hours(self, pair: str) -> float:
        """Get how long a position has been open in hours."""
        position = self.positions.get(pair)
        if not position:
            return 0

        age = datetime.now() - position.entry_time
        return age.total_seconds() / 3600
