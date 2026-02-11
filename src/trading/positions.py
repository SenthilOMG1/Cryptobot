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
    side: str  # "long" for bought crypto, "short" for futures short
    amount: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    mode: str = "spot"  # "spot" or "futures"
    leverage: int = 1  # Current leverage for futures positions
    max_leverage: int = 1  # Max leverage allowed (from confidence at entry)
    entry_confidence: float = 0.0  # Confidence when position was opened

    def update_price(self, price: float):
        """Update position with current price. Direction-aware P&L."""
        self.current_price = price

        if self.side == "short":
            # Shorts profit when price goes DOWN
            self.unrealized_pnl = (self.entry_price - price) * self.amount
            self.unrealized_pnl_pct = ((self.entry_price - price) / self.entry_price) * 100
        else:
            # Longs profit when price goes UP
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

    def sync_futures_from_exchange(self, futures_pairs: list):
        """
        Sync open futures positions from OKX on startup.

        Args:
            futures_pairs: List of spot pair names enabled for futures
        """
        try:
            positions_data = self.okx.get_futures_positions()
            for pos in positions_data:
                inst_id = pos.get("instId", "")  # e.g. "SOL-USDT-SWAP"
                pos_amt = float(pos.get("pos", 0))
                if pos_amt == 0:
                    continue

                # Convert SWAP instrument back to spot pair name
                pair = inst_id.replace("-SWAP", "")
                if pair not in futures_pairs:
                    continue

                avg_price = float(pos.get("avgPx", 0))
                side = "long" if pos_amt > 0 else "short"
                amount = abs(pos_amt)

                key = f"{pair}:futures"
                if key not in self.positions:
                    mark_price = float(pos.get("markPx", avg_price))
                    self.positions[key] = Position(
                        pair=pair,
                        side=side,
                        amount=amount,
                        entry_price=avg_price,
                        entry_time=datetime.now(),
                        current_price=mark_price,
                        highest_price=mark_price,
                        lowest_price=mark_price,
                        mode="futures"
                    )
                    logger.info(f"Recovered futures position: {pair} {side} x{amount} @ ${avg_price}")

        except Exception as e:
            logger.error(f"Failed to sync futures positions: {e}")

    def add_position(
        self,
        pair: str,
        amount: float,
        entry_price: float,
        side: str = "long",
        mode: str = "spot",
        leverage: int = 1,
        max_leverage: int = 1,
        entry_confidence: float = 0.0
    ) -> Position:
        """
        Add a new position.

        Args:
            pair: Trading pair
            amount: Amount of crypto (or contracts for futures)
            entry_price: Entry price
            side: "long" or "short"
            mode: "spot" or "futures"
            leverage: Current leverage used
            max_leverage: Max leverage allowed for this position
            entry_confidence: Confidence score at entry

        Returns:
            Position object
        """
        position = Position(
            pair=pair,
            side=side,
            amount=amount,
            entry_price=entry_price,
            entry_time=datetime.now(),
            current_price=entry_price,
            highest_price=entry_price,
            lowest_price=entry_price,
            mode=mode,
            leverage=leverage,
            max_leverage=max_leverage,
            entry_confidence=entry_confidence
        )

        # Use different key for futures to allow both spot and futures on same pair
        key = f"{pair}:futures" if mode == "futures" else pair
        self.positions[key] = position
        logger.info(f"Position opened: {pair} ({mode} {side}), amount: {amount}, entry: ${entry_price}")

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
        for key, position in self.positions.items():
            try:
                # For futures positions, fetch swap ticker; for spot, use spot ticker
                if position.mode == "futures":
                    swap_id = self.okx.spot_to_swap(position.pair)
                    ticker = self.okx.get_ticker(swap_id)
                else:
                    ticker = self.okx.get_ticker(position.pair)
                current_price = float(ticker.get("last", 0))
                position.update_price(current_price)
            except Exception as e:
                logger.error(f"Failed to update price for {key}: {e}")

    def get_position(self, pair: str, mode: str = "spot") -> Optional[Position]:
        """Get a specific position."""
        key = f"{pair}:futures" if mode == "futures" else pair
        return self.positions.get(key)

    def has_position(self, pair: str, mode: str = None) -> bool:
        """Check if we have a position in this pair.
        If mode is None, checks both spot and futures."""
        if mode is None:
            return pair in self.positions or f"{pair}:futures" in self.positions
        key = f"{pair}:futures" if mode == "futures" else pair
        return key in self.positions

    def remove_futures_position(self, pair: str) -> Optional[Position]:
        """Remove a futures position after closing."""
        key = f"{pair}:futures"
        if key in self.positions:
            position = self.positions.pop(key)
            logger.info(
                f"Futures position closed: {pair}, P&L: ${position.unrealized_pnl:.2f} "
                f"({position.unrealized_pnl_pct:.2f}%)"
            )
            return position
        return None

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
                    "side": p.side,
                    "mode": p.mode,
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
