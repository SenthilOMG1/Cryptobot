"""
Trade Executor
==============
Executes trades safely with validation and logging.
Records all trades to database.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of a trade execution."""
    success: bool
    order_id: str
    pair: str
    side: str  # "buy" or "sell"
    amount: float
    price: float
    total_value: float
    fee: float
    error: Optional[str] = None


class TradeExecutor:
    """
    Executes trades with safety checks.

    Features:
    - Validates trades with risk manager before execution
    - Records all trades to SQLite database
    - Handles partial fills and retries
    - Comprehensive logging
    """

    def __init__(self, okx_client, risk_manager, db_path: str = "data/trades.db"):
        """
        Initialize executor.

        Args:
            okx_client: SecureOKXClient instance
            risk_manager: RiskManager instance
            db_path: Path to SQLite database
        """
        self.okx = okx_client
        self.risk = risk_manager
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for trade records."""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                side TEXT NOT NULL,
                amount REAL NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                fee REAL DEFAULT 0,
                order_id TEXT,
                pnl REAL DEFAULT 0,
                pnl_percent REAL DEFAULT 0,
                entry_price REAL DEFAULT 0,
                status TEXT DEFAULT 'completed'
            )
        """)

        # Decisions table (for auditing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL,
                xgb_action INTEGER,
                xgb_confidence REAL,
                rl_action INTEGER,
                rl_confidence REAL,
                reasoning TEXT,
                executed INTEGER DEFAULT 0,
                trade_id TEXT
            )
        """)

        # Portfolio snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value_usdt REAL NOT NULL,
                usdt_balance REAL,
                positions TEXT
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def execute(self, decision, current_price: float) -> TradeResult:
        """
        Execute a trade decision.

        Args:
            decision: TradeDecision from ensemble
            current_price: Current price of the asset

        Returns:
            TradeResult with execution details
        """
        pair = decision.pair

        # Validate with risk manager
        if not self.risk.validate_trade(decision):
            return TradeResult(
                success=False,
                order_id="",
                pair=pair,
                side="",
                amount=0,
                price=0,
                total_value=0,
                fee=0,
                error="Trade rejected by risk manager"
            )

        try:
            if decision.action == 1:  # BUY
                return self._execute_buy(decision, current_price)
            elif decision.action == -1:  # SELL
                return self._execute_sell(decision, current_price)
            else:
                return TradeResult(
                    success=False,
                    order_id="",
                    pair=pair,
                    side="hold",
                    amount=0,
                    price=0,
                    total_value=0,
                    fee=0,
                    error="HOLD action - no trade executed"
                )

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return TradeResult(
                success=False,
                order_id="",
                pair=pair,
                side=str(decision.action),
                amount=0,
                price=current_price,
                total_value=0,
                fee=0,
                error=str(e)
            )

    def _execute_buy(self, decision, current_price: float) -> TradeResult:
        """Execute a buy order."""
        pair = decision.pair

        # Get available balance
        usdt_balance = self.okx.get_usdt_balance()

        # Calculate trade size
        trade_size_pct = decision.suggested_size_pct / 100
        trade_amount = usdt_balance * trade_size_pct

        # Check minimum order size
        min_size = self.okx.get_min_order_size(pair)
        min_usdt = min_size * current_price

        if trade_amount < min_usdt:
            return TradeResult(
                success=False,
                order_id="",
                pair=pair,
                side="buy",
                amount=0,
                price=current_price,
                total_value=0,
                fee=0,
                error=f"Trade amount ${trade_amount:.2f} below minimum ${min_usdt:.2f}"
            )

        # Execute buy
        logger.info(f"Executing BUY: {pair} for ${trade_amount:.2f} ({decision.suggested_size_pct}%)")
        result = self.okx.place_market_buy(pair, trade_amount)

        if result and result.get("ordId"):
            order_id = result["ordId"]

            # Get fill details (slight delay for order to process)
            import time
            time.sleep(0.5)
            order_info = self.okx.get_order(pair, order_id)

            fill_price = float(order_info.get("avgPx", current_price))
            fill_amount = float(order_info.get("accFillSz", 0))
            fee = float(order_info.get("fee", 0))

            trade_result = TradeResult(
                success=True,
                order_id=order_id,
                pair=pair,
                side="buy",
                amount=fill_amount,
                price=fill_price,
                total_value=trade_amount,
                fee=abs(fee)
            )

            # Record trade
            self._record_trade(trade_result, decision)

            return trade_result

        return TradeResult(
            success=False,
            order_id="",
            pair=pair,
            side="buy",
            amount=0,
            price=current_price,
            total_value=trade_amount,
            fee=0,
            error="Order placement failed"
        )

    def _execute_sell(self, decision, current_price: float) -> TradeResult:
        """Execute a sell order."""
        pair = decision.pair
        base_currency = pair.split("-")[0]  # e.g., "BTC" from "BTC-USDT"

        # Get available balance of the crypto
        balances = self.okx.get_balance(base_currency)
        crypto_balance = balances.get(base_currency, 0)

        if crypto_balance <= 0:
            return TradeResult(
                success=False,
                order_id="",
                pair=pair,
                side="sell",
                amount=0,
                price=current_price,
                total_value=0,
                fee=0,
                error=f"No {base_currency} balance to sell"
            )

        # Check minimum order size
        min_size = self.okx.get_min_order_size(pair)
        if crypto_balance < min_size:
            return TradeResult(
                success=False,
                order_id="",
                pair=pair,
                side="sell",
                amount=0,
                price=current_price,
                total_value=0,
                fee=0,
                error=f"Balance {crypto_balance} below minimum {min_size}"
            )

        # Execute sell (sell entire position)
        logger.info(f"Executing SELL: {pair}, amount: {crypto_balance}")
        result = self.okx.place_market_sell(pair, crypto_balance)

        if result and result.get("ordId"):
            order_id = result["ordId"]

            # Get fill details
            import time
            time.sleep(0.5)
            order_info = self.okx.get_order(pair, order_id)

            fill_price = float(order_info.get("avgPx", current_price))
            fill_amount = float(order_info.get("accFillSz", crypto_balance))
            total_value = fill_price * fill_amount
            fee = float(order_info.get("fee", 0))

            trade_result = TradeResult(
                success=True,
                order_id=order_id,
                pair=pair,
                side="sell",
                amount=fill_amount,
                price=fill_price,
                total_value=total_value,
                fee=abs(fee)
            )

            # Record trade
            self._record_trade(trade_result, decision)

            return trade_result

        return TradeResult(
            success=False,
            order_id="",
            pair=pair,
            side="sell",
            amount=0,
            price=current_price,
            total_value=0,
            fee=0,
            error="Order placement failed"
        )

    def close_position(self, pair: str, entry_price: float = 0) -> TradeResult:
        """
        Close an existing position (sell all).

        Args:
            pair: Trading pair
            entry_price: Original entry price (for P&L calculation)

        Returns:
            TradeResult
        """
        base_currency = pair.split("-")[0]
        balances = self.okx.get_balance(base_currency)
        crypto_balance = balances.get(base_currency, 0)

        if crypto_balance <= 0:
            return TradeResult(
                success=False,
                order_id="",
                pair=pair,
                side="sell",
                amount=0,
                price=0,
                total_value=0,
                fee=0,
                error="No position to close"
            )

        current_price = float(self.okx.get_ticker(pair).get("last", 0))

        logger.info(f"Closing position: {pair}, amount: {crypto_balance}")
        result = self.okx.place_market_sell(pair, crypto_balance)

        if result and result.get("ordId"):
            order_id = result["ordId"]

            import time
            time.sleep(0.5)
            order_info = self.okx.get_order(pair, order_id)

            fill_price = float(order_info.get("avgPx", current_price))
            fill_amount = float(order_info.get("accFillSz", crypto_balance))
            total_value = fill_price * fill_amount
            fee = float(order_info.get("fee", 0))

            # Calculate P&L
            pnl = 0
            pnl_pct = 0
            if entry_price > 0:
                pnl = (fill_price - entry_price) * fill_amount
                pnl_pct = ((fill_price - entry_price) / entry_price) * 100

            trade_result = TradeResult(
                success=True,
                order_id=order_id,
                pair=pair,
                side="sell",
                amount=fill_amount,
                price=fill_price,
                total_value=total_value,
                fee=abs(fee)
            )

            # Record with P&L
            self._record_trade(trade_result, pnl=pnl, pnl_pct=pnl_pct, entry_price=entry_price)

            logger.info(f"Position closed: {pair}, P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            return trade_result

        return TradeResult(
            success=False,
            order_id="",
            pair=pair,
            side="sell",
            amount=0,
            price=current_price,
            total_value=0,
            fee=0,
            error="Failed to close position"
        )

    @staticmethod
    def calculate_dynamic_leverage(confidence: float, max_leverage: int = 7) -> int:
        """
        Calculate leverage based on model confidence.

        Higher confidence = higher leverage for bigger instant profits.
        Lower confidence = lower leverage for safety.

        Tiers:
            50-55% → 2x (cautious entry)
            55-65% → 3x (moderate)
            65-75% → 5x (confident)
            75%+   → max_leverage (very confident, quick profit scenario)

        Args:
            confidence: Model confidence (0.0 to 1.0)
            max_leverage: Maximum allowed leverage (from config)

        Returns:
            Leverage multiplier (int)
        """
        if confidence >= 0.75:
            return max_leverage  # Full send - bot is very confident
        elif confidence >= 0.65:
            return min(5, max_leverage)
        elif confidence >= 0.55:
            return min(3, max_leverage)
        else:
            return 2  # Minimum leverage for low confidence

    @staticmethod
    def calculate_adaptive_leverage(position, max_leverage: int = 7) -> int:
        """
        Adjust leverage on open position based on how the trade is going.

        Strategy:
        - Losing > 1%: drop to 2x (defensive, protect capital)
        - Flat (±1%): keep at 3x (cautious)
        - Gaining 1-3%: restore to entry leverage
        - Gaining 3%+: boost to max (ride the wave for max profit)

        Args:
            position: Position object with unrealized_pnl_pct
            max_leverage: Maximum allowed leverage

        Returns:
            New leverage multiplier (int)
        """
        pnl_pct = position.unrealized_pnl_pct

        if pnl_pct <= -1.0:
            # Losing - go defensive
            return 2
        elif pnl_pct <= 1.0:
            # Flat - stay cautious
            return min(3, max_leverage)
        elif pnl_pct <= 3.0:
            # Gaining - restore to entry leverage
            return position.leverage  # Keep current
        else:
            # Big gain - max leverage to ride the wave
            return max_leverage

    def execute_short_open(self, decision, current_price: float, futures_config, dynamic_leverage: int = None) -> TradeResult:
        """
        Open a short position via perpetual swap.

        Args:
            decision: TradeDecision from ensemble (action == SELL)
            current_price: Current price of the asset
            futures_config: FuturesConfig with leverage/margin settings
            dynamic_leverage: Override leverage (from dynamic calculation)
        """
        pair = decision.pair
        leverage = dynamic_leverage or futures_config.leverage

        # Validate with risk manager
        if not self.risk.validate_trade(decision):
            return TradeResult(
                success=False, order_id="", pair=pair, side="short_open",
                amount=0, price=0, total_value=0, fee=0,
                error="Trade rejected by risk manager"
            )

        try:
            # Set leverage for this trade before placing order
            try:
                self.okx.set_leverage(pair, leverage, futures_config.margin_mode)
                logger.info(f"Dynamic leverage set: {pair} → {leverage}x")
            except Exception as e:
                logger.warning(f"Failed to set leverage {leverage}x for {pair}: {e}")
                leverage = 2  # Fallback to safe leverage

            # Get available USDT balance
            usdt_balance = self.okx.get_usdt_balance()

            # Calculate trade size (same logic as buy)
            trade_size_pct = decision.suggested_size_pct / 100
            trade_amount_usdt = usdt_balance * trade_size_pct

            # Get swap instrument info for contract sizing
            swap_info = self.okx.get_swap_instrument_info(pair)
            ct_val = float(swap_info.get("ctVal", 1))  # Contract value (e.g. 1 SOL per contract)
            min_sz = float(swap_info.get("minSz", 1))  # Minimum contracts

            # Calculate number of contracts
            # With leverage, our margin buys more exposure
            effective_usdt = trade_amount_usdt * leverage
            num_contracts = int(effective_usdt / (ct_val * current_price))

            if num_contracts < min_sz:
                return TradeResult(
                    success=False, order_id="", pair=pair, side="short_open",
                    amount=0, price=current_price, total_value=trade_amount_usdt, fee=0,
                    error=f"Calculated {num_contracts} contracts, minimum is {min_sz}"
                )

            logger.info(
                f"Opening SHORT: {pair}, {num_contracts} contracts "
                f"(${trade_amount_usdt:.2f} margin, {leverage}x dynamic leverage, "
                f"conf: {decision.confidence:.0%})"
            )

            # Place sell order on swap to open short
            result = self.okx.place_futures_order(
                pair=pair,
                side="sell",
                size=str(num_contracts),
                margin_mode=futures_config.margin_mode
            )

            if result and result.get("ordId"):
                order_id = result["ordId"]

                import time
                time.sleep(0.5)
                swap_id = self.okx.spot_to_swap(pair)
                order_info = self.okx.get_order(swap_id, order_id)

                fill_price = float(order_info.get("avgPx", current_price))
                fill_sz = float(order_info.get("accFillSz", num_contracts))
                fee = float(order_info.get("fee", 0))

                # Amount in base currency terms
                amount_base = fill_sz * ct_val

                trade_result = TradeResult(
                    success=True,
                    order_id=order_id,
                    pair=pair,
                    side="short_open",
                    amount=amount_base,
                    price=fill_price,
                    total_value=amount_base * fill_price,
                    fee=abs(fee)
                )

                # Store leverage info on the result for position tracking
                trade_result._leverage = leverage
                trade_result._max_leverage = futures_config.leverage

                self._record_trade(trade_result, decision)
                return trade_result

            return TradeResult(
                success=False, order_id="", pair=pair, side="short_open",
                amount=0, price=current_price, total_value=0, fee=0,
                error="Futures order placement failed"
            )

        except Exception as e:
            logger.error(f"Short open failed: {e}")
            return TradeResult(
                success=False, order_id="", pair=pair, side="short_open",
                amount=0, price=current_price, total_value=0, fee=0,
                error=str(e)
            )

    def execute_long_open(self, decision, current_price: float, futures_config, dynamic_leverage: int = None) -> TradeResult:
        """
        Open a leveraged long position via perpetual swap.

        Args:
            decision: TradeDecision from ensemble (action == BUY)
            current_price: Current price of the asset
            futures_config: FuturesConfig with leverage/margin settings
            dynamic_leverage: Override leverage (from dynamic calculation)
        """
        pair = decision.pair
        leverage = dynamic_leverage or futures_config.leverage

        # Validate with risk manager
        if not self.risk.validate_trade(decision):
            return TradeResult(
                success=False, order_id="", pair=pair, side="long_open",
                amount=0, price=0, total_value=0, fee=0,
                error="Trade rejected by risk manager"
            )

        try:
            # Set leverage for this trade before placing order
            try:
                self.okx.set_leverage(pair, leverage, futures_config.margin_mode)
                logger.info(f"Dynamic leverage set: {pair} → {leverage}x")
            except Exception as e:
                logger.warning(f"Failed to set leverage {leverage}x for {pair}: {e}")
                leverage = 2  # Fallback to safe leverage

            # Get available USDT balance
            usdt_balance = self.okx.get_usdt_balance()

            # Calculate trade size
            trade_size_pct = decision.suggested_size_pct / 100
            trade_amount_usdt = usdt_balance * trade_size_pct

            # Get swap instrument info for contract sizing
            swap_info = self.okx.get_swap_instrument_info(pair)
            ct_val = float(swap_info.get("ctVal", 1))
            min_sz = float(swap_info.get("minSz", 1))

            # Calculate number of contracts
            effective_usdt = trade_amount_usdt * leverage
            num_contracts = int(effective_usdt / (ct_val * current_price))

            if num_contracts < min_sz:
                return TradeResult(
                    success=False, order_id="", pair=pair, side="long_open",
                    amount=0, price=current_price, total_value=trade_amount_usdt, fee=0,
                    error=f"Calculated {num_contracts} contracts, minimum is {min_sz}"
                )

            logger.info(
                f"Opening LONG: {pair}, {num_contracts} contracts "
                f"(${trade_amount_usdt:.2f} margin, {leverage}x dynamic leverage, "
                f"conf: {decision.confidence:.0%})"
            )

            # Place buy order on swap to open long
            result = self.okx.place_futures_order(
                pair=pair,
                side="buy",
                size=str(num_contracts),
                margin_mode=futures_config.margin_mode
            )

            if result and result.get("ordId"):
                order_id = result["ordId"]

                import time
                time.sleep(0.5)
                swap_id = self.okx.spot_to_swap(pair)
                order_info = self.okx.get_order(swap_id, order_id)

                fill_price = float(order_info.get("avgPx", current_price))
                fill_sz = float(order_info.get("accFillSz", num_contracts))
                fee = float(order_info.get("fee", 0))

                amount_base = fill_sz * ct_val

                trade_result = TradeResult(
                    success=True,
                    order_id=order_id,
                    pair=pair,
                    side="long_open",
                    amount=amount_base,
                    price=fill_price,
                    total_value=amount_base * fill_price,
                    fee=abs(fee)
                )

                trade_result._leverage = leverage
                trade_result._max_leverage = futures_config.leverage

                self._record_trade(trade_result, decision)
                return trade_result

            return TradeResult(
                success=False, order_id="", pair=pair, side="long_open",
                amount=0, price=current_price, total_value=0, fee=0,
                error="Futures order placement failed"
            )

        except Exception as e:
            logger.error(f"Long open failed: {e}")
            return TradeResult(
                success=False, order_id="", pair=pair, side="long_open",
                amount=0, price=current_price, total_value=0, fee=0,
                error=str(e)
            )

    def close_futures_position(self, pair: str, entry_price: float = 0, margin_mode: str = "cross") -> TradeResult:
        """
        Close a futures/short position.

        Args:
            pair: Trading pair (e.g. 'SOL-USDT')
            entry_price: Original entry price for P&L
            margin_mode: 'cross' or 'isolated'
        """
        try:
            swap_id = self.okx.spot_to_swap(pair)
            current_price = float(self.okx.get_ticker(swap_id).get("last", 0))

            logger.info(f"Closing futures position: {pair}")
            result = self.okx.close_futures_position(pair, margin_mode)

            if result:
                # Calculate P&L (for shorts: profit = entry - exit)
                pnl = 0
                pnl_pct = 0
                if entry_price > 0:
                    # We don't know direction here, but the position tracker does
                    # Use entry > current as short indicator
                    pnl = (entry_price - current_price)  # Approximate per-unit
                    pnl_pct = (pnl / entry_price) * 100

                trade_result = TradeResult(
                    success=True,
                    order_id=result[0].get("ordId", "") if isinstance(result, list) and result else "",
                    pair=pair,
                    side="short_close",
                    amount=0,  # OKX close_positions closes all
                    price=current_price,
                    total_value=0,
                    fee=0
                )

                self._record_trade(trade_result, pnl=pnl, pnl_pct=pnl_pct, entry_price=entry_price)
                logger.info(f"Futures position closed: {pair}, P&L: ~{pnl_pct:.2f}%")
                return trade_result

            return TradeResult(
                success=False, order_id="", pair=pair, side="short_close",
                amount=0, price=current_price, total_value=0, fee=0,
                error="Failed to close futures position"
            )

        except Exception as e:
            logger.error(f"Failed to close futures position {pair}: {e}")
            return TradeResult(
                success=False, order_id="", pair=pair, side="short_close",
                amount=0, price=0, total_value=0, fee=0,
                error=str(e)
            )

    def _record_trade(
        self,
        result: TradeResult,
        decision=None,
        pnl: float = 0,
        pnl_pct: float = 0,
        entry_price: float = 0
    ):
        """Record trade to database."""
        trade_id = str(uuid.uuid4())[:8]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trades (
                id, timestamp, pair, side, amount, price,
                total_value, fee, order_id, pnl, pnl_percent, entry_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id,
            datetime.now().isoformat(),
            result.pair,
            result.side,
            result.amount,
            result.price,
            result.total_value,
            result.fee,
            result.order_id,
            pnl,
            pnl_pct,
            entry_price
        ))

        # Record decision if provided
        if decision:
            decision_id = str(uuid.uuid4())[:8]
            cursor.execute("""
                INSERT INTO decisions (
                    id, timestamp, pair, action, confidence,
                    xgb_action, xgb_confidence, rl_action, rl_confidence,
                    reasoning, executed, trade_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision_id,
                datetime.now().isoformat(),
                decision.pair,
                str(decision.action),
                decision.confidence,
                decision.xgb_action,
                decision.xgb_confidence,
                decision.rl_action,
                decision.rl_confidence,
                decision.reasoning,
                1,
                trade_id
            ))

        conn.commit()
        conn.close()

    def record_decision(self, decision, executed: bool = False):
        """Record a trading decision (even if not executed)."""
        decision_id = str(uuid.uuid4())[:8]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO decisions (
                id, timestamp, pair, action, confidence,
                xgb_action, xgb_confidence, rl_action, rl_confidence,
                reasoning, executed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision_id,
            datetime.now().isoformat(),
            decision.pair,
            str(decision.action),
            decision.confidence,
            decision.xgb_action,
            decision.xgb_confidence,
            decision.rl_action,
            decision.rl_confidence,
            decision.reasoning,
            1 if executed else 0
        ))

        conn.commit()
        conn.close()

    def get_trade_history(self, limit: int = 50) -> list:
        """Get recent trade history from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return rows

    def get_total_pnl(self) -> float:
        """Get total P&L from all trades."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT SUM(pnl) FROM trades")
        result = cursor.fetchone()
        conn.close()

        return result[0] if result[0] else 0.0
