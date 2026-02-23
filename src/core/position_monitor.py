"""
Position Monitor
================
Checks open positions for stop-loss, trailing-stop, take-profit,
break-even guard, and adaptive leverage each cycle.
"""

import logging
from datetime import datetime, timedelta

from src.notifications import notify_stoploss, notify_takeprofit, notify_close

logger = logging.getLogger(__name__)


class PositionMonitor:
    """Monitors and manages open positions each cycle."""

    def __init__(self, positions, risk, executor, collector, features, okx, config, stop_loss_mgr):
        self.positions = positions
        self.risk = risk
        self.executor = executor
        self.collector = collector
        self.features = features
        self.okx = okx
        self.config = config
        self.sl = stop_loss_mgr

    def check_all(self, pair_cooldowns: dict) -> None:
        """Run all position checks. Modifies pair_cooldowns in-place."""
        for position in self.positions.get_all_positions():
            pair = position.pair
            mode_label = f" ({position.mode} {position.side})" if position.mode == "futures" else ""
            lev_label = f" {position.leverage}x" if position.mode == "futures" else ""
            logger.info(
                f"  {pair}{mode_label}{lev_label}: ${position.current_price:.2f} | "
                f"P&L: {position.unrealized_pnl_pct:+.2f}%"
            )

            # --- Stop-loss ---
            atr_pct = self._get_atr_pct(pair, position.current_price)

            if self.risk.check_stop_loss(position, atr_pct=atr_pct):
                logger.warning(f"STOP-LOSS: {pair}{mode_label}")
                notify_stoploss(pair, position.side or "long", position.unrealized_pnl_pct)
                self._close(position)
                pair_cooldowns[pair] = datetime.now() + timedelta(hours=4)
                logger.info(f"[{pair}] 4h cooldown activated until {pair_cooldowns[pair].strftime('%H:%M')}")
                continue

            # --- Trailing stop ---
            if self.risk.check_trailing_stop(position):
                logger.info(f"TRAILING STOP: {pair}{mode_label} - locking profit")
                notify_close(pair, position.side or "long", position.unrealized_pnl_pct, 0, "Trailing stop")
                self._close(position)
                continue

            # --- Take-profit ---
            if self.risk.check_take_profit(position):
                logger.info(f"TAKE-PROFIT: {pair}{mode_label}")
                notify_takeprofit(pair, position.side or "long", position.unrealized_pnl_pct)
                self._close(position)
                continue

            # --- Break-even guard ---
            if (position.mode == "futures"
                    and position.stop_loss_algo_id
                    and position.unrealized_pnl_pct >= 1.5
                    and not getattr(position, '_breakeven_set', False)):
                if self.sl.move_to_breakeven(pair, position):
                    notify_close(pair, position.side, position.unrealized_pnl_pct, 0,
                                 "Stop moved to break-even (risk-free trade)")

            # --- Adaptive leverage ---
            if (position.mode == "futures"
                    and self.config.futures.enabled
                    and not position.stop_loss_algo_id):
                new_lev = self.executor.calculate_adaptive_leverage(
                    position, self.config.futures.leverage
                )
                if new_lev != position.leverage:
                    try:
                        self.okx.set_leverage(pair, new_lev, self.config.futures.margin_mode)
                        old_lev = position.leverage
                        position.leverage = new_lev
                        logger.info(
                            f"LEVERAGE ADJUSTED: {pair} {old_lev}x -> {new_lev}x "
                            f"(P&L: {position.unrealized_pnl_pct:+.1f}%)"
                        )
                    except Exception as e:
                        logger.warning(f"Leverage adjustment failed for {pair}: {e}")

    def _close(self, pos):
        """Close a position, cancelling its hard stop-loss first."""
        if pos.stop_loss_algo_id:
            self.sl.cancel(pos.pair, pos.stop_loss_algo_id)
        if pos.mode == "futures":
            result = self.executor.close_futures_position(
                pos.pair, pos.entry_price, self.config.futures.margin_mode, pos.amount
            )
            if result.success:
                self.positions.remove_futures_position(pos.pair)
                self.risk.record_trade()
        else:
            result = self.executor.close_position(pos.pair, pos.entry_price)
            if result.success:
                self.positions.remove_position(pos.pair)
                self.risk.record_trade()
        return result

    def _get_atr_pct(self, pair: str, current_price: float) -> float:
        """Compute ATR% for adaptive stop-loss."""
        try:
            candles = self.collector.get_candles(pair, "1h", 30)
            if len(candles) >= 14:
                feat = self.features.calculate_features(candles)
                if not feat.empty and "atr_14" in feat.columns:
                    atr_val = float(feat["atr_14"].iloc[-1])
                    if current_price > 0:
                        return (atr_val / current_price) * 100
        except Exception:
            pass
        return 0.0
