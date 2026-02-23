"""
Hard Stop-Loss Manager
======================
Places and cancels OKX algo orders for exchange-side stop-losses.
Positions stay protected even if bot/VPS crashes.
"""

import logging

logger = logging.getLogger(__name__)


class StopLossManager:
    """Manages OKX algo-order stop-losses for open positions."""

    def __init__(self, okx, positions, config):
        self.okx = okx
        self.positions = positions
        self.config = config

    def place(self, pair: str, side: str, entry_price: float, stop_pct: float = None) -> str:
        """Place a hard stop-loss. Returns algo_id or empty string."""
        try:
            stop_pct = stop_pct or self.config.trading.stop_loss_percent
            if side == "short":
                trigger_price = round(entry_price * (1 + stop_pct / 100), 6)
                close_side = "buy"
            else:
                trigger_price = round(entry_price * (1 - stop_pct / 100), 6)
                close_side = "sell"

            result = self.okx.place_stop_loss_order(
                pair=pair,
                side=close_side,
                trigger_price=str(trigger_price),
                margin_mode=self.config.futures.margin_mode,
            )
            algo_id = result.get("algoId", "")
            if algo_id:
                pos = self.positions.get_position(pair, mode="futures")
                if not pos:
                    pos = self.positions.get_position(pair, mode="spot")
                if pos:
                    pos.stop_loss_algo_id = algo_id
                logger.info(f"HARD STOP-LOSS set: {pair} {side} @ ${trigger_price} (algo: {algo_id})")
            return algo_id
        except Exception as e:
            logger.error(f"Failed to place hard stop-loss for {pair}: {e}")
            return ""

    def cancel(self, pair: str, algo_id: str):
        """Cancel a hard stop-loss when position is closed by the bot."""
        if not algo_id:
            return
        try:
            self.okx.cancel_algo_order(pair, algo_id)
            logger.info(f"Hard stop-loss cancelled: {pair} (algo: {algo_id})")
        except Exception as e:
            logger.warning(f"Failed to cancel stop-loss for {pair}: {e}")

    def move_to_breakeven(self, pair: str, position) -> bool:
        """Move stop-loss to entry price (break-even). Returns True on success."""
        try:
            self.cancel(pair, position.stop_loss_algo_id)
            close_side = "buy" if position.side == "short" else "sell"
            result = self.okx.place_stop_loss_order(
                pair=pair,
                side=close_side,
                trigger_price=str(position.entry_price),
                margin_mode=self.config.futures.margin_mode,
            )
            new_algo_id = result.get("algoId", "")
            if new_algo_id:
                position.stop_loss_algo_id = new_algo_id
                position._breakeven_set = True
                logger.info(
                    f"BREAK-EVEN GUARD: {pair} stop moved to entry ${position.entry_price:.4f} "
                    f"(risk-free at +{position.unrealized_pnl_pct:.1f}%)"
                )
                return True
        except Exception as e:
            logger.warning(f"Break-even guard failed for {pair}: {e}")
        return False

    def sync_on_startup(self):
        """Attach existing algo orders to positions, place missing ones."""
        try:
            existing_algos = self.okx.get_pending_algo_orders()
            for algo in existing_algos:
                inst_id = algo.get("instId", "").replace("-SWAP", "")
                algo_id = algo.get("algoId", "")
                if inst_id and algo_id:
                    pos = self.positions.get_position(inst_id, mode="futures")
                    if pos:
                        pos.stop_loss_algo_id = algo_id
                        logger.info(f"Attached existing stop-loss: {inst_id} (algo: {algo_id})")
        except Exception as e:
            logger.warning(f"Failed to check existing algo orders: {e}")

        for pos in self.positions.get_all_positions():
            if pos.mode == "futures" and not pos.stop_loss_algo_id:
                algo_id = self.place(pos.pair, pos.side, pos.entry_price)
                if algo_id:
                    logger.info(f"Startup protection: {pos.pair} hard stop-loss placed")
