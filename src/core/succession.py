"""
Succession Engine
=================
Auto-rotates weak positions for high-conviction signals when
all position slots are full.

Trigger: new signal > 0.75 AND weakest position alignment < 0.25
Safety: 1 per cycle, 2h cooldown on closed pair, Telegram alert
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.models.ensemble import Action
from src.notifications import notify_alert

logger = logging.getLogger(__name__)


class SuccessionEngine:
    """Evaluates and executes position rotations."""

    def __init__(self, okx, positions, ensemble, collector, features,
                 trend_filter, regime_detector, executor, risk,
                 config, stop_loss_mgr):
        self.okx = okx
        self.positions = positions
        self.ensemble = ensemble
        self.collector = collector
        self.features = features
        self.trend_filter = trend_filter
        self.regime_detector = regime_detector
        self.executor = executor
        self.risk = risk
        self.config = config
        self.sl = stop_loss_mgr

    def evaluate(self, candidates: List[Dict[str, Any]], pair_cooldowns: dict) -> bool:
        """
        Check if a succession swap should happen.
        Returns True if a rotation was executed.
        """
        if not candidates:
            return False

        best = max(candidates, key=lambda c: c["decision"].confidence)
        logger.info(
            f"SUCCESSION CHECK: {len(candidates)} candidate(s), "
            f"best={best['pair']} @ {best['decision'].confidence:.2f}"
        )

        convictions = self._scan_positions()
        if not convictions:
            return False

        weakest = min(convictions, key=lambda c: c["alignment"])

        if weakest["alignment"] >= 0.25:
            logger.info(
                f"Succession skipped: weakest={weakest['position'].pair} "
                f"alignment={weakest['alignment']:+.2f} (> 0.25 threshold)"
            )
            return False

        return self._execute_rotation(best, weakest, pair_cooldowns)

    def _scan_positions(self) -> List[Dict[str, Any]]:
        """Re-scan all open positions through the brain for alignment scores."""
        convictions = []

        for pos in self.positions.get_all_positions():
            if pos.mode != "futures":
                continue
            try:
                decision, price, vol = self._get_brain_signal(pos.pair)
                if decision is None:
                    continue

                alignment = self._calc_alignment(pos.side, decision)

                convictions.append({
                    "position": pos,
                    "alignment": alignment,
                    "brain_action": decision.action,
                    "brain_conf": decision.confidence,
                    "pnl_pct": pos.unrealized_pnl_pct,
                })
                logger.info(
                    f"  {pos.pair} {pos.side}: alignment={alignment:+.2f} "
                    f"(brain={'BUY' if decision.action == 1 else 'SELL' if decision.action == -1 else 'HOLD'}"
                    f"@{decision.confidence:.2f}), PnL={pos.unrealized_pnl_pct:+.2f}%"
                )
            except Exception as e:
                logger.debug(f"Succession scan error for {pos.pair}: {e}")

        return convictions

    def _get_brain_signal(self, pair: str):
        """Get the brain's current view on a pair."""
        candles_1h = self.collector.get_candles(pair, "1h", 300)
        if len(candles_1h) < 50:
            return None, 0, 0

        try:
            candles_4h = self.collector.get_candles(pair, "4h", 100)
            candles_1d = self.collector.get_candles(pair, "1d", 100)
        except Exception:
            candles_4h, candles_1d = None, None

        df = self.features.calculate_multi_tf_features(candles_1h, candles_4h, candles_1d)
        if df.empty:
            return None, 0, 0

        # Inject funding rate
        df = df.copy()
        try:
            fr_data = self.okx.get_funding_rate(pair)
            if fr_data and "fundingRate" in fr_data:
                fr = float(fr_data["fundingRate"])
                df["funding_rate"] = fr * 100
                df["funding_contrarian"] = -fr * 1000
            else:
                df["funding_rate"] = 0.0
                df["funding_contrarian"] = 0.0
        except Exception:
            df["funding_rate"] = 0.0
            df["funding_contrarian"] = 0.0

        latest = df.iloc[[-1]]
        latest_row = df.iloc[-1]
        price = float(candles_1h["close"].iloc[-1])
        vol = float(latest_row.get("volatility_14", 0)) if "volatility_14" in latest_row.index else 0.0

        portfolio_state = {
            "balance": self.okx.get_usdt_balance(),
            "position": 0, "entry_price": 0, "current_price": price,
        }
        trend_score = self.trend_filter._calculate_trend_score(latest_row)

        decision = self.ensemble.get_decision(
            latest, portfolio_state, pair,
            regime=self.regime_detector.detect_and_get_params(latest_row, df).name,
            volatility=vol,
            trend_strength=trend_score,
        )
        return decision, price, vol

    @staticmethod
    def _calc_alignment(side: str, decision) -> float:
        """Calculate how aligned the brain is with the position direction."""
        if side == "short":
            if decision.action == Action.SELL:
                return decision.confidence
            elif decision.action == Action.BUY:
                return -decision.confidence
        elif side == "long":
            if decision.action == Action.BUY:
                return decision.confidence
            elif decision.action == Action.SELL:
                return -decision.confidence
        return 0.0

    def _execute_rotation(self, best: dict, weakest: dict, pair_cooldowns: dict) -> bool:
        """Close weakest position, open best candidate."""
        w_pos = weakest["position"]
        logger.info(
            f"SUCCESSION TRIGGERED: Close {w_pos.pair} {w_pos.side} "
            f"(alignment={weakest['alignment']:+.2f}) → Open {best['pair']} "
            f"(conf={best['decision'].confidence:.2f})"
        )

        # Step 1: Close weakest
        if w_pos.stop_loss_algo_id:
            self.sl.cancel(w_pos.pair, w_pos.stop_loss_algo_id)

        close_result = self.executor.close_futures_position(
            w_pos.pair, w_pos.entry_price,
            self.config.futures.margin_mode, w_pos.amount,
        )
        if not close_result.success:
            logger.warning(f"Succession close failed for {w_pos.pair}")
            return False

        self.positions.remove_futures_position(w_pos.pair)
        self.risk.record_trade()
        close_pnl = w_pos.unrealized_pnl_pct
        pair_cooldowns[w_pos.pair] = datetime.now() + timedelta(hours=2)

        # Step 2: Open new position
        time.sleep(1)  # Brief pause for margin to free up
        b = best["decision"]
        b_price = best["price"]
        b_lev = best["dyn_leverage"]

        if b.action == Action.BUY:
            result = self.executor.execute_long_open(b, b_price, self.config.futures, dynamic_leverage=b_lev)
            side = "long"
        else:
            result = self.executor.execute_short_open(b, b_price, self.config.futures, dynamic_leverage=b_lev)
            side = "short"

        if result.success:
            self.positions.add_position(
                best["pair"], result.amount, result.price,
                side=side, mode="futures",
                leverage=b_lev, max_leverage=self.config.futures.leverage,
                entry_confidence=b.confidence,
            )
            self.risk.record_trade()
            self.sl.place(best["pair"], side, result.price)

            logger.info(f"SUCCESSION COMPLETE: {w_pos.pair}→{best['pair']} ({side} {b_lev}x @ ${result.price})")
            notify_alert(
                "SUCCESSION RULE",
                f"Rotated: {w_pos.pair} {w_pos.side} (alignment {weakest['alignment']:+.2f}, PnL {close_pnl:+.2f}%)\n"
                f"→ {best['pair']} {side} {b_lev}x @ ${result.price:.4f}\n"
                f"Conviction: {b.confidence:.2f} | Hard stop placed",
            )
            return True
        else:
            logger.warning(f"Succession open failed for {best['pair']}")
            notify_alert(
                "SUCCESSION PARTIAL",
                f"Closed {w_pos.pair} {w_pos.side} (PnL {close_pnl:+.2f}%) "
                f"but failed to open {best['pair']}. Slot freed.",
            )
            return False
