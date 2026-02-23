"""
Pair Analyzer
=============
Scans trading pairs, computes features, gets ensemble decisions,
applies filters, and executes trades.
"""

import logging
from typing import List, Dict, Any

from src.models.ensemble import Action
from src.notifications import notify_trade

logger = logging.getLogger(__name__)


class PairAnalyzer:
    """Scans pairs for trade signals and executes them."""

    def __init__(self, okx, collector, features, positions, ensemble,
                 trend_filter, regime_detector, funding_filter, correlation_filter,
                 executor, risk, pred_tracker, config, stop_loss_mgr):
        self.okx = okx
        self.collector = collector
        self.features = features
        self.positions = positions
        self.ensemble = ensemble
        self.trend_filter = trend_filter
        self.regime_detector = regime_detector
        self.funding_filter = funding_filter
        self.correlation_filter = correlation_filter
        self.executor = executor
        self.risk = risk
        self.pred_tracker = pred_tracker
        self.config = config
        self.sl = stop_loss_mgr

    def scan_pairs(self, pair_cooldowns: dict) -> List[Dict[str, Any]]:
        """
        Scan all trading pairs for signals. Execute trades when slots available.
        Returns list of blocked high-conviction candidates for succession.
        """
        from datetime import datetime

        succession_candidates = []
        now = datetime.now()

        self.correlation_filter.clear_cache()
        self.funding_filter.clear_cache()

        for pair in self.config.trading.trading_pairs:
            try:
                if self.positions.has_position(pair):
                    continue

                if pair in pair_cooldowns:
                    remaining = (pair_cooldowns[pair] - now).total_seconds() / 60
                    logger.info(f"[{pair}] Cooldown active ({remaining:.0f}min remaining)")
                    continue

                decision, price, volatility = self._analyze_pair(pair)
                if decision is None or decision.action == Action.HOLD:
                    continue

                # Trend filter
                trend = self.trend_filter.get_trend(
                    self._last_features_row
                )
                if not self.trend_filter.filter_decision(trend, decision.action):
                    action_name = "BUY" if decision.action == 1 else "SELL"
                    logger.info(f"[{pair}] FILTERED: {action_name} blocked by {trend.value.upper()} trend")
                    continue

                # Confidence check
                if decision.confidence < self.config.trading.min_confidence_to_trade:
                    logger.info(
                        f"[{pair}] Below confidence: {decision.confidence:.2f} < "
                        f"{self.config.trading.min_confidence_to_trade:.2f}"
                    )
                    continue

                # Max positions → queue for succession
                current_count = len(self.positions.get_all_positions())
                if current_count >= self.config.trading.max_open_positions:
                    if decision.confidence >= 0.75:
                        dyn_lev = self.executor.calculate_dynamic_leverage(
                            decision.confidence, self.config.futures.leverage
                        )
                        succession_candidates.append({
                            "pair": pair,
                            "decision": decision,
                            "price": price,
                            "dyn_leverage": dyn_lev,
                            "action": decision.action,
                        })
                        logger.info(
                            f"[{pair}] SUCCESSION CANDIDATE: "
                            f"{'BUY' if decision.action == Action.BUY else 'SELL'} "
                            f"@ {decision.confidence:.2f} (queued)"
                        )
                    continue

                # Execute trade
                self._execute_trade(pair, decision, price)

            except Exception as e:
                logger.error(f"Error on {pair}: {e}")

        return succession_candidates

    def get_pair_signal(self, pair: str):
        """Get brain signal for a single pair (used by succession scan)."""
        return self._analyze_pair(pair)

    def _analyze_pair(self, pair: str):
        """Fetch data, compute features, get ensemble decision. Returns (decision, price, volatility) or (None, 0, 0)."""
        candles_1h = self.collector.get_candles(pair, "1h", 300)
        if len(candles_1h) < 50:
            return None, 0, 0

        try:
            candles_4h = self.collector.get_candles(pair, "4h", 100)
            candles_1d = self.collector.get_candles(pair, "1d", 100)
        except Exception:
            candles_4h, candles_1d = None, None

        df_features = self.features.calculate_multi_tf_features(candles_1h, candles_4h, candles_1d)
        if df_features.empty:
            return None, 0, 0

        # Inject funding rate
        df_features = self._inject_funding_rate(pair, df_features)

        latest = df_features.iloc[[-1]]
        latest_row = df_features.iloc[-1]
        price = float(candles_1h["close"].iloc[-1])

        # Store for filter access
        self._last_features_row = latest_row
        self._last_features_df = df_features

        regime_params = self.regime_detector.detect_and_get_params(latest_row, df_features)
        logger.info(f"[{pair}] Regime: {regime_params.name} - {regime_params.description}")

        trend_score = self.trend_filter._calculate_trend_score(latest_row)
        volatility = float(latest_row.get("volatility_14", 0)) if "volatility_14" in latest_row.index else 0.0

        portfolio_state = {
            "balance": self.okx.get_usdt_balance(),
            "position": 0,
            "entry_price": 0,
            "current_price": price,
        }

        decision = self.ensemble.get_decision(
            latest, portfolio_state, pair,
            regime=regime_params.name,
            volatility=volatility,
            trend_strength=trend_score,
        )

        # Record prediction
        if decision.action != Action.HOLD:
            try:
                self.pred_tracker.record_prediction(
                    pair=pair,
                    xgb_action=decision.xgb_action, xgb_conf=decision.xgb_confidence,
                    rl_action=decision.rl_action, rl_conf=decision.rl_confidence,
                    lstm_action=decision.lstm_action, lstm_conf=decision.lstm_confidence,
                    ensemble_action=decision.action, ensemble_conf=decision.confidence,
                    price=price, volatility=volatility,
                    regime=regime_params.name, trend_strength=trend_score,
                )
            except Exception:
                pass

        return decision, price, volatility

    def _inject_funding_rate(self, pair, df):
        """Add funding rate columns to feature DataFrame."""
        df = df.copy()
        try:
            fr_data = self.okx.get_funding_rate(pair)
            if fr_data and "fundingRate" in fr_data:
                fr = float(fr_data["fundingRate"])
                df["funding_rate"] = fr * 100
                df["funding_contrarian"] = -fr * 1000
                return df
        except Exception:
            pass
        df["funding_rate"] = 0.0
        df["funding_contrarian"] = 0.0
        return df

    def _execute_trade(self, pair: str, decision, price: float):
        """Open a futures position based on the decision."""
        if not (self.config.futures.enabled and pair in self.config.futures.futures_pairs):
            return
        if self.positions.has_position(pair, mode="futures"):
            return

        dyn_leverage = self.executor.calculate_dynamic_leverage(
            decision.confidence, self.config.futures.leverage
        )

        if decision.action == Action.BUY:
            side = "long"
            logger.info(f"LONG {pair} (conf: {decision.confidence:.2f}, leverage: {dyn_leverage}x)")
            result = self.executor.execute_long_open(
                decision, price, self.config.futures, dynamic_leverage=dyn_leverage
            )
        elif decision.action == Action.SELL:
            side = "short"
            logger.info(f"SHORT {pair} (conf: {decision.confidence:.2f}, leverage: {dyn_leverage}x)")
            result = self.executor.execute_short_open(
                decision, price, self.config.futures, dynamic_leverage=dyn_leverage
            )
        else:
            return

        if result.success:
            self.positions.add_position(
                pair, result.amount, result.price,
                side=side, mode="futures",
                leverage=dyn_leverage,
                max_leverage=self.config.futures.leverage,
                entry_confidence=decision.confidence,
            )
            self.risk.record_trade()
            self.sl.place(pair, side, result.price)
            logger.info(f"{'Longed' if side == 'long' else 'Shorted'} {result.amount} {pair} @ ${result.price} ({dyn_leverage}x)")
            notify_trade(pair, "BUY" if side == "long" else "SELL", side, dyn_leverage,
                         decision.confidence, result.price, result.amount * result.price)
