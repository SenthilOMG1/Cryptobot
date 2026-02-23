"""
Autonomous AI Trading Agent V2 - Core Trading Logic
====================================================
Integrates all V2 components:
- TrendFilter: Block counter-trend trades
- MarketRegimeDetector: Adjust parameters per regime
- MetaLearnerEnsemble: 3-model voting (XGB + RL + LSTM)
- CorrelationFilter: Block correlated same-direction positions
- FundingRateFilter: Contrarian funding rate signals
"""

import os
import sys
import time
import logging
from datetime import datetime

from src.notifications import notify_trade, notify_stoploss, notify_takeprofit, notify_close, notify_alert

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_bot(status_dict):
    """Run the trading bot. Updates status_dict for dashboard."""

    logger.info("=" * 50)
    logger.info("TRADING BOT V2 STARTING")
    logger.info("=" * 50)

    try:
        # Import components
        from src.config import get_config
        from src.security.vault import SecureVault
        from src.trading.okx_client import SecureOKXClient
        from src.trading.executor import TradeExecutor
        from src.trading.positions import PositionTracker
        from src.data.collector import DataCollector
        from src.data.features import FeatureEngine
        from src.models.xgboost_model import XGBoostPredictor
        from src.models.rl_agent import RLTradingAgent
        from src.models.lstm_model import LSTMPredictor
        from src.models.ensemble import MetaLearnerEnsemble, Action
        from src.models.trend_filter import TrendFilter, Trend
        from src.models.regime_detector import MarketRegimeDetector, MarketRegime
        from src.models.funding_filter import FundingRateFilter
        from src.risk.manager import RiskManager, RiskLimits
        from src.risk.correlation_filter import CorrelationFilter
        from src.autonomous.watchdog import Watchdog
        from src.autonomous.retrainer import AutoRetrainer
        from src.autonomous.health import HealthMonitor
        from src.intelligence import PredictionTracker, EnsembleEvaluator, AdaptiveTriggerEngine

        config = get_config()
        logger.info("Config loaded")

        status_dict["state"] = "connecting"
        status_dict["message"] = "Connecting to OKX..."

        vault = SecureVault()
        okx = SecureOKXClient(vault, demo_mode=config.exchange.demo_mode)

        if not okx.test_connection():
            status_dict["state"] = "error"
            status_dict["message"] = "Failed to connect to OKX - check API keys"
            logger.error("OKX connection failed!")
            while True:
                time.sleep(60)

        logger.info("OKX connected!")

        # ==================== FUTURES SETUP ====================
        if config.futures.enabled:
            logger.info("Futures trading ENABLED - configuring...")
            try:
                okx.set_position_mode("net_mode")
                logger.info("Position mode set to net_mode")
            except Exception as e:
                if "59000" in str(e):
                    logger.info("Position mode already set (open positions exist) - OK")
                else:
                    logger.error(f"Position mode setup failed: {e}")
            leverage_ok = 0
            for pair in config.futures.futures_pairs:
                try:
                    okx.set_leverage(pair, 2, config.futures.margin_mode)
                    leverage_ok += 1
                except Exception as e:
                    logger.debug(f"Leverage init skipped for {pair}: {e}")
            logger.info(f"Futures initialized: {leverage_ok}/{len(config.futures.futures_pairs)} pairs set, max {config.futures.leverage}x dynamic")
        else:
            logger.info("Futures trading disabled (FUTURES_ENABLED=false)")

        status_dict["state"] = "initializing"
        status_dict["message"] = "Loading ML models..."

        collector = DataCollector(okx)
        features = FeatureEngine()
        positions = PositionTracker(okx)
        positions.sync_from_exchange()
        if config.futures.enabled:
            positions.sync_futures_from_exchange(config.futures.futures_pairs)

        # Load ML models
        xgb_model = XGBoostPredictor(model_path="models/xgboost_model.json")
        rl_agent = RLTradingAgent(model_path="models/rl_agent.zip")
        lstm_model = LSTMPredictor(model_path="models/lstm_model.pt")

        if not xgb_model.is_trained or not rl_agent.is_trained:
            status_dict["message"] = "Training ML models (first run)..."
            logger.info("Training models...")
            initial_training(collector, features, xgb_model, rl_agent, config.trading.trading_pairs)

        # Initialize V2 components
        ensemble = MetaLearnerEnsemble(
            xgb_model, rl_agent, lstm_model,
            min_confidence=config.trading.min_confidence_to_trade
        )
        trend_filter = TrendFilter(threshold=0.3)
        regime_detector = MarketRegimeDetector()
        funding_filter = FundingRateFilter(okx)
        correlation_filter = CorrelationFilter(collector)

        logger.info(f"V2 Components loaded:")
        logger.info(f"  - MetaLearnerEnsemble (3-model: XGB + RL + LSTM)")
        logger.info(f"  - TrendFilter (block counter-trend trades)")
        logger.info(f"  - MarketRegimeDetector (adaptive parameters)")
        logger.info(f"  - FundingRateFilter (contrarian signals)")
        logger.info(f"  - CorrelationFilter (prevent correlated positions)")
        logger.info(f"  - LSTM model: {'loaded' if lstm_model.is_trained else 'not trained yet'}")

        risk_limits = RiskLimits(
            max_position_pct=config.trading.max_position_percent,
            stop_loss_pct=config.trading.stop_loss_percent,
            take_profit_pct=config.trading.take_profit_percent,
            trailing_stop_pct=config.trading.trailing_stop_percent,
            daily_loss_limit_pct=config.trading.daily_loss_limit_percent,
            max_open_positions=config.trading.max_open_positions,
            min_confidence=config.trading.min_confidence_to_trade
        )
        risk = RiskManager(positions, risk_limits)
        executor = TradeExecutor(okx, risk)
        watchdog = Watchdog(max_retries=10, retry_delay=60)
        retrainer = AutoRetrainer(collector, features, xgb_model, rl_agent, lstm_model)
        health = HealthMonitor()

        # Intelligence module — Two-Brain System
        pred_tracker = PredictionTracker(db_path="data/trades.db")
        evaluator = EnsembleEvaluator()
        trigger_engine = AdaptiveTriggerEngine()
        logger.info(f"Intelligence module loaded: weights={evaluator.get_weights()}")

        logger.info("All components ready!")
        status_dict["state"] = "running"
        status_dict["message"] = "Trading actively (V2)"

        # Pair cooldown tracking: {pair: datetime_when_cooldown_expires}
        pair_cooldowns = {}

        def place_hard_stop_loss(pair: str, side: str, entry_price: float, amount: float = 0, stop_pct: float = None):
            """Place a hard stop-loss on OKX exchange so positions are protected even if bot crashes."""
            try:
                stop_pct = stop_pct or config.trading.stop_loss_percent
                if side == "short":
                    # Short stop-loss triggers when price goes UP
                    trigger_price = round(entry_price * (1 + stop_pct / 100), 6)
                    close_side = "buy"  # Buy to close short
                else:
                    # Long stop-loss triggers when price goes DOWN
                    trigger_price = round(entry_price * (1 - stop_pct / 100), 6)
                    close_side = "sell"  # Sell to close long

                result = okx.place_stop_loss_order(
                    pair=pair,
                    side=close_side,
                    trigger_price=str(trigger_price),
                    margin_mode=config.futures.margin_mode
                )
                algo_id = result.get("algoId", "")
                if algo_id:
                    # Store algo ID on the position
                    pos = positions.get_position(pair, mode="futures")
                    if not pos:
                        pos = positions.get_position(pair, mode="spot")
                    if pos:
                        pos.stop_loss_algo_id = algo_id
                    logger.info(f"HARD STOP-LOSS set: {pair} {side} @ ${trigger_price} (algo: {algo_id})")
                return algo_id
            except Exception as e:
                logger.error(f"Failed to place hard stop-loss for {pair}: {e}")
                return ""

        def cancel_hard_stop_loss(pair: str, algo_id: str):
            """Cancel a hard stop-loss when position is closed by the bot."""
            if not algo_id:
                return
            try:
                okx.cancel_algo_order(pair, algo_id)
                logger.info(f"Hard stop-loss cancelled: {pair} (algo: {algo_id})")
            except Exception as e:
                logger.warning(f"Failed to cancel stop-loss for {pair}: {e}")

        # Check for existing algo orders and attach to positions, then place missing ones
        try:
            existing_algos = okx.get_pending_algo_orders()
            for algo in existing_algos:
                inst_id = algo.get("instId", "").replace("-SWAP", "")
                algo_id = algo.get("algoId", "")
                if inst_id and algo_id:
                    pos = positions.get_position(inst_id, mode="futures")
                    if pos:
                        pos.stop_loss_algo_id = algo_id
                        logger.info(f"Attached existing stop-loss: {inst_id} (algo: {algo_id})")
        except Exception as e:
            logger.warning(f"Failed to check existing algo orders: {e}")

        # Place hard stop-losses on positions that don't have one yet
        for pos in positions.get_all_positions():
            if pos.mode == "futures" and not pos.stop_loss_algo_id:
                algo_id = place_hard_stop_loss(pos.pair, pos.side, pos.entry_price, pos.amount)
                if algo_id:
                    logger.info(f"Startup protection: {pos.pair} hard stop-loss placed")

        # Weekend Shield: store original max leverage to restore on weekdays
        _base_max_leverage = config.futures.leverage

        # Main loop
        cycle = 0
        while True:
            try:
                cycle += 1

                # ===== WEEKEND SHIELD =====
                # Friday 20:00 UTC → Monday 04:00 UTC: halve max leverage
                # Crypto weekends = low liquidity, wide spreads, Sunday dump risk
                utc_now = datetime.now(tz=__import__('datetime').timezone.utc)
                is_weekend_shield = (
                    (utc_now.weekday() == 4 and utc_now.hour >= 20) or  # Friday after 20:00 UTC
                    utc_now.weekday() == 5 or                            # Saturday
                    utc_now.weekday() == 6 or                            # Sunday
                    (utc_now.weekday() == 0 and utc_now.hour < 4)       # Monday before 04:00 UTC
                )
                if is_weekend_shield:
                    weekend_leverage = max(2, _base_max_leverage // 2)
                    if config.futures.leverage != weekend_leverage:
                        config.futures.leverage = weekend_leverage
                        logger.info(f"WEEKEND SHIELD ACTIVE: max leverage capped {_base_max_leverage}x → {weekend_leverage}x")
                else:
                    if config.futures.leverage != _base_max_leverage:
                        config.futures.leverage = _base_max_leverage
                        logger.info(f"WEEKEND SHIELD OFF: max leverage restored to {_base_max_leverage}x")

                logger.info(f"\n=== CYCLE {cycle} ===")

                # Re-sync futures positions to remove any closed externally
                if config.futures.enabled:
                    positions.sync_futures_from_exchange(config.futures.futures_pairs)

                # Update portfolio
                positions.update_prices()
                portfolio = positions.get_portfolio_summary()
                status_dict["portfolio"] = portfolio.get("total_value", 0)

                # === FAST BRAIN: Resolve prediction outcomes ===
                try:
                    current_prices = {}
                    for p in config.trading.trading_pairs:
                        try:
                            ticker = okx.get_ticker(p)
                            if ticker:
                                current_prices[p] = float(ticker.get("last", 0))
                        except Exception:
                            pass
                    resolved = pred_tracker.resolve_pending_outcomes(current_prices)

                    # Update ensemble weights if enough new outcomes
                    if pred_tracker.has_new_outcomes(min_count=5):
                        recent = pred_tracker.get_recent_outcomes(window=30)
                        if len(recent) >= 15:
                            new_weights = evaluator.update_weights(recent)
                            ensemble.set_weights(new_weights)
                            pred_tracker.reset_outcome_counter()
                except Exception as e:
                    logger.debug(f"Intelligence update: {e}")
                logger.info(f"Portfolio: ${portfolio['total_value']:.2f}")

                # Check risk
                if risk.should_pause_trading():
                    status_dict["message"] = "Trading paused (risk limit)"
                    time.sleep(3600)
                    continue

                # Clear caches at start of each cycle
                correlation_filter.clear_cache()
                funding_filter.clear_cache()

                # Check positions for stop-loss/trailing-stop/take-profit + adaptive leverage
                from datetime import timedelta
                for position in positions.get_all_positions():
                    pair = position.pair
                    mode_label = f" ({position.mode} {position.side})" if position.mode == "futures" else ""
                    lev_label = f" {position.leverage}x" if position.mode == "futures" else ""
                    logger.info(
                        f"  {pair}{mode_label}{lev_label}: ${position.current_price:.2f} | "
                        f"P&L: {position.unrealized_pnl_pct:+.2f}%"
                    )

                    def _close_this_position(pos=position):
                        # Cancel hard stop-loss first (we're closing manually)
                        if pos.stop_loss_algo_id:
                            cancel_hard_stop_loss(pos.pair, pos.stop_loss_algo_id)
                        if pos.mode == "futures":
                            result = executor.close_futures_position(
                                pos.pair, pos.entry_price, config.futures.margin_mode, pos.amount
                            )
                            if result.success:
                                positions.remove_futures_position(pos.pair)
                                risk.record_trade()
                            return result
                        else:
                            result = executor.close_position(pos.pair, pos.entry_price)
                            if result.success:
                                positions.remove_position(pos.pair)
                                risk.record_trade()
                            return result

                    # Compute ATR% for adaptive stop loss
                    atr_pct = 0.0
                    try:
                        sl_candles = collector.get_candles(pair, "1h", 30)
                        if len(sl_candles) >= 14:
                            sl_features = features.calculate_features(sl_candles)
                            if not sl_features.empty and "atr_14" in sl_features.columns:
                                atr_val = float(sl_features["atr_14"].iloc[-1])
                                if position.current_price > 0:
                                    atr_pct = (atr_val / position.current_price) * 100
                    except Exception:
                        pass

                    if risk.check_stop_loss(position, atr_pct=atr_pct):
                        logger.warning(f"STOP-LOSS: {pair}{mode_label}")
                        notify_stoploss(pair, position.side or "long", position.unrealized_pnl_pct)
                        _close_this_position()
                        # 4-hour cooldown after a loss
                        pair_cooldowns[pair] = datetime.now() + timedelta(hours=4)
                        logger.info(f"[{pair}] 4h cooldown activated until {pair_cooldowns[pair].strftime('%H:%M')}")
                        continue

                    if risk.check_trailing_stop(position):
                        logger.info(f"TRAILING STOP: {pair}{mode_label} - locking profit")
                        notify_close(pair, position.side or "long", position.unrealized_pnl_pct, 0, "Trailing stop")
                        _close_this_position()
                        continue

                    if risk.check_take_profit(position):
                        logger.info(f"TAKE-PROFIT: {pair}{mode_label}")
                        notify_takeprofit(pair, position.side or "long", position.unrealized_pnl_pct)
                        _close_this_position()
                        continue

                    # Break-even guard: move stop-loss to entry price at +1.5% profit
                    if (position.mode == "futures" and position.stop_loss_algo_id
                            and position.unrealized_pnl_pct >= 1.5
                            and not getattr(position, '_breakeven_set', False)):
                        try:
                            # Cancel current stop-loss
                            cancel_hard_stop_loss(pair, position.stop_loss_algo_id)
                            # Place new stop at entry price (break-even)
                            if position.side == "short":
                                close_side = "buy"
                            else:
                                close_side = "sell"
                            result = okx.place_stop_loss_order(
                                pair=pair,
                                side=close_side,
                                trigger_price=str(position.entry_price),
                                margin_mode=config.futures.margin_mode
                            )
                            new_algo_id = result.get("algoId", "")
                            if new_algo_id:
                                position.stop_loss_algo_id = new_algo_id
                                position._breakeven_set = True
                                logger.info(
                                    f"BREAK-EVEN GUARD: {pair} stop moved to entry ${position.entry_price:.4f} "
                                    f"(was at -5%, now risk-free at +{position.unrealized_pnl_pct:.1f}%)"
                                )
                                notify_close(pair, position.side, position.unrealized_pnl_pct, 0,
                                           f"Stop moved to break-even (risk-free trade)")
                        except Exception as e:
                            logger.warning(f"Break-even guard failed for {pair}: {e}")

                    # Adaptive leverage adjustment for open futures positions
                    # Skip if hard stop-loss exists (OKX blocks leverage changes with algo orders)
                    if position.mode == "futures" and config.futures.enabled and not position.stop_loss_algo_id:
                        new_lev = executor.calculate_adaptive_leverage(
                            position, config.futures.leverage
                        )
                        if new_lev != position.leverage:
                            try:
                                okx.set_leverage(pair, new_lev, config.futures.margin_mode)
                                old_lev = position.leverage
                                position.leverage = new_lev
                                logger.info(
                                    f"LEVERAGE ADJUSTED: {pair} {old_lev}x -> {new_lev}x "
                                    f"(P&L: {position.unrealized_pnl_pct:+.1f}%)"
                                )
                            except Exception as e:
                                logger.warning(f"Leverage adjustment failed for {pair}: {e}")

                # Clean expired cooldowns
                now = datetime.now()
                expired = [p for p, t in pair_cooldowns.items() if now >= t]
                for p in expired:
                    pair_cooldowns.pop(p)
                    logger.info(f"[{p}] Cooldown expired, pair re-enabled")

                # ===== SUCCESSION RULE: Track blocked high-conviction signals =====
                succession_candidates = []  # [(pair, decision, price, dyn_leverage, df_features)]
                succession_done_this_cycle = False

                # Analyze pairs
                for pair in config.trading.trading_pairs:
                    try:
                        if positions.has_position(pair):
                            continue

                        # Check 4h cooldown after loss
                        if pair in pair_cooldowns:
                            remaining = (pair_cooldowns[pair] - now).total_seconds() / 60
                            logger.info(f"[{pair}] Cooldown active ({remaining:.0f}min remaining)")
                            continue

                        candles_1h = collector.get_candles(pair, "1h", 300)
                        if len(candles_1h) < 50:
                            continue

                        # Fetch higher timeframes
                        try:
                            candles_4h = collector.get_candles(pair, "4h", 100)
                            candles_1d = collector.get_candles(pair, "1d", 100)
                        except Exception:
                            candles_4h, candles_1d = None, None

                        df_features = features.calculate_multi_tf_features(
                            candles_1h, candles_4h, candles_1d
                        )
                        if df_features.empty:
                            continue

                        # Inject funding rate as feature (pair-specific)
                        try:
                            fr_data = okx.get_funding_rate(pair)
                            if fr_data and "fundingRate" in fr_data:
                                fr = float(fr_data["fundingRate"])
                                df_features = df_features.copy()
                                df_features["funding_rate"] = fr * 100
                                df_features["funding_contrarian"] = -fr * 1000
                            else:
                                df_features = df_features.copy()
                                df_features["funding_rate"] = 0.0
                                df_features["funding_contrarian"] = 0.0
                        except Exception:
                            df_features = df_features.copy()
                            df_features["funding_rate"] = 0.0
                            df_features["funding_contrarian"] = 0.0

                        latest = df_features.iloc[[-1]]
                        latest_row = df_features.iloc[-1]
                        price = float(candles_1h["close"].iloc[-1])

                        # ===== V2: DETECT REGIME =====
                        regime_params = regime_detector.detect_and_get_params(latest_row, df_features)
                        logger.info(f"[{pair}] Regime: {regime_params.name} - {regime_params.description}")

                        # ===== V2: GET TREND =====
                        trend = trend_filter.get_trend(latest_row)
                        trend_score = trend_filter._calculate_trend_score(latest_row)

                        # ===== V2: 3-MODEL ENSEMBLE DECISION =====
                        portfolio_state = {
                            "balance": okx.get_usdt_balance(),
                            "position": 0,
                            "entry_price": 0,
                            "current_price": price
                        }

                        # Get volatility for meta-features
                        volatility = float(latest_row.get("volatility_14", 0)) if "volatility_14" in latest_row.index else 0.0

                        decision = ensemble.get_decision(
                            latest, portfolio_state, pair,
                            regime=regime_params.name,
                            volatility=volatility,
                            trend_strength=trend_score
                        )

                        # === FAST BRAIN: Record prediction ===
                        if decision.action != Action.HOLD:
                            try:
                                _pred_id = pred_tracker.record_prediction(
                                    pair=pair,
                                    xgb_action=decision.xgb_action,
                                    xgb_conf=decision.xgb_confidence,
                                    rl_action=decision.rl_action,
                                    rl_conf=decision.rl_confidence,
                                    lstm_action=decision.lstm_action,
                                    lstm_conf=decision.lstm_confidence,
                                    ensemble_action=decision.action,
                                    ensemble_conf=decision.confidence,
                                    price=price,
                                    volatility=volatility,
                                    regime=regime_params.name,
                                    trend_strength=trend_score,
                                )
                            except Exception as e:
                                _pred_id = None
                                logger.debug(f"Prediction tracking: {e}")

                        if decision.action == Action.HOLD:
                            continue

                        # ===== V2: TREND FILTER =====
                        if not trend_filter.filter_decision(trend, decision.action):
                            trend_name = trend.value.upper()
                            action_name = "BUY" if decision.action == 1 else "SELL"
                            logger.info(
                                f"[{pair}] FILTERED: {action_name} blocked by {trend_name} trend "
                                f"(score: {trend_score:.2f})"
                            )
                            continue

                        # ===== V2: SIMPLE CONFIDENCE CHECK =====
                        # Removed funding rate, regime offset, and correlation filters
                        # They were over-filtering: only 4.4% of signals survived all 5 layers
                        if decision.confidence < config.trading.min_confidence_to_trade:
                            logger.info(
                                f"[{pair}] Below confidence threshold: {decision.confidence:.2f} < "
                                f"{config.trading.min_confidence_to_trade:.2f}"
                            )
                            continue

                        # ===== EXECUTE TRADE =====
                        # Apply regime position size multiplier
                        effective_size_mult = regime_params.position_size_mult

                        # Check if we're at max positions — queue for succession instead
                        current_pos_count = len(positions.get_all_positions())
                        at_max_positions = current_pos_count >= config.trading.max_open_positions

                        if at_max_positions and decision.confidence >= 0.75:
                            # High-conviction signal blocked by full slots — candidate for succession
                            dyn_leverage = executor.calculate_dynamic_leverage(
                                decision.confidence, config.futures.leverage
                            )
                            succession_candidates.append({
                                "pair": pair,
                                "decision": decision,
                                "price": price,
                                "dyn_leverage": dyn_leverage,
                                "action": decision.action,
                            })
                            logger.info(
                                f"[{pair}] SUCCESSION CANDIDATE: "
                                f"{'BUY' if decision.action == Action.BUY else 'SELL'} @ {decision.confidence:.2f} conf "
                                f"(queued — max positions reached)"
                            )
                            continue

                        if decision.action == Action.BUY:
                            if (config.futures.enabled
                                    and pair in config.futures.futures_pairs
                                    and not positions.has_position(pair, mode="futures")):
                                # Futures long — leverage maximizes gains on small capital
                                dyn_leverage = executor.calculate_dynamic_leverage(
                                    decision.confidence, config.futures.leverage
                                )
                                logger.info(
                                    f"LONG {pair} (conf: {decision.confidence:.2f}, "
                                    f"leverage: {dyn_leverage}x, regime: {regime_params.name})"
                                )
                                result = executor.execute_long_open(
                                    decision, price, config.futures,
                                    dynamic_leverage=dyn_leverage
                                )
                                if result.success:
                                    positions.add_position(
                                        pair, result.amount, result.price,
                                        side="long", mode="futures",
                                        leverage=dyn_leverage,
                                        max_leverage=config.futures.leverage,
                                        entry_confidence=decision.confidence
                                    )
                                    risk.record_trade()
                                    # Place hard stop-loss on exchange
                                    place_hard_stop_loss(pair, "long", result.price, result.amount)
                                    logger.info(
                                        f"Longed {result.amount} {pair} @ ${result.price} "
                                        f"({dyn_leverage}x leverage)"
                                    )
                                    notify_trade(pair, "BUY", "long", dyn_leverage,
                                                 decision.confidence, result.price, result.amount * result.price)

                        elif decision.action == Action.SELL:
                            if (config.futures.enabled
                                    and pair in config.futures.futures_pairs
                                    and not positions.has_position(pair, mode="futures")):
                                dyn_leverage = executor.calculate_dynamic_leverage(
                                    decision.confidence, config.futures.leverage
                                )
                                logger.info(
                                    f"SHORT {pair} (conf: {decision.confidence:.2f}, "
                                    f"leverage: {dyn_leverage}x, regime: {regime_params.name})"
                                )
                                result = executor.execute_short_open(
                                    decision, price, config.futures,
                                    dynamic_leverage=dyn_leverage
                                )
                                if result.success:
                                    positions.add_position(
                                        pair, result.amount, result.price,
                                        side="short", mode="futures",
                                        leverage=dyn_leverage,
                                        max_leverage=config.futures.leverage,
                                        entry_confidence=decision.confidence
                                    )
                                    risk.record_trade()
                                    # Place hard stop-loss on exchange
                                    place_hard_stop_loss(pair, "short", result.price, result.amount)
                                    logger.info(
                                        f"Shorted {result.amount} {pair} @ ${result.price} "
                                        f"({dyn_leverage}x leverage)"
                                    )
                                    notify_trade(pair, "SELL", "short", dyn_leverage,
                                                 decision.confidence, result.price, result.amount * result.price)
                            else:
                                logger.debug(f"SELL signal for {pair} but no position")

                    except Exception as e:
                        logger.error(f"Error on {pair}: {e}")

                # ===== SUCCESSION RULE: Auto-rotate weak positions for strong signals =====
                if succession_candidates and not succession_done_this_cycle:
                    try:
                        # Pick the BEST candidate (highest conviction)
                        best = max(succession_candidates, key=lambda c: c["decision"].confidence)
                        logger.info(
                            f"SUCCESSION CHECK: {len(succession_candidates)} candidate(s), "
                            f"best={best['pair']} @ {best['decision'].confidence:.2f}"
                        )

                        # Re-scan existing positions through the brain to get CURRENT conviction
                        position_convictions = []
                        for pos in positions.get_all_positions():
                            if pos.mode != "futures":
                                continue
                            try:
                                pos_candles_1h = collector.get_candles(pos.pair, "1h", 300)
                                if len(pos_candles_1h) < 50:
                                    continue
                                try:
                                    pos_candles_4h = collector.get_candles(pos.pair, "4h", 100)
                                    pos_candles_1d = collector.get_candles(pos.pair, "1d", 100)
                                except Exception:
                                    pos_candles_4h, pos_candles_1d = None, None

                                pos_df = features.calculate_multi_tf_features(
                                    pos_candles_1h, pos_candles_4h, pos_candles_1d
                                )
                                if pos_df.empty:
                                    continue

                                # Inject funding rate
                                try:
                                    fr_data = okx.get_funding_rate(pos.pair)
                                    if fr_data and "fundingRate" in fr_data:
                                        fr = float(fr_data["fundingRate"])
                                        pos_df = pos_df.copy()
                                        pos_df["funding_rate"] = fr * 100
                                        pos_df["funding_contrarian"] = -fr * 1000
                                    else:
                                        pos_df = pos_df.copy()
                                        pos_df["funding_rate"] = 0.0
                                        pos_df["funding_contrarian"] = 0.0
                                except Exception:
                                    pos_df = pos_df.copy()
                                    pos_df["funding_rate"] = 0.0
                                    pos_df["funding_contrarian"] = 0.0

                                pos_latest = pos_df.iloc[[-1]]
                                pos_latest_row = pos_df.iloc[-1]
                                pos_price = float(pos_candles_1h["close"].iloc[-1])
                                pos_vol = float(pos_latest_row.get("volatility_14", 0)) if "volatility_14" in pos_latest_row.index else 0.0

                                pos_portfolio_state = {
                                    "balance": okx.get_usdt_balance(),
                                    "position": 0,
                                    "entry_price": 0,
                                    "current_price": pos_price
                                }
                                pos_trend_score = trend_filter._calculate_trend_score(pos_latest_row)

                                pos_decision = ensemble.get_decision(
                                    pos_latest, pos_portfolio_state, pos.pair,
                                    regime=regime_detector.detect_and_get_params(pos_latest_row, pos_df).name,
                                    volatility=pos_vol,
                                    trend_strength=pos_trend_score
                                )

                                # Calculate "alignment score": how much does brain agree with current position?
                                # If position is SHORT and brain says SELL, alignment is confidence
                                # If position is SHORT and brain says BUY, alignment is -confidence (brain disagrees)
                                if pos.side == "short":
                                    if pos_decision.action == Action.SELL:
                                        alignment = pos_decision.confidence
                                    elif pos_decision.action == Action.BUY:
                                        alignment = -pos_decision.confidence  # Brain flipped!
                                    else:
                                        alignment = 0.0  # HOLD = neutral
                                elif pos.side == "long":
                                    if pos_decision.action == Action.BUY:
                                        alignment = pos_decision.confidence
                                    elif pos_decision.action == Action.SELL:
                                        alignment = -pos_decision.confidence  # Brain flipped!
                                    else:
                                        alignment = 0.0
                                else:
                                    alignment = 0.0

                                position_convictions.append({
                                    "position": pos,
                                    "alignment": alignment,
                                    "brain_action": pos_decision.action,
                                    "brain_conf": pos_decision.confidence,
                                    "pnl_pct": pos.unrealized_pnl_pct,
                                })
                                logger.info(
                                    f"  {pos.pair} {pos.side}: alignment={alignment:+.2f} "
                                    f"(brain={'BUY' if pos_decision.action==1 else 'SELL' if pos_decision.action==-1 else 'HOLD'}"
                                    f"@{pos_decision.confidence:.2f}), PnL={pos.unrealized_pnl_pct:+.2f}%"
                                )
                            except Exception as e:
                                logger.debug(f"Succession scan error for {pos.pair}: {e}")

                        if position_convictions:
                            # Find weakest position (lowest alignment = brain most disagrees)
                            weakest = min(position_convictions, key=lambda c: c["alignment"])

                            # SUCCESSION TRIGGER:
                            # - Best candidate conviction > 0.75 (already filtered above)
                            # - Weakest position alignment < 0.25 (brain no longer supports it)
                            # - OR weakest alignment is NEGATIVE (brain actively says opposite direction)
                            if weakest["alignment"] < 0.25:
                                w_pos = weakest["position"]
                                logger.info(
                                    f"SUCCESSION TRIGGERED: "
                                    f"Close {w_pos.pair} {w_pos.side} (alignment={weakest['alignment']:+.2f}) "
                                    f"→ Open {best['pair']} (conf={best['decision'].confidence:.2f})"
                                )

                                # Step 1: Close the weakest position
                                if w_pos.stop_loss_algo_id:
                                    cancel_hard_stop_loss(w_pos.pair, w_pos.stop_loss_algo_id)
                                close_result = executor.close_futures_position(
                                    w_pos.pair, w_pos.entry_price,
                                    config.futures.margin_mode, w_pos.amount
                                )

                                if close_result.success:
                                    positions.remove_futures_position(w_pos.pair)
                                    risk.record_trade()
                                    close_pnl = w_pos.unrealized_pnl_pct

                                    # Add cooldown on closed pair
                                    pair_cooldowns[w_pos.pair] = datetime.now() + timedelta(hours=2)

                                    # Step 2: Open the new sniper position
                                    time.sleep(1)  # Brief pause for margin to free up
                                    b_decision = best["decision"]
                                    b_price = best["price"]
                                    b_leverage = best["dyn_leverage"]

                                    if b_decision.action == Action.BUY:
                                        open_result = executor.execute_long_open(
                                            b_decision, b_price, config.futures,
                                            dynamic_leverage=b_leverage
                                        )
                                        open_side = "long"
                                    else:
                                        open_result = executor.execute_short_open(
                                            b_decision, b_price, config.futures,
                                            dynamic_leverage=b_leverage
                                        )
                                        open_side = "short"

                                    if open_result.success:
                                        positions.add_position(
                                            best["pair"], open_result.amount, open_result.price,
                                            side=open_side, mode="futures",
                                            leverage=b_leverage,
                                            max_leverage=config.futures.leverage,
                                            entry_confidence=b_decision.confidence
                                        )
                                        risk.record_trade()
                                        place_hard_stop_loss(best["pair"], open_side, open_result.price, open_result.amount)

                                        succession_done_this_cycle = True
                                        logger.info(
                                            f"SUCCESSION COMPLETE: {w_pos.pair}→{best['pair']} "
                                            f"({open_side} {b_leverage}x @ ${open_result.price})"
                                        )

                                        # Telegram notification
                                        notify_alert(
                                            "SUCCESSION RULE",
                                            f"Rotated: {w_pos.pair} {w_pos.side} (alignment {weakest['alignment']:+.2f}, PnL {close_pnl:+.2f}%)\n"
                                            f"→ {best['pair']} {open_side} {b_leverage}x @ ${open_result.price:.4f}\n"
                                            f"Conviction: {b_decision.confidence:.2f} | Hard stop placed"
                                        )
                                    else:
                                        logger.warning(f"Succession open failed for {best['pair']}: {open_result}")
                                        notify_alert(
                                            "SUCCESSION PARTIAL",
                                            f"Closed {w_pos.pair} {w_pos.side} (PnL {close_pnl:+.2f}%) "
                                            f"but failed to open {best['pair']}. Slot freed."
                                        )
                                else:
                                    logger.warning(f"Succession close failed for {w_pos.pair}")
                            else:
                                logger.info(
                                    f"Succession skipped: weakest={weakest['position'].pair} "
                                    f"alignment={weakest['alignment']:+.2f} (still > 0.25 threshold)"
                                )
                    except Exception as e:
                        logger.error(f"Succession rule error: {e}")
                        import traceback
                        logger.error(traceback.format_exc())

                # === SLOW BRAIN: Adaptive retrain check ===
                try:
                    perf_metrics = pred_tracker.get_performance_metrics()
                    market_ctx = {"volatility": volatility} if 'volatility' in dir() else {}
                    trigger_engine.update_metrics(perf_metrics, market_ctx)

                    if trigger_engine.should_retrain():
                        trigger_info = trigger_engine.get_trigger_info()
                        severity = trigger_info.get("severity", "medium")
                        reason = trigger_info.get("reason", "unknown")
                        logger.info(
                            f"RETRAIN TRIGGERED: {reason} (severity={severity})"
                        )
                        notify_alert(
                            f"Retrain triggered: {reason} (severity={severity})\n"
                            f"Metrics: {perf_metrics}"
                        )

                        status_dict["message"] = f"Retraining ({severity})..."
                        retrainer.retrain_models(config.trading.trading_pairs)
                        trigger_engine.record_retrain_completed(severity)
                        # Reload weights after retrain
                        evaluator_weights = evaluator.get_weights()
                        ensemble.set_weights(evaluator_weights)
                        status_dict["message"] = "Trading actively (V2)"
                    elif retrainer.should_retrain():
                        # Fallback: calendar-based retrain still runs
                        status_dict["message"] = "Scheduled retraining..."
                        retrainer.retrain_models(config.trading.trading_pairs)
                        trigger_engine.record_retrain_completed("medium")
                        status_dict["message"] = "Trading actively (V2)"
                except Exception as e:
                    logger.debug(f"Slow brain check: {e}")

                # Periodic cleanup (every 100 cycles)
                if cycle % 100 == 0:
                    pred_tracker.cleanup_old_records(keep_days=90)

                status_dict["message"] = f"Cycle {cycle} done. Next in {config.trading.analysis_interval_minutes}min"
                time.sleep(config.trading.analysis_interval_minutes * 60)

            except Exception as e:
                logger.error(f"Loop error: {e}")
                if not watchdog._on_error(e):
                    break

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        status_dict["state"] = "error"
        status_dict["message"] = str(e)
        while True:
            time.sleep(60)


def initial_training(collector, features, xgb_model, rl_agent, trading_pairs):
    """Train models on historical data with multi-timeframe features."""
    import pandas as pd
    from src.data.features import create_target_labels

    logger.info("Collecting training data (180 days, multi-timeframe)...")
    all_data = []

    for pair in trading_pairs:
        try:
            df_1h = collector.get_historical_data(pair, days=180, timeframe="1h")
            try:
                df_4h = collector.get_historical_data(pair, days=180, timeframe="4h")
                df_1d = collector.get_historical_data(pair, days=180, timeframe="1d")
            except Exception:
                df_4h, df_1d = None, None

            df_features = features.calculate_multi_tf_features(df_1h, df_4h, df_1d)
            df_features["target"] = create_target_labels(df_features)
            df_features["pair"] = pair
            all_data.append(df_features)
        except Exception as e:
            logger.error(f"Data collection failed for {pair}: {e}")

    if not all_data:
        raise ValueError("No training data!")

    combined = pd.concat(all_data, ignore_index=True).dropna()
    logger.info(f"Training on {len(combined)} samples")

    feature_cols = features.get_feature_names()
    X = combined[feature_cols]
    y = combined["target"]

    xgb_model.train(X, y)
    rl_agent.train(combined, feature_cols, total_timesteps=500000)

    logger.info("Training complete!")


if __name__ == "__main__":
    # For direct running (not via app.py)
    status = {"state": "starting", "message": "", "portfolio": 0}
    run_bot(status)
