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

        logger.info("All components ready!")
        status_dict["state"] = "running"
        status_dict["message"] = "Trading actively (V2)"

        # Main loop
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"\n=== CYCLE {cycle} ===")

                # Update portfolio
                positions.update_prices()
                portfolio = positions.get_portfolio_summary()
                status_dict["portfolio"] = portfolio.get("total_value", 0)
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
                for position in positions.get_all_positions():
                    pair = position.pair
                    mode_label = f" ({position.mode} {position.side})" if position.mode == "futures" else ""
                    lev_label = f" {position.leverage}x" if position.mode == "futures" else ""
                    logger.info(
                        f"  {pair}{mode_label}{lev_label}: ${position.current_price:.2f} | "
                        f"P&L: {position.unrealized_pnl_pct:+.2f}%"
                    )

                    def _close_this_position(pos=position):
                        if pos.mode == "futures":
                            result = executor.close_futures_position(
                                pos.pair, pos.entry_price, config.futures.margin_mode
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

                    if risk.check_stop_loss(position):
                        logger.warning(f"STOP-LOSS: {pair}{mode_label}")
                        notify_stoploss(pair, position.side or "long", position.unrealized_pnl_pct)
                        _close_this_position()
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

                    # Adaptive leverage adjustment for open futures positions
                    if position.mode == "futures" and config.futures.enabled:
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

                # Analyze pairs
                for pair in config.trading.trading_pairs:
                    try:
                        if positions.has_position(pair):
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

                        # ===== V2: FUNDING RATE ADJUSTMENT =====
                        adjusted_conf, funding_reason = funding_filter.adjust_decision_confidence(
                            pair, decision.action, decision.confidence
                        )
                        decision.confidence = adjusted_conf

                        # ===== V2: REGIME-ADJUSTED CONFIDENCE THRESHOLD =====
                        effective_threshold = config.trading.min_confidence_to_trade + regime_params.confidence_offset
                        if decision.confidence < effective_threshold:
                            logger.info(
                                f"[{pair}] Below regime threshold: {decision.confidence:.2f} < "
                                f"{effective_threshold:.2f} ({regime_params.name})"
                            )
                            continue

                        # ===== V2: CORRELATION FILTER =====
                        all_positions = positions.get_all_positions()
                        direction = 1 if decision.action == Action.BUY else -1
                        corr_allowed, corr_reason = correlation_filter.should_allow_trade(
                            pair, direction, all_positions
                        )
                        if not corr_allowed:
                            logger.info(f"[{pair}] Correlation filter: {corr_reason}")
                            continue

                        # ===== EXECUTE TRADE =====
                        # Apply regime position size multiplier
                        effective_size_mult = regime_params.position_size_mult

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

                # Auto-retrain check
                if retrainer.should_retrain():
                    status_dict["message"] = "Retraining models..."
                    retrainer.retrain_models(config.trading.trading_pairs)
                    status_dict["message"] = "Trading actively (V2)"

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
