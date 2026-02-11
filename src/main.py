"""
Autonomous AI Trading Agent - Core Trading Logic
=================================================
"""

import os
import sys
import time
import logging
from datetime import datetime

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
    logger.info("TRADING BOT STARTING")
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
        from src.models.ensemble import EnsembleDecider, Action
        from src.risk.manager import RiskManager, RiskLimits
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
                # Error 59000 = can't change mode with open positions (already set)
                if "59000" in str(e):
                    logger.info("Position mode already set (open positions exist) - OK")
                else:
                    logger.error(f"Position mode setup failed: {e}")
            try:
                for pair in config.futures.futures_pairs:
                    okx.set_leverage(pair, config.futures.leverage, config.futures.margin_mode)
                    logger.info(f"Leverage set: {pair} -> {config.futures.leverage}x ({config.futures.margin_mode})")
            except Exception as e:
                logger.error(f"Leverage setup failed: {e}")
                logger.warning("Continuing with spot-only trading")
                config.futures.enabled = False
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

        xgb_model = XGBoostPredictor(model_path="models/xgboost_model.json")
        rl_agent = RLTradingAgent(model_path="models/rl_agent.zip")

        if not xgb_model.is_trained or not rl_agent.is_trained:
            status_dict["message"] = "Training ML models (first run)..."
            logger.info("Training models...")
            initial_training(collector, features, xgb_model, rl_agent, config.trading.trading_pairs)

        ensemble = EnsembleDecider(xgb_model, rl_agent, min_confidence=config.trading.min_confidence_to_trade)

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
        retrainer = AutoRetrainer(collector, features, xgb_model, rl_agent)
        health = HealthMonitor()

        logger.info("All components ready!")
        status_dict["state"] = "running"
        status_dict["message"] = "Trading actively"

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

                # Check positions for stop-loss/trailing-stop/take-profit
                for position in positions.get_all_positions():
                    pair = position.pair
                    mode_label = f" ({position.mode} {position.side})" if position.mode == "futures" else ""
                    logger.info(
                        f"  {pair}{mode_label}: ${position.current_price:.2f} | "
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
                        _close_this_position()
                        continue

                    if risk.check_trailing_stop(position):
                        logger.info(f"TRAILING STOP: {pair}{mode_label} - locking profit")
                        _close_this_position()
                        continue

                    if risk.check_take_profit(position):
                        logger.info(f"TAKE-PROFIT: {pair}{mode_label}")
                        _close_this_position()
                        continue

                # Analyze pairs
                for pair in config.trading.trading_pairs:
                    try:
                        if positions.has_position(pair):
                            continue

                        candles_1h = collector.get_candles(pair, "1h", 300)
                        if len(candles_1h) < 50:
                            continue

                        # Fetch higher timeframes for better context
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
                        price = float(candles_1h["close"].iloc[-1])

                        portfolio_state = {
                            "balance": okx.get_usdt_balance(),
                            "position": 0,
                            "entry_price": 0,
                            "current_price": price
                        }

                        decision = ensemble.get_decision(latest, portfolio_state, pair)

                        if decision.action == Action.BUY:
                            logger.info(f"BUY {pair} (conf: {decision.confidence:.2f})")
                            result = executor.execute(decision, price)
                            if result.success:
                                positions.add_position(pair, result.amount, result.price)
                                logger.info(f"Bought {result.amount} {pair} @ ${result.price}")

                        elif decision.action == Action.SELL:
                            # SELL signal with no spot position - open short if futures enabled
                            if (config.futures.enabled
                                    and pair in config.futures.futures_pairs
                                    and not positions.has_position(pair, mode="futures")):
                                logger.info(f"SHORT {pair} (conf: {decision.confidence:.2f})")
                                result = executor.execute_short_open(decision, price, config.futures)
                                if result.success:
                                    positions.add_position(
                                        pair, result.amount, result.price,
                                        side="short", mode="futures"
                                    )
                                    risk.record_trade()
                                    logger.info(f"Shorted {result.amount} {pair} @ ${result.price}")
                            else:
                                logger.debug(f"SELL signal for {pair} but no position")

                    except Exception as e:
                        logger.error(f"Error on {pair}: {e}")

                # Auto-retrain check
                if retrainer.should_retrain():
                    status_dict["message"] = "Retraining models..."
                    retrainer.retrain_models(config.trading.trading_pairs)
                    status_dict["message"] = "Trading actively"

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
