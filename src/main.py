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
        status_dict["state"] = "initializing"
        status_dict["message"] = "Loading ML models..."

        collector = DataCollector(okx)
        features = FeatureEngine()
        positions = PositionTracker(okx)
        positions.sync_from_exchange()

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

                # Check positions for stop-loss/take-profit
                for pos in positions.get_all_positions():
                    if risk.check_stop_loss(pos):
                        logger.warning(f"STOP-LOSS: {pos.pair}")
                        result = executor.close_position(pos.pair, pos.entry_price)
                        if result.success:
                            positions.remove_position(pos.pair)
                    elif risk.check_take_profit(pos):
                        logger.info(f"TAKE-PROFIT: {pos.pair}")
                        result = executor.close_position(pos.pair, pos.entry_price)
                        if result.success:
                            positions.remove_position(pos.pair)

                # Analyze pairs
                for pair in config.trading.trading_pairs:
                    try:
                        if positions.has_position(pair):
                            continue

                        candles = collector.get_candles(pair, "1h", 100)
                        if len(candles) < 50:
                            continue

                        df_features = features.calculate_features(candles)
                        if df_features.empty:
                            continue

                        latest = df_features.iloc[[-1]]
                        price = float(candles["close"].iloc[-1])

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
    """Train models on historical data."""
    import pandas as pd
    from src.data.features import create_target_labels

    logger.info("Collecting training data...")
    all_data = []

    for pair in trading_pairs:
        try:
            df = collector.get_historical_data(pair, days=60, timeframe="1h")
            df_features = features.calculate_features(df)
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
    rl_agent.train(combined, feature_cols, total_timesteps=50000)

    logger.info("Training complete!")


if __name__ == "__main__":
    # For direct running (not via app.py)
    status = {"state": "starting", "message": "", "portfolio": 0}
    run_bot(status)
