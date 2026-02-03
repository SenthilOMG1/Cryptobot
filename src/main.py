"""
Autonomous AI Trading Agent - Main Entry Point
===============================================
This is the heart of the system. It runs forever, trading autonomously.

Usage:
    python -m src.main
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime
from typing import Optional

# Ensure data directories exist FIRST
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point - runs the autonomous trading loop."""

    logger.info("=" * 60)
    logger.info("AUTONOMOUS AI TRADING AGENT - STARTING")
    logger.info("=" * 60)

    # ==================== START DASHBOARD IMMEDIATELY ====================
    # This must happen FIRST so Railway sees the service as healthy

    from src.dashboard import run_dashboard, set_components

    dashboard_port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting dashboard on port {dashboard_port}...")
    run_dashboard(port=dashboard_port)
    logger.info(f"Dashboard LIVE at http://0.0.0.0:{dashboard_port}")

    # Give dashboard a moment to start
    time.sleep(2)

    # ==================== INITIALIZE TRADING (can take time) ====================

    logger.info("Initializing trading components...")

    try:
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

        # Load configuration
        config = get_config()
        logger.info(f"Configuration loaded")

        # Initialize security vault
        logger.info("Initializing secure vault...")
        vault = SecureVault()

        # Initialize OKX client
        logger.info("Connecting to OKX...")
        okx = SecureOKXClient(vault, demo_mode=config.exchange.demo_mode)

        # Test connection
        if not okx.test_connection():
            logger.error("Failed to connect to OKX! Check your API keys.")
            logger.error("Bot will keep running - fix API keys and restart.")
            # Don't exit - keep dashboard running
            while True:
                time.sleep(60)

        logger.info("OKX connection successful!")

        # Initialize components
        logger.info("Initializing trading components...")

        # Data collection
        collector = DataCollector(okx)
        features = FeatureEngine()

        # Position tracking
        positions = PositionTracker(okx)
        positions.sync_from_exchange()

        # ML models
        xgb_model = XGBoostPredictor(model_path="models/xgboost_model.json")
        rl_agent = RLTradingAgent(model_path="models/rl_agent.zip")

        # Check if models are trained
        if not xgb_model.is_trained or not rl_agent.is_trained:
            logger.warning("ML models not found! Running initial training...")
            initial_training(collector, features, xgb_model, rl_agent, config.trading.trading_pairs)

        # Ensemble decider
        ensemble = EnsembleDecider(
            xgb_model, rl_agent,
            min_confidence=config.trading.min_confidence_to_trade
        )

        # Risk management
        risk_limits = RiskLimits(
            max_position_pct=config.trading.max_position_percent,
            stop_loss_pct=config.trading.stop_loss_percent,
            take_profit_pct=config.trading.take_profit_percent,
            daily_loss_limit_pct=config.trading.daily_loss_limit_percent,
            max_open_positions=config.trading.max_open_positions,
            min_confidence=config.trading.min_confidence_to_trade
        )
        risk = RiskManager(positions, risk_limits)

        # Trade executor
        executor = TradeExecutor(okx, risk)

        # Autonomous systems
        watchdog = Watchdog(max_retries=10, retry_delay=60)
        retrainer = AutoRetrainer(collector, features, xgb_model, rl_agent)
        health = HealthMonitor()

        # Register components for health monitoring
        health.register_component("okx_connection", True)
        health.register_component("xgb_model", xgb_model.is_trained)
        health.register_component("rl_agent", rl_agent.is_trained)

        logger.info("All components initialized successfully!")

        # Update dashboard with component references
        set_components({
            "positions": positions,
            "risk": risk,
            "health": health,
            "executor": executor
        })

        # ==================== MAIN TRADING LOOP ====================

        logger.info("Starting autonomous trading loop...")
        logger.info(f"Trading pairs: {config.trading.trading_pairs}")
        logger.info(f"Analysis interval: {config.trading.analysis_interval_minutes} minutes")

        cycle_count = 0

        while True:
            try:
                cycle_count += 1
                cycle_start = datetime.now()

                logger.info(f"\n{'='*50}")
                logger.info(f"CYCLE #{cycle_count} - {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*50}")

                # Health Check
                health_status = health.check_system_health()
                if not health_status["is_healthy"]:
                    logger.warning(f"System unhealthy: {health_status}")

                # Check if trading should pause
                if risk.should_pause_trading():
                    logger.warning("Trading paused by risk manager. Waiting...")
                    time.sleep(3600)
                    continue

                # Update positions
                positions.update_prices()
                portfolio = positions.get_portfolio_summary()
                logger.info(f"Portfolio: ${portfolio['total_value']:.2f} USDT")
                logger.info(f"Open positions: {portfolio['num_positions']}")

                # Check existing positions
                for position in positions.get_all_positions():
                    pair = position.pair
                    logger.info(
                        f"  {pair}: ${position.current_price:.2f} | "
                        f"P&L: {position.unrealized_pnl_pct:+.2f}%"
                    )

                    if risk.check_stop_loss(position):
                        logger.warning(f"STOP-LOSS triggered for {pair}")
                        result = executor.close_position(pair, position.entry_price)
                        if result.success:
                            positions.remove_position(pair)
                            risk.record_trade()
                        continue

                    if risk.check_take_profit(position):
                        logger.info(f"TAKE-PROFIT triggered for {pair}")
                        result = executor.close_position(pair, position.entry_price)
                        if result.success:
                            positions.remove_position(pair)
                            risk.record_trade()
                        continue

                # Analyze each trading pair
                for pair in config.trading.trading_pairs:
                    try:
                        logger.info(f"\nAnalyzing {pair}...")

                        if positions.has_position(pair):
                            logger.debug(f"Already have position in {pair}, skipping")
                            continue

                        candles = collector.get_candles(pair, "1h", 100)
                        if len(candles) < 50:
                            logger.warning(f"Not enough data for {pair}")
                            continue

                        df_features = features.calculate_features(candles)
                        if df_features.empty:
                            logger.warning(f"Feature calculation failed for {pair}")
                            continue

                        latest_features = df_features.iloc[[-1]]
                        current_price = float(candles["close"].iloc[-1])

                        portfolio_state = {
                            "balance": okx.get_usdt_balance(),
                            "position": 0,
                            "entry_price": 0,
                            "current_price": current_price
                        }

                        decision = ensemble.get_decision(
                            latest_features,
                            portfolio_state,
                            pair
                        )

                        executor.record_decision(decision, executed=False)

                        if decision.action != Action.HOLD:
                            if decision.action == Action.BUY:
                                logger.info(f"BUY signal for {pair} (confidence: {decision.confidence:.2f})")
                                result = executor.execute(decision, current_price)

                                if result.success:
                                    positions.add_position(pair, result.amount, result.price)
                                    risk.record_trade()
                                    logger.info(f"Bought {result.amount} {pair} at ${result.price}")

                            elif decision.action == Action.SELL:
                                logger.debug(f"SELL signal for {pair} but no position")

                    except Exception as e:
                        logger.error(f"Error analyzing {pair}: {e}")
                        health.record_error()
                        continue

                # Auto-retrain if needed
                if retrainer.should_retrain():
                    logger.info("Scheduled model retraining...")
                    try:
                        retrainer.retrain_models(config.trading.trading_pairs)
                    except Exception as e:
                        logger.error(f"Retraining failed: {e}")

                # Cycle complete
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                logger.info(f"\nCycle #{cycle_count} complete in {cycle_duration:.1f}s")
                logger.info(health.get_summary())

                wait_minutes = config.trading.analysis_interval_minutes
                logger.info(f"Waiting {wait_minutes} minutes until next cycle...")
                time.sleep(wait_minutes * 60)

            except KeyboardInterrupt:
                logger.info("\nShutdown requested. Exiting gracefully...")
                break

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                health.record_error()

                if not watchdog._on_error(e):
                    logger.critical("Max retries exceeded. Shutting down.")
                    break

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        logger.error("Dashboard still running - check logs and fix the issue.")
        # Keep running so Railway doesn't restart endlessly
        while True:
            time.sleep(60)

    logger.info("Autonomous trading agent stopped.")


def initial_training(collector, features, xgb_model, rl_agent, trading_pairs):
    """Perform initial model training if models don't exist."""
    import pandas as pd
    from src.data.features import create_target_labels

    logger.info("Performing initial model training...")

    all_data = []
    for pair in trading_pairs:
        logger.info(f"Collecting historical data for {pair}...")
        try:
            df = collector.get_historical_data(pair, days=60, timeframe="1h")
            df_features = features.calculate_features(df)
            df_features["target"] = create_target_labels(df_features)
            df_features["pair"] = pair
            all_data.append(df_features)
        except Exception as e:
            logger.error(f"Failed to collect data for {pair}: {e}")

    if not all_data:
        raise ValueError("No training data available!")

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna()

    logger.info(f"Training on {len(combined_df)} samples...")

    feature_cols = features.get_feature_names()
    X = combined_df[feature_cols]
    y = combined_df["target"]

    logger.info("Training XGBoost model...")
    xgb_model.train(X, y)

    logger.info("Training RL agent...")
    rl_agent.train(combined_df, feature_cols, total_timesteps=50000)

    logger.info("Initial training complete!")


if __name__ == "__main__":
    main()
