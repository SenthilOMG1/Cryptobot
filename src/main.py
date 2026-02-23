"""
Autonomous AI Trading Agent V4.1 — Main Orchestrator
=====================================================
Wires up all components and runs the main trading loop.

Each cycle:
  1. Weekend Shield — cap leverage if needed
  2. Sync positions from exchange
  3. Fast Brain — resolve prediction outcomes, update ensemble weights
  4. Position Monitor — stops, take-profits, break-even guards
  5. Pair Analyzer — scan pairs, get signals, execute trades
  6. Succession Rule — auto-rotate weak positions for strong signals
  7. Slow Brain — check if retraining is needed
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

from src.notifications import notify_alert

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_bot(status_dict):
    """Run the trading bot. Updates status_dict for dashboard."""

    logger.info("=" * 50)
    logger.info("TRADING BOT V4.1 STARTING")
    logger.info("=" * 50)

    try:
        # ==================== INIT ====================
        components = _init_components(status_dict)
        config      = components["config"]
        okx         = components["okx"]
        positions   = components["positions"]
        ensemble    = components["ensemble"]
        risk        = components["risk"]
        executor    = components["executor"]
        watchdog    = components["watchdog"]
        retrainer   = components["retrainer"]
        pred_tracker = components["pred_tracker"]
        evaluator   = components["evaluator"]
        trigger_engine = components["trigger_engine"]

        # Core modules
        from src.core import StopLossManager, WeekendShield, PositionMonitor, PairAnalyzer, SuccessionEngine

        sl_mgr = StopLossManager(okx, positions, config)
        shield = WeekendShield(config.futures.leverage)
        monitor = PositionMonitor(
            positions, risk, executor,
            components["collector"], components["features"],
            okx, config, sl_mgr,
        )
        analyzer = PairAnalyzer(
            okx, components["collector"], components["features"],
            positions, ensemble,
            components["trend_filter"], components["regime_detector"],
            components["funding_filter"], components["correlation_filter"],
            executor, risk, pred_tracker, config, sl_mgr,
        )
        succession = SuccessionEngine(
            okx, positions, ensemble,
            components["collector"], components["features"],
            components["trend_filter"], components["regime_detector"],
            executor, risk, config, sl_mgr,
        )

        # Sync stop-losses on startup
        sl_mgr.sync_on_startup()

        logger.info("All components ready!")
        status_dict["state"] = "running"
        status_dict["message"] = "Trading actively (V4.1)"

        # ==================== MAIN LOOP ====================
        pair_cooldowns = {}
        cycle = 0

        while True:
            try:
                cycle += 1

                # 1. Weekend Shield
                shield.apply(config)

                logger.info(f"\n=== CYCLE {cycle} ===")

                # 2. Sync positions
                if config.futures.enabled:
                    positions.sync_futures_from_exchange(config.futures.futures_pairs)
                positions.update_prices()
                portfolio = positions.get_portfolio_summary()
                status_dict["portfolio"] = portfolio.get("total_value", 0)

                # 3. Fast Brain — resolve predictions, update weights
                _run_fast_brain(
                    pred_tracker, evaluator, ensemble, okx,
                    config.trading.trading_pairs,
                )
                logger.info(f"Portfolio: ${portfolio['total_value']:.2f}")

                # Risk pause check
                if risk.should_pause_trading():
                    status_dict["message"] = "Trading paused (risk limit)"
                    time.sleep(3600)
                    continue

                # 4. Position Monitor
                monitor.check_all(pair_cooldowns)

                # Clean expired cooldowns
                now = datetime.now()
                for p in [p for p, t in pair_cooldowns.items() if now >= t]:
                    pair_cooldowns.pop(p)
                    logger.info(f"[{p}] Cooldown expired, pair re-enabled")

                # 5. Pair Analyzer — scan for signals, execute, collect blocked candidates
                succession_candidates = analyzer.scan_pairs(pair_cooldowns)

                # 6. Succession Rule — rotate weak→strong if needed
                succession.evaluate(succession_candidates, pair_cooldowns)

                # 7. Slow Brain — adaptive retrain check
                _run_slow_brain(
                    pred_tracker, trigger_engine, retrainer, evaluator,
                    ensemble, config, status_dict,
                )

                # Periodic cleanup
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


# ==================== HELPERS ====================

def _init_components(status_dict) -> dict:
    """Initialize all trading components. Returns a dict of named components."""
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
    from src.models.ensemble import MetaLearnerEnsemble
    from src.models.trend_filter import TrendFilter
    from src.models.regime_detector import MarketRegimeDetector
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

    # Futures setup
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
        logger.info(
            f"Futures initialized: {leverage_ok}/{len(config.futures.futures_pairs)} pairs set, "
            f"max {config.futures.leverage}x dynamic"
        )
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

    # Models
    xgb_model = XGBoostPredictor(model_path="models/xgboost_model.json")
    rl_agent = RLTradingAgent(model_path="models/rl_agent.zip")
    lstm_model = LSTMPredictor(model_path="models/lstm_model.pt")

    if not xgb_model.is_trained or not rl_agent.is_trained:
        status_dict["message"] = "Training ML models (first run)..."
        logger.info("Training models...")
        initial_training(collector, features, xgb_model, rl_agent, config.trading.trading_pairs)

    ensemble = MetaLearnerEnsemble(
        xgb_model, rl_agent, lstm_model,
        min_confidence=config.trading.min_confidence_to_trade,
    )
    trend_filter = TrendFilter(threshold=0.3)
    regime_detector = MarketRegimeDetector()
    funding_filter = FundingRateFilter(okx)
    correlation_filter = CorrelationFilter(collector)

    logger.info("V4.1 Components loaded:")
    logger.info("  - MetaLearnerEnsemble (3-model: XGB + RL + LSTM)")
    logger.info("  - TrendFilter, MarketRegimeDetector, FundingRateFilter")
    logger.info(f"  - LSTM: {'loaded' if lstm_model.is_trained else 'not trained'}")

    risk_limits = RiskLimits(
        max_position_pct=config.trading.max_position_percent,
        stop_loss_pct=config.trading.stop_loss_percent,
        take_profit_pct=config.trading.take_profit_percent,
        trailing_stop_pct=config.trading.trailing_stop_percent,
        daily_loss_limit_pct=config.trading.daily_loss_limit_percent,
        max_open_positions=config.trading.max_open_positions,
        min_confidence=config.trading.min_confidence_to_trade,
    )
    risk = RiskManager(positions, risk_limits)
    executor = TradeExecutor(okx, risk)
    watchdog = Watchdog(max_retries=10, retry_delay=60)
    retrainer = AutoRetrainer(collector, features, xgb_model, rl_agent, lstm_model)

    pred_tracker = PredictionTracker(db_path="data/trades.db")
    evaluator = EnsembleEvaluator()
    trigger_engine = AdaptiveTriggerEngine()
    logger.info(f"Intelligence module loaded: weights={evaluator.get_weights()}")

    return {
        "config": config, "okx": okx, "collector": collector, "features": features,
        "positions": positions, "ensemble": ensemble,
        "trend_filter": trend_filter, "regime_detector": regime_detector,
        "funding_filter": funding_filter, "correlation_filter": correlation_filter,
        "risk": risk, "executor": executor, "watchdog": watchdog, "retrainer": retrainer,
        "pred_tracker": pred_tracker, "evaluator": evaluator, "trigger_engine": trigger_engine,
    }


def _run_fast_brain(pred_tracker, evaluator, ensemble, okx, trading_pairs):
    """Resolve prediction outcomes and update ensemble weights."""
    try:
        current_prices = {}
        for p in trading_pairs:
            try:
                ticker = okx.get_ticker(p)
                if ticker:
                    current_prices[p] = float(ticker.get("last", 0))
            except Exception:
                pass
        pred_tracker.resolve_pending_outcomes(current_prices)

        if pred_tracker.has_new_outcomes(min_count=5):
            recent = pred_tracker.get_recent_outcomes(window=30)
            if len(recent) >= 15:
                new_weights = evaluator.update_weights(recent)
                ensemble.set_weights(new_weights)
                pred_tracker.reset_outcome_counter()
    except Exception as e:
        logger.debug(f"Intelligence update: {e}")


def _run_slow_brain(pred_tracker, trigger_engine, retrainer, evaluator,
                    ensemble, config, status_dict):
    """Check if model retraining is needed (adaptive or calendar-based)."""
    try:
        perf_metrics = pred_tracker.get_performance_metrics()
        trigger_engine.update_metrics(perf_metrics, {})

        if trigger_engine.should_retrain():
            trigger_info = trigger_engine.get_trigger_info()
            severity = trigger_info.get("severity", "medium")
            reason = trigger_info.get("reason", "unknown")
            logger.info(f"RETRAIN TRIGGERED: {reason} (severity={severity})")
            notify_alert(f"Retrain triggered: {reason} (severity={severity})", f"Metrics: {perf_metrics}")

            status_dict["message"] = f"Retraining ({severity})..."
            retrainer.retrain_models(config.trading.trading_pairs)
            trigger_engine.record_retrain_completed(severity)
            ensemble.set_weights(evaluator.get_weights())
            status_dict["message"] = "Trading actively (V4.1)"

        elif retrainer.should_retrain():
            status_dict["message"] = "Scheduled retraining..."
            retrainer.retrain_models(config.trading.trading_pairs)
            trigger_engine.record_retrain_completed("medium")
            status_dict["message"] = "Trading actively (V4.1)"
    except Exception as e:
        logger.debug(f"Slow brain check: {e}")


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
    status = {"state": "starting", "message": "", "portfolio": 0}
    run_bot(status)
