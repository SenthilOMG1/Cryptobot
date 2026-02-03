"""
Initial Model Training Script
=============================
Run this once to train the ML models before starting the bot.

Usage:
    python train_models.py
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Train models for the first time."""

    logger.info("=" * 60)
    logger.info("INITIAL MODEL TRAINING")
    logger.info("=" * 60)

    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Import components
    from src.config import get_config
    from src.security.vault import SecureVault
    from src.trading.okx_client import SecureOKXClient
    from src.data.collector import DataCollector
    from src.data.features import FeatureEngine, create_target_labels
    from src.models.xgboost_model import XGBoostPredictor
    from src.models.rl_agent import RLTradingAgent
    import pandas as pd

    # Load config
    config = get_config()
    trading_pairs = config.trading.trading_pairs

    logger.info(f"Training for pairs: {trading_pairs}")

    # Initialize vault and OKX client
    logger.info("Connecting to OKX...")
    vault = SecureVault()
    okx = SecureOKXClient(vault, demo_mode=True)  # Use demo for data fetching

    if not okx.test_connection():
        logger.error("Failed to connect to OKX!")
        sys.exit(1)

    # Initialize data collection
    collector = DataCollector(okx)
    features = FeatureEngine()

    # Collect historical data
    logger.info("Collecting historical data...")
    all_data = []

    for pair in trading_pairs:
        logger.info(f"  Fetching {pair}...")
        try:
            # Get 60 days of hourly data
            df = collector.get_historical_data(pair, days=60, timeframe="1h")
            logger.info(f"    Got {len(df)} candles")

            # Calculate features
            df_features = features.calculate_features(df)
            logger.info(f"    Calculated {len(features.get_feature_names())} features")

            # Create labels
            df_features["target"] = create_target_labels(df_features)
            df_features["pair"] = pair

            all_data.append(df_features)

        except Exception as e:
            logger.error(f"    Failed: {e}")

    if not all_data:
        logger.error("No data collected! Check your API connection.")
        sys.exit(1)

    # Combine data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna()
    logger.info(f"Total training samples: {len(combined_df)}")

    # Prepare features
    feature_cols = features.get_feature_names()
    X = combined_df[feature_cols]
    y = combined_df["target"]

    # Show class distribution
    logger.info("Target distribution:")
    logger.info(f"  BUY (1):  {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    logger.info(f"  HOLD (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    logger.info(f"  SELL (-1): {(y == -1).sum()} ({(y == -1).mean()*100:.1f}%)")

    # Train XGBoost
    logger.info("\n" + "=" * 40)
    logger.info("Training XGBoost Model")
    logger.info("=" * 40)

    xgb_model = XGBoostPredictor(model_path="models/xgboost_model.json")
    xgb_metrics = xgb_model.train(X, y)

    logger.info(f"XGBoost Results:")
    logger.info(f"  Train Accuracy: {xgb_metrics['train_accuracy']:.3f}")
    logger.info(f"  Val Accuracy:   {xgb_metrics['validation_accuracy']:.3f}")

    # Show feature importance
    logger.info("\nTop 10 Important Features:")
    importance = xgb_model.get_feature_importance(top_n=10)
    for _, row in importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Train RL Agent
    logger.info("\n" + "=" * 40)
    logger.info("Training RL Agent (this may take a while...)")
    logger.info("=" * 40)

    rl_agent = RLTradingAgent(model_path="models/rl_agent.zip")
    rl_metrics = rl_agent.train(
        combined_df,
        feature_cols,
        total_timesteps=50000,
        initial_balance=1000
    )

    logger.info(f"RL Agent Results:")
    logger.info(f"  Total Return: {rl_metrics['total_return']:.2%}")
    logger.info(f"  Final Portfolio: ${rl_metrics['final_portfolio_value']:.2f}")
    logger.info(f"  Total Trades: {rl_metrics['total_trades']}")

    # Done!
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info("Models saved to:")
    logger.info("  - models/xgboost_model.json")
    logger.info("  - models/rl_agent.zip")
    logger.info("\nYou can now start the trading bot with:")
    logger.info("  python -m src.main")


if __name__ == "__main__":
    main()
