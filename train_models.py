"""
Full Model Training Script
===========================
Trains all 3 models: XGBoost, LSTM, RL Agent.
Sends progress updates via Telegram.

Usage:
    python train_models.py
"""

import os
import sys
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def send_telegram(msg):
    """Send training status update via Telegram."""
    try:
        import requests
        requests.post(
            'https://api.telegram.org/bot8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY/sendMessage',
            data={'chat_id': 7997570468, 'text': msg},
            timeout=10
        )
    except Exception:
        pass


def main():
    """Train all 3 models."""

    logger.info("=" * 60)
    logger.info("FULL MODEL TRAINING - ALL 3 MODELS")
    logger.info("=" * 60)
    start_time = time.time()

    send_telegram("Training started! All 3 models being retrained from scratch.\n\nFixes applied:\n- RL: higher entropy (0.15), removed label bias, gentler penalties\n- LSTM: no shuffle, higher dropout, better regularization\n- Labels: adaptive threshold for balanced BUY/SELL distribution\n\nThis will take ~45-60 min. I'll update you as each model finishes.")

    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/backup", exist_ok=True)

    # Backup existing models
    import shutil
    for f in ["models/xgboost_model.json", "models/rl_agent.zip", "models/lstm_model.pt"]:
        if os.path.exists(f):
            backup = f"models/backup/{os.path.basename(f)}.bak_{datetime.now().strftime('%Y%m%d_%H%M')}"
            shutil.copy2(f, backup)
            logger.info(f"Backed up {f} -> {backup}")

    # Import components
    from src.config import get_config
    from src.security.vault import SecureVault
    from src.trading.okx_client import SecureOKXClient
    from src.data.collector import DataCollector
    from src.data.features import FeatureEngine, create_target_labels
    from src.models.xgboost_model import XGBoostPredictor
    from src.models.rl_agent import RLTradingAgent
    from src.models.lstm_model import LSTMPredictor
    import pandas as pd

    # Load config
    config = get_config()
    trading_pairs = config.trading.trading_pairs

    logger.info(f"Training for pairs: {trading_pairs}")

    # Initialize vault and OKX client
    logger.info("Connecting to OKX...")
    vault = SecureVault()
    okx = SecureOKXClient(vault, demo_mode=config.exchange.demo_mode)

    if not okx.test_connection():
        logger.error("Failed to connect to OKX!")
        send_telegram("Training FAILED - can't connect to OKX!")
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
            df_1h = collector.get_historical_data(pair, days=180, timeframe="1h")
            logger.info(f"    Got {len(df_1h)} 1H candles")

            df_4h = collector.get_historical_data(pair, days=180, timeframe="4h")
            df_1d = collector.get_historical_data(pair, days=180, timeframe="1d")
            logger.info(f"    Got {len(df_4h)} 4H + {len(df_1d)} 1D candles")

            df_features = features.calculate_multi_tf_features(df_1h, df_4h, df_1d)
            logger.info(f"    Calculated {len(features.get_feature_names())} features")

            df_features["target"] = create_target_labels(df_features)
            df_features["pair"] = pair
            all_data.append(df_features)

        except Exception as e:
            logger.error(f"    Failed: {e}")

    if not all_data:
        logger.error("No data collected! Check your API connection.")
        send_telegram("Training FAILED - no data collected!")
        sys.exit(1)

    # Combine data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna()
    logger.info(f"Total training samples: {len(combined_df)}")

    # Prepare features
    feature_cols = features.get_feature_names()
    X = combined_df[feature_cols]
    y = combined_df["target"]

    # Show class distribution (important — should be roughly balanced now)
    n_buy = (y == 1).sum()
    n_hold = (y == 0).sum()
    n_sell = (y == -1).sum()
    total = len(y)
    logger.info("Target distribution (with adaptive threshold):")
    logger.info(f"  BUY (1):  {n_buy} ({n_buy/total*100:.1f}%)")
    logger.info(f"  HOLD (0): {n_hold} ({n_hold/total*100:.1f}%)")
    logger.info(f"  SELL (-1): {n_sell} ({n_sell/total*100:.1f}%)")

    data_time = time.time() - start_time
    send_telegram(f"Data collected! {len(combined_df)} samples across {len(trading_pairs)} pairs.\n\nLabel distribution:\n- BUY: {n_buy} ({n_buy/total*100:.1f}%)\n- HOLD: {n_hold} ({n_hold/total*100:.1f}%)\n- SELL: {n_sell} ({n_sell/total*100:.1f}%)\n\nNow training XGBoost...")

    # ============ TRAIN XGBOOST ============
    logger.info("\n" + "=" * 40)
    logger.info("Training XGBoost Model")
    logger.info("=" * 40)

    t0 = time.time()
    xgb_model = XGBoostPredictor(model_path="models/xgboost_model.json")
    xgb_metrics = xgb_model.train(X, y)
    xgb_time = time.time() - t0

    logger.info(f"XGBoost Results:")
    logger.info(f"  Train Accuracy: {xgb_metrics['train_accuracy']:.3f}")
    logger.info(f"  Val Accuracy:   {xgb_metrics['validation_accuracy']:.3f}")

    importance = xgb_model.get_feature_importance(top_n=10)
    top_features = ", ".join(importance['feature'].head(5).tolist())

    send_telegram(f"XGBoost done! ({xgb_time:.0f}s)\n- Train acc: {xgb_metrics['train_accuracy']:.3f}\n- Val acc: {xgb_metrics['validation_accuracy']:.3f}\n- Top features: {top_features}\n\nNow training LSTM...")

    # ============ TRAIN LSTM ============
    logger.info("\n" + "=" * 40)
    logger.info("Training LSTM Model")
    logger.info("=" * 40)

    t0 = time.time()
    lstm_model = LSTMPredictor(model_path="models/lstm_model.pt")
    lstm_metrics = lstm_model.train(X, y)
    lstm_time = time.time() - t0

    logger.info(f"LSTM Results:")
    logger.info(f"  Train Accuracy: {lstm_metrics.get('train_accuracy', 0):.3f}")
    logger.info(f"  Val Accuracy:   {lstm_metrics.get('validation_accuracy', 0):.3f}")
    logger.info(f"  Epochs trained: {lstm_metrics.get('epochs_trained', 0)}")

    send_telegram(f"LSTM done! ({lstm_time:.0f}s)\n- Train acc: {lstm_metrics.get('train_accuracy', 0):.3f}\n- Val acc: {lstm_metrics.get('validation_accuracy', 0):.3f}\n- Epochs: {lstm_metrics.get('epochs_trained', 0)}\n\nNow training RL Agent (the big one)...")

    # ============ TRAIN RL AGENT ============
    logger.info("\n" + "=" * 40)
    logger.info("Training RL Agent (750K timesteps)")
    logger.info("=" * 40)

    t0 = time.time()
    # Delete old model to force fresh training (no resuming from collapsed policy)
    for old_file in ["models/rl_agent.zip", "models/rl_agent_meta.pkl"]:
        if os.path.exists(old_file):
            os.remove(old_file)
            logger.info(f"Removed old {old_file} for fresh training")

    rl_agent = RLTradingAgent()  # No model_path = fresh init
    rl_agent.model_path = "models/rl_agent.zip"
    rl_metrics = rl_agent.train(
        combined_df,
        feature_cols,
        total_timesteps=750000,  # More timesteps for better exploration
        initial_balance=1000
    )
    rl_time = time.time() - t0

    logger.info(f"RL Agent Results:")
    logger.info(f"  Total Return: {rl_metrics['total_return']:.2%}")
    logger.info(f"  Final Portfolio: ${rl_metrics['final_portfolio_value']:.2f}")
    logger.info(f"  Total Trades: {rl_metrics['total_trades']}")

    total_time = time.time() - start_time

    send_telegram(f"RL Agent done! ({rl_time:.0f}s)\n- Return: {rl_metrics['total_return']:.2%}\n- Final portfolio: ${rl_metrics['final_portfolio_value']:.2f}\n- Trades: {rl_metrics['total_trades']}\n\n{'='*30}\nALL 3 MODELS TRAINED!\nTotal time: {total_time/60:.1f} min\n\nModels saved. Restart the bot to use new models.\n\nRestarting bot now...")

    # ============ RESTART BOT ============
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE! Restarting bot...")
    logger.info("=" * 60)

    os.system("systemctl restart cryptobot")
    logger.info("Bot restarted with new models!")

    send_telegram("Bot restarted with new models! Monitoring for first signals...")


if __name__ == "__main__":
    main()
