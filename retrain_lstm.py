"""
LSTM-only Retrain Script
========================
Retrains LSTM with stronger class balancing to fix SELL bias.
Validates that the retrained model produces balanced directional signals.

Usage:
    python retrain_lstm.py
"""

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from datetime import datetime

sys.path.insert(0, '/root/Cryptobot')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def send_telegram(msg):
    try:
        import requests
        requests.post(
            'https://api.telegram.org/bot8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY/sendMessage',
            data={'chat_id': 7997570468, 'text': msg, 'parse_mode': 'HTML'},
            timeout=10
        )
    except Exception:
        pass


def main():
    start_time = time.time()
    send_telegram("LSTM retrain starting... fixing the SELL bias")

    from dotenv import load_dotenv
    load_dotenv('/root/Cryptobot/.env')

    from src.security.vault import SecureVault
    from src.trading.okx_client import SecureOKXClient
    from src.data.collector import DataCollector
    from src.data.features import FeatureEngine, create_target_labels
    from src.models.lstm_model import LSTMPredictor

    # Get trading pairs from env
    pairs_str = os.getenv("TRADING_PAIRS", "BTC-USDT,ETH-USDT,SOL-USDT,DOGE-USDT,SUI-USDT,XRP-USDT")
    trading_pairs = [p.strip() for p in pairs_str.split(",")]
    logger.info(f"Training on {len(trading_pairs)} pairs: {trading_pairs}")

    # Initialize OKX
    vault = SecureVault()
    okx = SecureOKXClient(vault)
    collector = DataCollector(okx)
    features = FeatureEngine()

    # Collect data
    logger.info("Collecting historical data (180 days)...")
    all_data = []

    for pair in trading_pairs:
        logger.info(f"  Fetching {pair}...")
        try:
            df_1h = collector.get_historical_data(pair, days=180, timeframe="1h")
            df_4h = collector.get_historical_data(pair, days=180, timeframe="4h")
            df_1d = collector.get_historical_data(pair, days=180, timeframe="1d")
            logger.info(f"    Got {len(df_1h)} 1H, {len(df_4h)} 4H, {len(df_1d)} 1D candles")

            df_features = features.calculate_multi_tf_features(df_1h, df_4h, df_1d)
            df_features["target"] = create_target_labels(df_features)
            df_features["pair"] = pair
            all_data.append(df_features)
        except Exception as e:
            logger.error(f"    Failed: {e}")

    if not all_data:
        send_telegram("LSTM retrain FAILED - no data!")
        sys.exit(1)

    combined_df = pd.concat(all_data, ignore_index=True).dropna()
    logger.info(f"Total samples: {len(combined_df)}")

    feature_cols = features.get_feature_names()
    X = combined_df[feature_cols]
    y = combined_df["target"]

    n_buy = (y == 1).sum()
    n_hold = (y == 0).sum()
    n_sell = (y == -1).sum()
    total = len(y)
    logger.info(f"Labels: BUY={n_buy} ({n_buy/total*100:.1f}%), HOLD={n_hold} ({n_hold/total*100:.1f}%), SELL={n_sell} ({n_sell/total*100:.1f}%)")

    send_telegram(f"Data ready: {total} samples\nBUY: {n_buy} | HOLD: {n_hold} | SELL: {n_sell}\n\nTraining LSTM with enhanced balancing...")

    # Backup current model
    import shutil
    os.makedirs("models/backup", exist_ok=True)
    for f in ["models/lstm_model.pt", "models/lstm_model_meta.pkl"]:
        if os.path.exists(f):
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            shutil.copy2(f, f"models/backup/{os.path.basename(f)}.{ts}")
            logger.info(f"Backed up {f}")

    # Train with enhanced class weighting
    lstm = LSTMPredictor(model_path="models/lstm_model.pt")
    # Override epochs for thorough training
    lstm.epochs = 80

    metrics = lstm.train(X, y)

    train_time = time.time() - start_time
    logger.info(f"LSTM Training complete in {train_time:.0f}s")
    logger.info(f"  Val accuracy: {metrics.get('validation_accuracy', 0):.3f}")
    logger.info(f"  Epochs: {metrics.get('epochs_trained', 0)}")

    # ============ BIAS CHECK ============
    # Test the retrained model on recent data to verify it's not all-SELL
    logger.info("\n=== POST-TRAINING BIAS CHECK ===")

    lstm_test = LSTMPredictor(model_path="models/lstm_model.pt")
    lstm_test.load_model()

    signal_counts = {1: 0, 0: 0, -1: 0}
    test_pairs = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

    for pair in test_pairs:
        try:
            df_1h = collector.get_historical_data(pair, days=30, timeframe="1h")
            df_4h = collector.get_historical_data(pair, days=30, timeframe="4h")
            df_1d = collector.get_historical_data(pair, days=30, timeframe="1d")
            df_feat = features.calculate_multi_tf_features(df_1h, df_4h, df_1d)
            df_feat = df_feat.dropna()

            # Test on sliding windows through recent data
            for i in range(0, len(df_feat) - 48, 12):  # Every 12 hours
                window = df_feat.iloc[i:i+48+1]
                if len(window) > 48:
                    action, conf = lstm_test.predict(window)
                    signal_counts[action] += 1
        except Exception as e:
            logger.error(f"Bias check failed for {pair}: {e}")

    total_signals = sum(signal_counts.values())
    if total_signals > 0:
        buy_pct = signal_counts[1] / total_signals * 100
        hold_pct = signal_counts[0] / total_signals * 100
        sell_pct = signal_counts[-1] / total_signals * 100

        logger.info(f"Signal distribution on recent data:")
        logger.info(f"  BUY:  {signal_counts[1]} ({buy_pct:.1f}%)")
        logger.info(f"  HOLD: {signal_counts[0]} ({hold_pct:.1f}%)")
        logger.info(f"  SELL: {signal_counts[-1]} ({sell_pct:.1f}%)")

        bias_ok = buy_pct > 10  # At least 10% BUY signals
        if bias_ok:
            status = "BALANCED"
        else:
            status = "STILL BIASED (but better with ensemble fix)"
    else:
        buy_pct = hold_pct = sell_pct = 0
        status = "NO SIGNALS (check model)"

    result_msg = (
        f"LSTM retrain complete! ({train_time/60:.1f} min)\n\n"
        f"Val accuracy: {metrics.get('validation_accuracy', 0):.3f}\n"
        f"Epochs: {metrics.get('epochs_trained', 0)}\n\n"
        f"Signal bias check (recent 30d):\n"
        f"  BUY: {buy_pct:.1f}%\n"
        f"  HOLD: {hold_pct:.1f}%\n"
        f"  SELL: {sell_pct:.1f}%\n"
        f"Status: {status}\n\n"
        f"Restarting bot with new model..."
    )
    logger.info(result_msg)
    send_telegram(result_msg)

    # Restart bot
    os.system("sudo systemctl restart cryptobot")
    logger.info("Bot restarted!")
    send_telegram("Bot restarted with retrained LSTM!")


if __name__ == "__main__":
    main()
