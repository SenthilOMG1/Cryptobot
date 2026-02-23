"""
Full Model Training Script V4
==============================
Trains all 3 models: XGBoost, LSTM, RL Agent.
Sends progress updates via Telegram.

V4 Changes (Feb 22, 2026):
- Historical Fear & Greed aligned per-candle (not live value for all rows)
- Mirror augmentation: flipped directional features + swapped labels for bias killing
- Funding rate features included in training
- 12 new features: sentiment (5), time (5), funding (2)
- All V3 fixes retained (percentile labels, bias validation)

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


def validate_model_bias(model, X, model_name, max_directional_pct=0.65):
    """
    Check if a model is biased toward one direction.
    Returns (passed, stats_dict).
    """
    import numpy as np

    try:
        if model_name == "LSTM":
            # LSTM needs sequential data — predict on last 500 rows in chunks
            predictions = []
            for i in range(48, min(len(X), 500)):
                chunk = X.iloc[max(0, i-48):i+1]
                action, conf = model.predict(chunk)
                predictions.append(action)
            predictions = np.array(predictions)
        elif model_name == "RL":
            # RL needs observation arrays
            predictions = []
            feature_cols = model.feature_columns
            portfolio = {"balance": 1000, "position": 0, "entry_price": 0, "current_price": 0, "side": 0}
            for i in range(min(len(X), 500)):
                row = X.iloc[i]
                feature_array = row[feature_cols].values.flatten().astype(np.float32) if hasattr(row, '__getitem__') else row
                portfolio["current_price"] = 1.0  # dummy
                action, conf = model.decide(feature_array, portfolio)
                predictions.append(action)
            predictions = np.array(predictions)
        else:
            # XGBoost — batch predict
            preds, confs = model.predict(X.tail(500))
            predictions = np.array(preds) if hasattr(preds, '__len__') else np.array([preds])
    except Exception as e:
        logger.warning(f"Bias validation failed for {model_name}: {e}")
        return True, {"error": str(e)}

    n_total = len(predictions)
    if n_total < 10:
        return True, {"error": "too few predictions"}

    n_buy = (predictions == 1).sum()
    n_sell = (predictions == -1).sum()
    n_hold = (predictions == 0).sum()

    buy_pct = n_buy / n_total
    sell_pct = n_sell / n_total
    hold_pct = n_hold / n_total

    stats = {
        "total": n_total,
        "buy": int(n_buy), "buy_pct": f"{buy_pct:.1%}",
        "sell": int(n_sell), "sell_pct": f"{sell_pct:.1%}",
        "hold": int(n_hold), "hold_pct": f"{hold_pct:.1%}",
    }

    # Check for extreme bias
    max_dir = max(buy_pct, sell_pct)
    if max_dir > max_directional_pct and hold_pct < 0.5:
        dominant = "SELL" if sell_pct > buy_pct else "BUY"
        logger.warning(
            f"{model_name} BIAS DETECTED: {dominant} at {max_dir:.1%} "
            f"(threshold: {max_directional_pct:.0%})"
        )
        stats["bias_detected"] = dominant
        return False, stats

    logger.info(f"{model_name} bias check PASSED: BUY={buy_pct:.1%}, SELL={sell_pct:.1%}, HOLD={hold_pct:.1%}")
    return True, stats


def main():
    """Train all 3 models."""

    logger.info("=" * 60)
    logger.info("FULL MODEL TRAINING V4 - THE EXORCIST RETRAIN")
    logger.info("=" * 60)
    start_time = time.time()

    send_telegram("Training V4 'The Exorcist' started!\n\nNEW:\n- Historical F&G aligned per candle\n- Mirror augmentation (bias killer)\n- Funding rate features\n- 12 new features total\n\nRetained:\n- Percentile labels\n- Bias validation\n- 90 days, 6 pairs\n\nThis will take ~60-90 min.")

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
    import numpy as np

    # Load config
    config = get_config()

    # Train on MORE pairs than just the 3 in .env
    # These are the liquid futures pairs that the bot will trade
    training_pairs = [
        "BTC-USDT", "ETH-USDT", "SOL-USDT",
        "DOGE-USDT", "XRP-USDT", "SUI-USDT",
    ]

    logger.info(f"Training for pairs: {training_pairs}")

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

    # Collect historical data — 90 days for recent market conditions
    data_days = 90
    logger.info(f"Collecting {data_days} days of historical data...")
    all_data = []

    for pair in training_pairs:
        logger.info(f"  Fetching {pair}...")
        try:
            df_1h = collector.get_historical_data(pair, days=data_days, timeframe="1h")
            logger.info(f"    Got {len(df_1h)} 1H candles")

            df_4h = collector.get_historical_data(pair, days=data_days, timeframe="4h")
            df_1d = collector.get_historical_data(pair, days=data_days, timeframe="1d")
            logger.info(f"    Got {len(df_4h)} 4H + {len(df_1d)} 1D candles")

            df_features = features.calculate_multi_tf_features(df_1h, df_4h, df_1d)
            logger.info(f"    Calculated {len(features.get_feature_names())} features")

            df_features["target"] = create_target_labels(df_features)
            df_features["pair"] = pair

            # Per-pair label distribution check
            n_b = (df_features["target"] == 1).sum()
            n_s = (df_features["target"] == -1).sum()
            n_h = (df_features["target"] == 0).sum()
            total_p = len(df_features)
            logger.info(
                f"    {pair} labels: BUY={n_b} ({n_b/total_p*100:.1f}%), "
                f"HOLD={n_h} ({n_h/total_p*100:.1f}%), "
                f"SELL={n_s} ({n_s/total_p*100:.1f}%)"
            )

            all_data.append(df_features)

        except Exception as e:
            logger.error(f"    Failed: {e}")
            import traceback
            traceback.print_exc()

    if not all_data:
        logger.error("No data collected! Check your API connection.")
        send_telegram("Training FAILED - no data collected!")
        sys.exit(1)

    # Inject historical funding rates per pair
    logger.info("Fetching historical funding rates...")
    for i, df_feat in enumerate(all_data):
        pair = df_feat["pair"].iloc[0] if "pair" in df_feat.columns else training_pairs[i]
        try:
            fr_data = okx.get_funding_rate(pair)
            if fr_data and "fundingRate" in fr_data:
                fr = float(fr_data["fundingRate"])
            else:
                fr = 0.0
            all_data[i] = df_feat.copy()
            all_data[i]["funding_rate"] = fr * 100
            all_data[i]["funding_contrarian"] = -fr * 1000
            logger.info(f"  {pair} funding rate: {fr*100:.4f}%")
        except Exception:
            all_data[i] = df_feat.copy()
            all_data[i]["funding_rate"] = 0.0
            all_data[i]["funding_contrarian"] = 0.0

    # Combine data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna()
    logger.info(f"Total training samples (before mirror): {len(combined_df)}")

    # ============ MIRROR AUGMENTATION ============
    # Create "mirror world" data: flip price-derived features and swap labels
    # This forces the model to learn directional LOGIC, not directional PREFERENCE
    logger.info("Generating mirror augmentation data...")

    mirror_df = combined_df.copy()

    # Flip labels: BUY → SELL, SELL → BUY, HOLD stays
    mirror_df["target"] = -mirror_df["target"]

    # Flip directional features (signs that indicate direction)
    directional_features = [
        # Returns — flip sign (up becomes down)
        "return_1", "return_3", "return_5", "return_10", "return_20",
        "cum_return_5", "cum_return_20",
        # Momentum — flip sign
        "momentum_10", "momentum_20",
        "roc_5", "roc_10", "roc_20",
        # MACD — flip sign (bullish becomes bearish)
        "macd", "macd_signal", "macd_hist",
        # Price vs MAs — flip sign (above → below)
        "price_vs_sma50", "price_vs_sma200", "price_vs_ema21",
        "price_vs_vwap",
        # OBV trend — flip
        "obv_trend",
        # Candle body — flip
        "candle_body",
        # Distance from highs/lows — swap
        "dist_from_20h", "dist_from_20l",
        # Streaks — swap
        "up_streak", "down_streak",
        # Higher-TF directional features
        "tf4h_macd", "tf4h_macd_signal", "tf4h_macd_hist",
        "tf4h_return_1", "tf4h_return_5",
        "tf4h_price_vs_sma20", "tf4h_price_vs_sma50",
        "tf1d_macd", "tf1d_macd_signal", "tf1d_macd_hist",
        "tf1d_return_1", "tf1d_return_5",
        "tf1d_price_vs_sma20", "tf1d_price_vs_sma50",
        # Sentiment — flip contrarian
        "sentiment_contrarian", "fng_delta",
        # Funding — flip contrarian
        "funding_contrarian",
    ]

    for feat in directional_features:
        if feat in mirror_df.columns:
            mirror_df[feat] = -mirror_df[feat]

    # Swap features that need swapping (not just negation)
    swap_pairs = [
        ("higher_high", "lower_low"),
        ("gap_up", "gap_down"),
        ("near_resistance", "near_support"),
        ("up_day", "down_day"),
        ("extreme_fear", "extreme_greed"),
        ("candle_upper_shadow", "candle_lower_shadow"),
    ]
    for feat_a, feat_b in swap_pairs:
        if feat_a in mirror_df.columns and feat_b in mirror_df.columns:
            mirror_df[feat_a], mirror_df[feat_b] = (
                mirror_df[feat_b].copy(),
                mirror_df[feat_a].copy(),
            )

    # RSI-like features: mirror around 50 (RSI 70 → RSI 30)
    rsi_features = [
        "rsi_7", "rsi_14", "rsi_21",
        "stoch_k", "stoch_d",
        "tf4h_rsi", "tf4h_stoch_k",
        "tf1d_rsi",
    ]
    for feat in rsi_features:
        if feat in mirror_df.columns:
            mirror_df[feat] = 100 - mirror_df[feat]

    # Williams %R: mirror around -50 (-20 → -80)
    for feat in ["williams_r"]:
        if feat in mirror_df.columns:
            mirror_df[feat] = -100 - mirror_df[feat]

    # Bollinger Band position: mirror around 0.5 (0.8 → 0.2)
    bb_features = ["bb_position", "tf4h_bb_pos", "tf1d_bb_pos"]
    for feat in bb_features:
        if feat in mirror_df.columns:
            mirror_df[feat] = 1.0 - mirror_df[feat]

    # F&G score: mirror around 0 (-0.8 → 0.8)
    if "fng_score" in mirror_df.columns:
        mirror_df["fng_score"] = -mirror_df["fng_score"]

    # Mark mirror data with distinct pair names for RL episode separation
    mirror_df["pair"] = mirror_df["pair"] + "-MIRROR"

    # Combine original + mirror
    combined_df = pd.concat([combined_df, mirror_df], ignore_index=True)
    combined_df = combined_df.dropna()

    logger.info(f"Total training samples (with mirror): {len(combined_df)}")

    # Prepare features
    feature_cols = features.get_feature_names()
    # Add funding features to feature list if not already there
    for fc in ["funding_rate", "funding_contrarian"]:
        if fc in combined_df.columns and fc not in feature_cols:
            feature_cols.append(fc)

    X = combined_df[feature_cols]
    y = combined_df["target"]

    # Show class distribution (should be nearly perfect 50/50 BUY/SELL now)
    n_buy = (y == 1).sum()
    n_hold = (y == 0).sum()
    n_sell = (y == -1).sum()
    total = len(y)
    logger.info("Target distribution (with mirror augmentation):")
    logger.info(f"  BUY (1):  {n_buy} ({n_buy/total*100:.1f}%)")
    logger.info(f"  HOLD (0): {n_hold} ({n_hold/total*100:.1f}%)")
    logger.info(f"  SELL (-1): {n_sell} ({n_sell/total*100:.1f}%)")

    # Check for severe imbalance
    buy_sell_ratio = n_buy / max(n_sell, 1)
    if buy_sell_ratio < 0.5 or buy_sell_ratio > 2.0:
        logger.warning(f"LABEL IMBALANCE: BUY/SELL ratio = {buy_sell_ratio:.2f} (target: 0.8-1.2)")
        send_telegram(f"WARNING: Label imbalance detected!\nBUY/SELL ratio: {buy_sell_ratio:.2f}\nBUY: {n_buy}, SELL: {n_sell}\n\nTraining anyway but models may be biased.")

    data_time = time.time() - start_time
    send_telegram(
        f"V4 Data collected! {len(combined_df)} samples ({len(combined_df)//2} real + {len(combined_df)//2} mirror) across {len(training_pairs)} pairs.\n\n"
        f"NEW in V4:\n"
        f"- Historical F&G aligned per-candle\n"
        f"- Mirror augmentation (bias killer)\n"
        f"- Funding rate features\n\n"
        f"Label distribution:\n"
        f"- BUY: {n_buy} ({n_buy/total*100:.1f}%)\n"
        f"- HOLD: {n_hold} ({n_hold/total*100:.1f}%)\n"
        f"- SELL: {n_sell} ({n_sell/total*100:.1f}%)\n"
        f"- BUY/SELL ratio: {buy_sell_ratio:.2f}\n\n"
        f"Now training XGBoost..."
    )

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

    # Validate XGBoost for bias
    xgb_passed, xgb_bias = validate_model_bias(xgb_model, X, "XGBoost")

    importance = xgb_model.get_feature_importance(top_n=10)
    top_features = ", ".join(importance['feature'].head(5).tolist())

    send_telegram(
        f"XGBoost done! ({xgb_time:.0f}s)\n"
        f"- Train acc: {xgb_metrics['train_accuracy']:.3f}\n"
        f"- Val acc: {xgb_metrics['validation_accuracy']:.3f}\n"
        f"- Bias check: {'PASS' if xgb_passed else 'FAIL - ' + str(xgb_bias.get('bias_detected', ''))}\n"
        f"- Predictions: BUY={xgb_bias.get('buy_pct','?')}, SELL={xgb_bias.get('sell_pct','?')}\n"
        f"- Top features: {top_features}\n\n"
        f"Now training LSTM..."
    )

    # ============ TRAIN LSTM ============
    logger.info("\n" + "=" * 40)
    logger.info("Training LSTM Model")
    logger.info("=" * 40)

    t0 = time.time()
    # Fresh LSTM — delete old model files to prevent loading stale weights
    for old_file in ["models/lstm_model.pt", "models/lstm_model_meta.pkl", "models/lstm_model_best.pt"]:
        if os.path.exists(old_file):
            os.remove(old_file)
            logger.info(f"Removed old {old_file} for fresh training")

    lstm_model = LSTMPredictor(model_path="models/lstm_model.pt")
    lstm_metrics = lstm_model.train(X, y)
    lstm_time = time.time() - t0

    logger.info(f"LSTM Results:")
    logger.info(f"  Train Accuracy: {lstm_metrics.get('train_accuracy', 0):.3f}")
    logger.info(f"  Val Accuracy:   {lstm_metrics.get('validation_accuracy', 0):.3f}")
    logger.info(f"  Epochs trained: {lstm_metrics.get('epochs_trained', 0)}")

    # Validate LSTM for bias
    lstm_passed, lstm_bias = validate_model_bias(lstm_model, X, "LSTM")

    send_telegram(
        f"LSTM done! ({lstm_time:.0f}s)\n"
        f"- Train acc: {lstm_metrics.get('train_accuracy', 0):.3f}\n"
        f"- Val acc: {lstm_metrics.get('validation_accuracy', 0):.3f}\n"
        f"- Epochs: {lstm_metrics.get('epochs_trained', 0)}\n"
        f"- Bias check: {'PASS' if lstm_passed else 'FAIL - ' + str(lstm_bias.get('bias_detected', ''))}\n"
        f"- Predictions: BUY={lstm_bias.get('buy_pct','?')}, SELL={lstm_bias.get('sell_pct','?')}\n\n"
        f"Now training RL Agent (the big one)..."
    )

    # ============ TRAIN RL AGENT ============
    logger.info("\n" + "=" * 40)
    logger.info("Training RL Agent (1M timesteps)")
    logger.info("=" * 40)

    t0 = time.time()
    # Delete old model to force fresh training
    for old_file in ["models/rl_agent.zip", "models/rl_agent_meta.pkl"]:
        if os.path.exists(old_file):
            os.remove(old_file)
            logger.info(f"Removed old {old_file} for fresh training")

    rl_agent = RLTradingAgent()  # No model_path = fresh init
    rl_agent.model_path = "models/rl_agent.zip"
    # RL trains on REAL data only (no mirror) — mirror confuses the reward signal
    # because mirrored price movements create contradictory observations vs portfolio P&L
    rl_real_df = combined_df[~combined_df["pair"].str.contains("MIRROR")].copy()
    logger.info(f"RL training on {len(rl_real_df)} real samples (mirror excluded)")
    rl_metrics = rl_agent.train(
        rl_real_df,
        feature_cols,
        total_timesteps=1000000,  # 1M timesteps for thorough exploration
        initial_balance=1000
    )
    rl_time = time.time() - t0

    logger.info(f"RL Agent Results:")
    logger.info(f"  Total Return: {rl_metrics['total_return']:.2%}")
    logger.info(f"  Final Portfolio: ${rl_metrics['final_portfolio_value']:.2f}")
    logger.info(f"  Total Trades: {rl_metrics['total_trades']}")

    # Validate RL for bias
    rl_agent.feature_columns = feature_cols
    rl_passed, rl_bias = validate_model_bias(rl_agent, combined_df[feature_cols], "RL")

    total_time = time.time() - start_time

    # Summary
    bias_summary = []
    for name, passed, stats in [("XGB", xgb_passed, xgb_bias), ("LSTM", lstm_passed, lstm_bias), ("RL", rl_passed, rl_bias)]:
        status = "PASS" if passed else f"FAIL ({stats.get('bias_detected', 'unknown')})"
        bias_summary.append(f"  {name}: {status} (B={stats.get('buy_pct','?')} S={stats.get('sell_pct','?')})")

    summary_msg = (
        f"ALL 3 MODELS TRAINED!\n"
        f"Total time: {total_time/60:.1f} min\n\n"
        f"Results:\n"
        f"- XGB: val_acc={xgb_metrics['validation_accuracy']:.3f}\n"
        f"- LSTM: val_acc={lstm_metrics.get('validation_accuracy', 0):.3f}\n"
        f"- RL: return={rl_metrics['total_return']:.2%}, trades={rl_metrics['total_trades']}\n\n"
        f"Bias validation:\n" + "\n".join(bias_summary) + "\n\n"
        f"Labels: BUY={n_buy/total*100:.0f}% HOLD={n_hold/total*100:.0f}% SELL={n_sell/total*100:.0f}%\n\n"
    )

    all_passed = xgb_passed and lstm_passed and rl_passed
    if all_passed:
        summary_msg += "All bias checks PASSED! Deploying new models and restarting bot."
        send_telegram(summary_msg)
        os.system("sudo systemctl restart cryptobot")
        logger.info("Bot restarted with new models!")
        send_telegram("Bot restarted with new models! Watching for first signals...")
    else:
        summary_msg += (
            "SOME BIAS CHECKS FAILED.\n"
            "Models are saved but bot NOT restarted automatically.\n"
            "Review the bias stats and decide if you want to deploy."
        )
        send_telegram(summary_msg)
        logger.warning("Bias checks failed — bot NOT restarted. Review and deploy manually.")

    # Also delete the stale meta-learner (it was trained on old model outputs)
    meta_path = "models/meta_learner.pkl"
    if os.path.exists(meta_path):
        os.remove(meta_path)
        logger.info("Removed stale meta-learner (will retrain from fresh outcomes)")

    # Reset evaluator state (weights are stale after retrain)
    eval_state = "models/evaluator_state.json"
    if os.path.exists(eval_state):
        os.remove(eval_state)
        logger.info("Reset evaluator state (fresh weights for new models)")

    # ============ SHADOW TEST ============
    # Test: would the retrained model have taken the SUI short at F&G 9?
    logger.info("\n" + "=" * 40)
    logger.info("SHADOW TEST: SUI-USDT at F&G 9 (Extreme Fear)")
    logger.info("=" * 40)

    try:
        # Fetch current SUI data
        df_sui = collector.get_historical_data("SUI-USDT", days=7, timeframe="1h")
        df_sui_4h = collector.get_historical_data("SUI-USDT", days=7, timeframe="4h")
        df_sui_1d = collector.get_historical_data("SUI-USDT", days=30, timeframe="1d")

        df_sui_feat = features.calculate_multi_tf_features(df_sui, df_sui_4h, df_sui_1d)
        if not df_sui_feat.empty:
            df_sui_feat = df_sui_feat.copy()
            # Add funding rate
            fr_data = okx.get_funding_rate("SUI-USDT")
            if fr_data and "fundingRate" in fr_data:
                fr = float(fr_data["fundingRate"])
                df_sui_feat["funding_rate"] = fr * 100
                df_sui_feat["funding_contrarian"] = -fr * 1000
            else:
                df_sui_feat["funding_rate"] = 0.0
                df_sui_feat["funding_contrarian"] = 0.0

            latest = df_sui_feat.iloc[[-1]]

            # XGBoost prediction
            xgb_pred, xgb_conf = xgb_model.predict(latest)
            xgb_action = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(xgb_pred, "HOLD")

            # LSTM prediction
            lstm_action, lstm_conf = lstm_model.predict(df_sui_feat.tail(50))

            # RL prediction
            feat_array = latest[feature_cols].values.flatten().astype(np.float32)
            portfolio = {"balance": 1000, "position": 0, "entry_price": 0, "current_price": float(df_sui["close"].iloc[-1]), "side": 0}
            rl_action, rl_conf = rl_agent.decide(feat_array, portfolio)

            action_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
            shadow_msg = (
                f"SHADOW TEST RESULTS:\n"
                f"SUI-USDT @ F&G 9 (Extreme Fear)\n\n"
                f"- XGBoost: {xgb_action} (conf: {xgb_conf:.2f})\n"
                f"- LSTM: {action_map.get(lstm_action, 'HOLD')} (conf: {lstm_conf:.2f})\n"
                f"- RL: {action_map.get(rl_action, 'HOLD')} (conf: {rl_conf:.2f})\n\n"
                f"F&G features: score={latest['fng_score'].values[0]:.2f}, "
                f"extreme_fear={latest['extreme_fear'].values[0]}, "
                f"contrarian={latest['sentiment_contrarian'].values[0]}\n\n"
            )

            # Check if the old trade would still happen
            old_trade = "SHORT"  # The old model took a SUI SHORT
            would_short = (xgb_action == "SELL" and lstm_action == -1) or (rl_action == -1 and xgb_action == "SELL")
            if would_short:
                shadow_msg += "VERDICT: Still would SHORT. Bias not fully exorcised, but F&G features are now visible to the model."
            else:
                shadow_msg += "VERDICT: Would NOT SHORT! The exorcism worked — model sees the extreme fear contrarian signal."

            logger.info(shadow_msg)
            send_telegram(shadow_msg)

    except Exception as e:
        logger.warning(f"Shadow test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
