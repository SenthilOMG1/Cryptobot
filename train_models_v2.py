"""
Enhanced Model Training Script V2
=================================
Trains all 3 models: XGBoost (with hard mining + walk-forward) -> LSTM -> RL (2M steps)
Designed for tmux: `tmux new -s training && python train_models_v2.py`

If interrupted: re-run script, RL resumes from latest checkpoint.
Training state tracked in models/training_state.json.
"""

import os
import sys
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

STATE_FILE = "models/training_state.json"


def load_state() -> dict:
    """Load training state to support resume."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"xgboost": False, "lstm": False, "rl": False}


def save_state(state: dict):
    """Save training state."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    logger.info("=" * 60)
    logger.info("CRYPTOBOT V2 - ENHANCED MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Started: {datetime.now().isoformat()}")

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    # Load training state for resume capability
    state = load_state()
    logger.info(f"Training state: {state}")

    # Import components
    from src.config import get_config
    from src.security.vault import SecureVault
    from src.trading.okx_client import SecureOKXClient
    from src.data.collector import DataCollector
    from src.data.features import FeatureEngine, create_target_labels
    from src.models.xgboost_model import XGBoostPredictor
    from src.models.rl_agent import RLTradingAgent
    from src.models.lstm_model import LSTMPredictor
    from src.models.walk_forward import WalkForwardValidator
    import pandas as pd

    # Load config
    config = get_config()
    trading_pairs = config.trading.trading_pairs

    logger.info(f"Training for {len(trading_pairs)} pairs: {trading_pairs}")

    # Connect to OKX
    logger.info("Connecting to OKX...")
    vault = SecureVault()
    okx = SecureOKXClient(vault, demo_mode=config.exchange.demo_mode)

    if not okx.test_connection():
        logger.error("Failed to connect to OKX!")
        sys.exit(1)

    # Initialize data collection
    collector = DataCollector(okx)
    features = FeatureEngine()

    # ===== COLLECT DATA =====
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 0: DATA COLLECTION")
    logger.info("=" * 40)

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
            df_features["target"] = create_target_labels(df_features)
            df_features["pair"] = pair
            all_data.append(df_features)
        except Exception as e:
            logger.error(f"    Failed: {e}")

    if not all_data:
        logger.error("No data collected!")
        sys.exit(1)

    combined_df = pd.concat(all_data, ignore_index=True).dropna()
    logger.info(f"Total training samples: {len(combined_df)}")

    feature_cols = features.get_feature_names()
    X = combined_df[feature_cols]
    y = combined_df["target"]

    # Show class distribution
    logger.info("Target distribution:")
    logger.info(f"  BUY (1):  {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    logger.info(f"  HOLD (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    logger.info(f"  SELL (-1): {(y == -1).sum()} ({(y == -1).mean()*100:.1f}%)")

    # ===== PHASE 1: XGBOOST =====
    if not state.get("xgboost"):
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 1: XGBoost (with hard mining + walk-forward)")
        logger.info("=" * 40)

        xgb_model = XGBoostPredictor(model_path="models/xgboost_model.json")
        xgb_metrics = xgb_model.train(X, y)

        logger.info(f"XGBoost Results:")
        logger.info(f"  Train Accuracy: {xgb_metrics['train_accuracy']:.3f}")
        logger.info(f"  Val Accuracy:   {xgb_metrics['validation_accuracy']:.3f}")

        # Top features
        logger.info("\nTop 10 Important Features:")
        importance = xgb_model.get_feature_importance(top_n=10)
        for _, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Walk-forward validation
        logger.info("\nRunning walk-forward validation...")
        validator = WalkForwardValidator(n_windows=4, min_accuracy=0.36)
        passed, wf_results = validator.validate_xgboost(XGBoostPredictor, X, y)
        logger.info(f"Walk-forward: {'PASSED' if passed else 'FAILED'} (mean_acc={wf_results['mean_accuracy']:.3f})")

        state["xgboost"] = True
        save_state(state)
        logger.info("XGBoost training complete!")
    else:
        logger.info("XGBoost already trained, skipping...")

    # ===== PHASE 2: LSTM =====
    if not state.get("lstm"):
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 2: LSTM (50 epochs)")
        logger.info("=" * 40)

        lstm_model = LSTMPredictor(model_path="models/lstm_model.pt", epochs=50)
        lstm_metrics = lstm_model.train(X, y)

        logger.info(f"LSTM Results:")
        logger.info(f"  Train Accuracy: {lstm_metrics.get('train_accuracy', 0):.3f}")
        logger.info(f"  Val Accuracy:   {lstm_metrics.get('validation_accuracy', 0):.3f}")
        logger.info(f"  Epochs Trained: {lstm_metrics.get('epochs_trained', 0)}")

        state["lstm"] = True
        save_state(state)
        logger.info("LSTM training complete!")
    else:
        logger.info("LSTM already trained, skipping...")

    # ===== PHASE 3: RL AGENT =====
    if not state.get("rl"):
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 3: RL Agent (2M steps with checkpoints)")
        logger.info("=" * 40)

        rl_agent = RLTradingAgent(model_path="models/rl_agent.zip")

        # Check for checkpoint to resume from
        checkpoint_path = "models/checkpoints/rl_latest"
        resume_from = None
        if os.path.exists(checkpoint_path + ".zip"):
            logger.info(f"Found checkpoint: {checkpoint_path}")
            resume_from = checkpoint_path

        rl_metrics = rl_agent.train(
            combined_df,
            feature_cols,
            total_timesteps=2000000,
            initial_balance=1000,
            resume_from=resume_from
        )

        logger.info(f"RL Agent Results:")
        logger.info(f"  Total Return: {rl_metrics['total_return']:.2%}")
        logger.info(f"  Final Portfolio: ${rl_metrics['final_portfolio_value']:.2f}")
        logger.info(f"  Total Trades: {rl_metrics['total_trades']}")

        state["rl"] = True
        save_state(state)
        logger.info("RL training complete!")
    else:
        logger.info("RL already trained, skipping...")

    # ===== DONE =====
    logger.info("\n" + "=" * 60)
    logger.info("ALL TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info("Models saved to:")
    logger.info("  - models/xgboost_model.json (with hard_examples.pkl)")
    logger.info("  - models/lstm_model.pt (with lstm_model_meta.pkl)")
    logger.info("  - models/rl_agent.zip (with checkpoints/)")
    logger.info(f"\nFinished: {datetime.now().isoformat()}")
    logger.info("\nRestart the bot to use new models:")
    logger.info("  systemctl restart cryptobot")

    # Reset state for next training run
    save_state({"xgboost": False, "lstm": False, "rl": False})


if __name__ == "__main__":
    main()
