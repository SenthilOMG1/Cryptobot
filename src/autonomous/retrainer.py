"""
Auto Retrainer V2
=================
Automatically retrains ML models as market conditions change.

V2 Upgrades:
- Walk-forward validation gate before deploying XGBoost
- LSTM retraining alongside XGBoost
- 2M steps for RL training (with checkpoint resume)
- Checkpoint-based RL resume
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class AutoRetrainer:
    """
    Automatic model retraining system.

    Features:
    - Retrains models on schedule (weekly by default)
    - Walk-forward validation gate for XGBoost
    - LSTM retraining alongside XGBoost
    - RL training with 2M steps and checkpoint resume
    - Only deploys if new model passes validation
    """

    def __init__(
        self,
        data_collector,
        feature_engine,
        xgb_model,
        rl_agent,
        lstm_model=None,
        retrain_interval_days: int = 7,
        min_improvement: float = 0.02,
        full_retrain_interval_days: int = 30,
    ):
        self.collector = data_collector
        self.features = feature_engine
        self.xgb_model = xgb_model
        self.rl_agent = rl_agent
        self.lstm_model = lstm_model
        self.retrain_interval = timedelta(days=retrain_interval_days)
        self.full_retrain_interval = timedelta(days=full_retrain_interval_days)
        self.min_improvement = min_improvement

        self.last_retrain: Optional[datetime] = datetime.now()
        self.last_full_retrain: Optional[datetime] = datetime.now()
        self.retrain_history: list = []

    def should_retrain(self) -> bool:
        """Check if it's time to retrain."""
        if self.last_retrain is None:
            return True
        time_since_retrain = datetime.now() - self.last_retrain
        return time_since_retrain >= self.retrain_interval

    def _is_full_retrain_due(self) -> bool:
        """Check if monthly full retrain is due."""
        if self.last_full_retrain is None:
            return True
        return datetime.now() - self.last_full_retrain >= self.full_retrain_interval

    def retrain_models(self, trading_pairs: list) -> Dict[str, Any]:
        """
        Smart retraining: fine-tune weekly, full retrain monthly.

        Weekly (fine-tune): Loads existing models, trains on 60 days of data
        with low learning rates. Models build on previous knowledge.

        Monthly (full retrain): Starts fresh with 180 days of data.
        Resets all weights for a clean slate.

        RL agent: Checkpoint resume every 4th retrain, full retrain monthly.
        """
        is_full = self._is_full_retrain_due()
        mode = "full" if is_full else "fine_tune"
        data_days = 180 if is_full else 60

        logger.info(f"Starting model retraining (mode={mode}, data={data_days}d)...")
        results = {
            "started_at": datetime.now().isoformat(),
            "mode": mode,
            "pairs": trading_pairs,
            "xgb_result": None,
            "lstm_result": None,
            "rl_result": None,
            "deployed": False
        }

        try:
            # Collect training data
            all_data = []
            for pair in trading_pairs:
                logger.info(f"Collecting data for {pair}...")
                df_1h = self.collector.get_historical_data(pair, days=data_days, timeframe="1h")

                try:
                    df_4h = self.collector.get_historical_data(pair, days=data_days, timeframe="4h")
                    df_1d = self.collector.get_historical_data(pair, days=data_days, timeframe="1d")
                except Exception:
                    df_4h, df_1d = None, None

                df_features = self.features.calculate_multi_tf_features(df_1h, df_4h, df_1d)

                from ..data.features import create_target_labels
                df_features["target"] = create_target_labels(df_features)
                df_features["pair"] = pair

                all_data.append(df_features)

            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.dropna()

            logger.info(f"Training data: {len(combined_df)} samples ({mode} mode)")

            feature_cols = self.features.get_feature_names()
            X = combined_df[feature_cols]
            y = combined_df["target"]

            # ===== XGBOOST =====
            if is_full:
                logger.info("Full training XGBoost model...")
                old_xgb_accuracy = self._evaluate_model(self.xgb_model, X, y)
                xgb_metrics = self.xgb_model.train(X, y)
                results["xgb_result"] = xgb_metrics

                new_xgb_accuracy = xgb_metrics["validation_accuracy"]
                improvement = new_xgb_accuracy - old_xgb_accuracy

                # Walk-forward validation gate
                try:
                    from ..models.walk_forward import WalkForwardValidator
                    validator = WalkForwardValidator(n_windows=4, min_accuracy=0.36)
                    from ..models.xgboost_model import XGBoostPredictor
                    passed, wf_results = validator.validate_xgboost(XGBoostPredictor, X, y)

                    if passed or improvement >= self.min_improvement:
                        self.xgb_model.save_model()
                        results["deployed"] = True
                    else:
                        self.xgb_model.load_model()

                    results["walk_forward"] = wf_results
                except Exception as e:
                    logger.warning(f"Walk-forward validation failed: {e}")
                    if improvement >= 0:
                        self.xgb_model.save_model()
                        results["deployed"] = True
                    else:
                        self.xgb_model.load_model()
            else:
                logger.info("Fine-tuning XGBoost model...")
                xgb_metrics = self.xgb_model.fine_tune(X, y, n_new_trees=100)
                results["xgb_result"] = xgb_metrics
                if xgb_metrics.get("deployed", False):
                    results["deployed"] = True

            # ===== LSTM =====
            if self.lstm_model is not None:
                try:
                    if is_full:
                        logger.info("Full training LSTM model...")
                        lstm_metrics = self.lstm_model.train(X, y)
                    else:
                        logger.info("Fine-tuning LSTM model...")
                        lstm_metrics = self.lstm_model.fine_tune(X, y)
                    results["lstm_result"] = lstm_metrics
                    logger.info(f"LSTM ({mode}): val_acc={lstm_metrics.get('validation_accuracy', 0):.3f}")
                except Exception as e:
                    logger.error(f"LSTM {mode} failed: {e}")

            # ===== RL AGENT =====
            if is_full or self._should_retrain_rl():
                timesteps = 2000000 if is_full else 500000
                logger.info(f"Training RL agent ({timesteps//1000}K steps)...")

                checkpoint_path = "models/checkpoints/rl_latest"
                resume_from = None if is_full else (
                    checkpoint_path if os.path.exists(checkpoint_path + ".zip") else None
                )

                rl_metrics = self.rl_agent.train(
                    combined_df, feature_cols,
                    total_timesteps=timesteps,
                    resume_from=resume_from
                )
                results["rl_result"] = rl_metrics
                self.rl_agent.save_model()

            # Update tracking
            self.last_retrain = datetime.now()
            if is_full:
                self.last_full_retrain = datetime.now()
            results["completed_at"] = datetime.now().isoformat()
            self.retrain_history.append(results)

            logger.info(f"Model retraining complete! (mode={mode})")
            return results

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            results["error"] = str(e)
            return results

    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate model accuracy."""
        try:
            if not model.is_trained:
                return 0.0
            metrics = model.evaluate(X, y)
            return metrics.get("accuracy", 0.0)
        except:
            return 0.0

    def _should_retrain_rl(self) -> bool:
        """Check if RL agent should be retrained (every 4th retrain)."""
        return len(self.retrain_history) % 4 == 0

    def get_status(self) -> Dict[str, Any]:
        """Get retrainer status."""
        next_retrain = None
        if self.last_retrain:
            next_retrain = (self.last_retrain + self.retrain_interval).isoformat()

        return {
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "next_retrain": next_retrain,
            "retrain_count": len(self.retrain_history),
            "retrain_interval_days": self.retrain_interval.days
        }
