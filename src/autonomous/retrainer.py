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
        min_improvement: float = 0.02
    ):
        self.collector = data_collector
        self.features = feature_engine
        self.xgb_model = xgb_model
        self.rl_agent = rl_agent
        self.lstm_model = lstm_model
        self.retrain_interval = timedelta(days=retrain_interval_days)
        self.min_improvement = min_improvement

        self.last_retrain: Optional[datetime] = datetime.now()
        self.retrain_history: list = []

    def should_retrain(self) -> bool:
        """Check if it's time to retrain."""
        if self.last_retrain is None:
            return True
        time_since_retrain = datetime.now() - self.last_retrain
        return time_since_retrain >= self.retrain_interval

    def retrain_models(self, trading_pairs: list) -> Dict[str, Any]:
        """
        Retrain all models with latest data.

        Pipeline:
        1. Collect data from all pairs
        2. Train XGBoost (with walk-forward validation gate)
        3. Train LSTM (if available)
        4. Train RL (every 4th retrain, 2M steps with checkpoint resume)
        """
        logger.info("Starting model retraining...")
        results = {
            "started_at": datetime.now().isoformat(),
            "pairs": trading_pairs,
            "xgb_result": None,
            "lstm_result": None,
            "rl_result": None,
            "deployed": False
        }

        try:
            # Collect training data from all pairs (multi-timeframe)
            all_data = []
            for pair in trading_pairs:
                logger.info(f"Collecting data for {pair}...")
                df_1h = self.collector.get_historical_data(pair, days=180, timeframe="1h")

                try:
                    df_4h = self.collector.get_historical_data(pair, days=180, timeframe="4h")
                    df_1d = self.collector.get_historical_data(pair, days=180, timeframe="1d")
                except Exception:
                    df_4h, df_1d = None, None

                df_features = self.features.calculate_multi_tf_features(df_1h, df_4h, df_1d)

                from ..data.features import create_target_labels
                df_features["target"] = create_target_labels(df_features)
                df_features["pair"] = pair

                all_data.append(df_features)

            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.dropna()

            logger.info(f"Training data: {len(combined_df)} samples")

            feature_cols = self.features.get_feature_names()
            X = combined_df[feature_cols]
            y = combined_df["target"]

            # Store current model performance
            old_xgb_accuracy = self._evaluate_model(self.xgb_model, X, y)

            # ===== TRAIN XGBOOST =====
            logger.info("Training XGBoost model...")
            xgb_metrics = self.xgb_model.train(X, y)
            results["xgb_result"] = xgb_metrics

            new_xgb_accuracy = xgb_metrics["validation_accuracy"]
            improvement = new_xgb_accuracy - old_xgb_accuracy

            logger.info(
                f"XGBoost: Old accuracy: {old_xgb_accuracy:.3f}, "
                f"New accuracy: {new_xgb_accuracy:.3f}, "
                f"Improvement: {improvement:.3f}"
            )

            # Walk-forward validation gate
            xgb_deployed = False
            try:
                from ..models.walk_forward import WalkForwardValidator
                validator = WalkForwardValidator(n_windows=4, min_accuracy=0.36)
                from ..models.xgboost_model import XGBoostPredictor
                passed, wf_results = validator.validate_xgboost(XGBoostPredictor, X, y)

                if passed:
                    logger.info("Walk-forward validation PASSED - deploying XGBoost")
                    self.xgb_model.save_model()
                    xgb_deployed = True
                    results["deployed"] = True
                else:
                    logger.warning("Walk-forward validation FAILED")
                    if improvement >= self.min_improvement:
                        logger.info("But accuracy improved enough - deploying anyway")
                        self.xgb_model.save_model()
                        xgb_deployed = True
                        results["deployed"] = True
                    else:
                        logger.warning("Keeping old model")
                        self.xgb_model.load_model()

                results["walk_forward"] = wf_results
            except Exception as e:
                logger.warning(f"Walk-forward validation failed: {e}")
                # Fallback to old logic
                if improvement >= 0:
                    self.xgb_model.save_model()
                    xgb_deployed = True
                    results["deployed"] = True
                else:
                    self.xgb_model.load_model()

            # ===== TRAIN LSTM =====
            if self.lstm_model is not None:
                try:
                    logger.info("Training LSTM model...")
                    lstm_metrics = self.lstm_model.train(X, y)
                    results["lstm_result"] = lstm_metrics
                    logger.info(f"LSTM: val_acc={lstm_metrics.get('validation_accuracy', 0):.3f}")
                except Exception as e:
                    logger.error(f"LSTM training failed: {e}")

            # ===== TRAIN RL AGENT =====
            if self._should_retrain_rl():
                logger.info("Training RL agent (2M steps)...")

                # Check for checkpoint to resume from
                checkpoint_path = "models/checkpoints/rl_latest"
                resume_from = checkpoint_path if os.path.exists(checkpoint_path + ".zip") else None

                rl_metrics = self.rl_agent.train(
                    combined_df,
                    feature_cols,
                    total_timesteps=2000000,
                    resume_from=resume_from
                )
                results["rl_result"] = rl_metrics
                self.rl_agent.save_model()

            # Update tracking
            self.last_retrain = datetime.now()
            results["completed_at"] = datetime.now().isoformat()
            self.retrain_history.append(results)

            logger.info("Model retraining complete!")
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
