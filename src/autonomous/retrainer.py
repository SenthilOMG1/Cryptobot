"""
Auto Retrainer
==============
Automatically retrains ML models as market conditions change.
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
    - Uses latest market data
    - Validates new models before deployment
    - Only deploys if new model is better
    """

    def __init__(
        self,
        data_collector,
        feature_engine,
        xgb_model,
        rl_agent,
        retrain_interval_days: int = 7,
        min_improvement: float = 0.02  # 2% minimum improvement required
    ):
        """
        Initialize auto retrainer.

        Args:
            data_collector: DataCollector instance
            feature_engine: FeatureEngine instance
            xgb_model: XGBoostPredictor instance
            rl_agent: RLTradingAgent instance
            retrain_interval_days: Days between retraining
            min_improvement: Minimum accuracy improvement to deploy new model
        """
        self.collector = data_collector
        self.features = feature_engine
        self.xgb_model = xgb_model
        self.rl_agent = rl_agent
        self.retrain_interval = timedelta(days=retrain_interval_days)
        self.min_improvement = min_improvement

        self.last_retrain: Optional[datetime] = datetime.now()  # Don't retrain on first boot
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

        Args:
            trading_pairs: List of trading pairs to train on

        Returns:
            Dict with retraining results
        """
        logger.info("Starting model retraining...")
        results = {
            "started_at": datetime.now().isoformat(),
            "pairs": trading_pairs,
            "xgb_result": None,
            "rl_result": None,
            "deployed": False
        }

        try:
            # Collect training data from all pairs
            all_data = []
            for pair in trading_pairs:
                logger.info(f"Collecting data for {pair}...")
                df = self.collector.get_historical_data(pair, days=90, timeframe="1h")

                # Calculate features
                df_features = self.features.calculate_features(df)

                # Add target labels
                from ..data.features import create_target_labels
                df_features["target"] = create_target_labels(df_features)
                df_features["pair"] = pair

                all_data.append(df_features)

            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.dropna()

            logger.info(f"Training data: {len(combined_df)} samples")

            # Prepare features and labels
            feature_cols = self.features.get_feature_names()
            X = combined_df[feature_cols]
            y = combined_df["target"]

            # Store current model performance
            old_xgb_accuracy = self._evaluate_model(self.xgb_model, X, y)

            # Train new XGBoost model
            logger.info("Training XGBoost model...")
            xgb_metrics = self.xgb_model.train(X, y)
            results["xgb_result"] = xgb_metrics

            # Evaluate new model
            new_xgb_accuracy = xgb_metrics["validation_accuracy"]
            improvement = new_xgb_accuracy - old_xgb_accuracy

            logger.info(
                f"XGBoost: Old accuracy: {old_xgb_accuracy:.3f}, "
                f"New accuracy: {new_xgb_accuracy:.3f}, "
                f"Improvement: {improvement:.3f}"
            )

            # Only deploy if improved enough
            if improvement >= self.min_improvement:
                logger.info("XGBoost model improved - deploying!")
                self.xgb_model.save_model()
                results["deployed"] = True
            elif improvement >= 0:
                logger.info("XGBoost model slightly improved - deploying anyway")
                self.xgb_model.save_model()
                results["deployed"] = True
            else:
                logger.warning("New XGBoost model is worse - keeping old model")
                self.xgb_model.load_model()  # Reload old model

            # Train RL agent (less frequently, more expensive)
            if self._should_retrain_rl():
                logger.info("Training RL agent...")
                rl_metrics = self.rl_agent.train(
                    combined_df,
                    feature_cols,
                    total_timesteps=200000
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
        """
        Check if RL agent should be retrained.
        RL training is expensive, so we do it less frequently.
        """
        # Retrain RL every 4th XGBoost retrain
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
