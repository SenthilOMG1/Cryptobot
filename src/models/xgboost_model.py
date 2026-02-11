"""
XGBoost Price Direction Predictor
=================================
Gradient boosting model that predicts if price will go UP, DOWN, or stay NEUTRAL.
This is one of the AI brains that votes on trading decisions.
"""

import os
import logging
import pickle
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """
    XGBoost-based price direction predictor.

    Predicts:
    - 1 (BUY): Price will go up significantly
    - -1 (SELL): Price will go down significantly
    - 0 (HOLD): Price will stay relatively flat

    Features:
    - Uses gradient boosting for robust predictions
    - Provides confidence scores for each prediction
    - Supports incremental training
    - Auto-saves/loads models
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize XGBoost predictor.

        Args:
            model_path: Path to load existing model (optional)
        """
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.model_path = model_path or "models/xgboost_model.json"
        self.is_trained = False

        # Default hyperparameters (optimized for trading - upgraded for VPS hardware)
        self.params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.03,
            "objective": "multi:softprob",
            "num_class": 3,  # UP, DOWN, HOLD
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,  # Use all CPU cores
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "colsample_bylevel": 0.7,
            "reg_alpha": 0.3,
            "reg_lambda": 2.0,
            "max_delta_step": 1,
            "early_stopping_rounds": 30,
        }

        # Try to load existing model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        early_stopping: int = 10
    ) -> dict:
        """
        Train the XGBoost model on historical data.

        Args:
            X: Feature DataFrame
            y: Target labels (1=BUY, 0=HOLD, -1=SELL)
            test_size: Fraction of data for validation
            early_stopping: Stop if no improvement for N rounds

        Returns:
            dict with training metrics
        """
        logger.info(f"Training XGBoost on {len(X)} samples")

        # Store feature names
        self.feature_names = list(X.columns)

        # Convert labels: -1->0, 0->1, 1->2 (XGBoost needs 0-indexed classes)
        y_encoded = y.map({-1: 0, 0: 1, 1: 2})

        # Split data (time-series aware - no shuffle!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y_encoded.iloc[:split_idx], y_encoded.iloc[split_idx:]

        logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

        # Calculate class weights to handle imbalanced data
        from collections import Counter
        counts = Counter(y_train.values)
        total = len(y_train)
        n_classes = len(counts)
        class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}
        sample_weights = y_train.map(class_weights).values

        # Create model (extract early_stopping_rounds from params)
        model_params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
        early_stopping = self.params.get("early_stopping_rounds", 30)
        self.model = xgb.XGBClassifier(**model_params)

        # Train with class balancing and early stopping
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        self.is_trained = True

        metrics = {
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "features_used": len(self.feature_names),
        }

        logger.info(f"Training complete - Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

        # Save model
        self.save_model()

        return metrics

    def predict(self, features: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict price direction with confidence.

        Args:
            features: DataFrame with features for single sample or multiple

        Returns:
            Tuple of (prediction, confidence)
            prediction: 1=BUY, 0=HOLD, -1=SELL
            confidence: 0.0 to 1.0
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained! Call train() first or load a saved model.")

        # Ensure correct feature order
        if isinstance(features, pd.DataFrame):
            # Filter to only use features the model knows
            available_features = [f for f in self.feature_names if f in features.columns]
            missing_features = [f for f in self.feature_names if f not in features.columns]

            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features, using defaults")
                for f in missing_features:
                    features[f] = 0

            X = features[self.feature_names]
        else:
            X = features

        # Get prediction probabilities
        proba = self.model.predict_proba(X)

        # Get most likely class
        pred_encoded = np.argmax(proba, axis=1)

        # Get confidence (probability of predicted class)
        confidence = np.max(proba, axis=1)

        # Convert back to original labels: 0->-1, 1->0, 2->1
        label_map = {0: -1, 1: 0, 2: 1}
        predictions = np.array([label_map[p] for p in pred_encoded])

        # Return single values if single sample
        if len(predictions) == 1:
            return int(predictions[0]), float(confidence[0])

        return predictions, confidence

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities for all classes.

        Returns:
            Array of shape (n_samples, 3) with probabilities for [SELL, HOLD, BUY]
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained!")

        if isinstance(features, pd.DataFrame):
            X = features[self.feature_names]
        else:
            X = features

        return self.model.predict_proba(X)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained!")

        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        return importance_df.head(top_n)

    def save_model(self, path: Optional[str] = None):
        """Save model to file."""
        path = path or self.model_path

        if self.model is None:
            raise ValueError("No model to save!")

        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save XGBoost model
        self.model.save_model(path)

        # Save feature names separately
        meta_path = path.replace(".json", "_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "feature_names": self.feature_names,
                "params": self.params
            }, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: Optional[str] = None):
        """Load model from file."""
        path = path or self.model_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)

        # Load metadata
        meta_path = path.replace(".json", "_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.feature_names = meta.get("feature_names", [])
                self.params = meta.get("params", self.params)

        self.is_trained = True
        logger.info(f"Model loaded from {path}")

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate model performance.

        Returns:
            dict with accuracy and per-class metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained!")

        y_encoded = y.map({-1: 0, 0: 1, 1: 2})
        predictions = self.model.predict(X[self.feature_names])

        accuracy = accuracy_score(y_encoded, predictions)
        report = classification_report(
            y_encoded, predictions,
            target_names=["SELL", "HOLD", "BUY"],
            output_dict=True
        )

        return {
            "accuracy": accuracy,
            "report": report
        }
