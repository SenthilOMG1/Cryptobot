"""
XGBoost Price Direction Predictor
=================================
Gradient boosting model that predicts if price will go UP, DOWN, or stay NEUTRAL.
This is one of the AI brains that votes on trading decisions.

V2 Upgrades:
- Hard example mining (3x weight on misclassified samples)
- Recency weighting (recent data matters more)
- Persistent hard examples between retrains
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

HARD_EXAMPLES_PATH = "models/hard_examples.pkl"


class XGBoostPredictor:
    """
    XGBoost-based price direction predictor.

    Predicts:
    - 1 (BUY): Price will go up significantly
    - -1 (SELL): Price will go down significantly
    - 0 (HOLD): Price will stay relatively flat
    """

    def __init__(self, model_path: Optional[str] = None):
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
            "num_class": 3,
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
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

        # Compute combined sample weights (class balance + hard examples + recency)
        sample_weights = self._compute_sample_weights(X_train, y_train)

        # Create model (extract early_stopping_rounds from params)
        model_params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
        self.model = xgb.XGBClassifier(**model_params)

        # Train with combined weights and early stopping
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

        # Identify and save hard examples for next retrain
        self._identify_hard_examples(X_train, y_train)

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

    def fine_tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        n_new_trees: int = 100
    ) -> dict:
        """
        Incrementally fine-tune the existing XGBoost model on new data.

        Adds new trees on top of the existing model using XGBoost's native
        incremental training via xgb_model parameter. The existing trees are
        preserved and new trees learn to correct remaining errors.

        Args:
            X: Feature DataFrame (recent data)
            y: Target labels (1=BUY, 0=HOLD, -1=SELL)
            test_size: Fraction for validation
            n_new_trees: Number of new boosting rounds to add

        Returns:
            dict with training metrics
        """
        if not self.is_trained or self.model is None:
            logger.warning("No existing model to fine-tune, falling back to full train")
            return self.train(X, y)

        logger.info(f"Fine-tuning XGBoost on {len(X)} new samples (+{n_new_trees} trees)")

        # Ensure same feature set
        missing_cols = [c for c in self.feature_names if c not in X.columns]
        extra_cols = [c for c in X.columns if c not in self.feature_names]
        if missing_cols:
            logger.warning(f"Fine-tune: {len(missing_cols)} features missing, padding with 0")
            for col in missing_cols:
                X = X.copy()
                X[col] = 0.0
        X = X[self.feature_names]

        y_encoded = y.map({-1: 0, 0: 1, 1: 2})

        # Chronological split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y_encoded.iloc[:split_idx], y_encoded.iloc[split_idx:]

        logger.info(f"Fine-tune split: {len(X_train)} train, {len(X_val)} val")

        sample_weights = self._compute_sample_weights(X_train, y_train)

        # Use existing booster as base and add new trees
        fine_tune_params = self.params.copy()
        fine_tune_params["n_estimators"] = n_new_trees
        fine_tune_params["learning_rate"] = self.params["learning_rate"] * 0.5  # Lower LR

        model_params = {k: v for k, v in fine_tune_params.items() if k != "early_stopping_rounds"}
        new_model = xgb.XGBClassifier(**model_params)

        new_model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            xgb_model=self.model.get_booster(),
            verbose=False
        )

        # Evaluate old vs new
        old_val_pred = self.model.predict(X_val)
        new_val_pred = new_model.predict(X_val)
        old_acc = accuracy_score(y_val, old_val_pred)
        new_acc = accuracy_score(y_val, new_val_pred)

        logger.info(f"Val accuracy: {old_acc:.3f} (old) → {new_acc:.3f} (fine-tuned)")

        # Only deploy if improved (or within 1% - fine-tune on new data may shift)
        if new_acc >= old_acc - 0.01:
            self.model = new_model
            self.is_trained = True
            self._identify_hard_examples(X_train, y_train)
            self.save_model()
            logger.info(f"Fine-tuned model deployed (acc: {new_acc:.3f})")
            deployed = True
        else:
            logger.warning(f"Fine-tuned model worse ({new_acc:.3f} < {old_acc:.3f} - 0.01), keeping old")
            deployed = False

        return {
            "mode": "fine_tune",
            "old_val_accuracy": old_acc,
            "new_val_accuracy": new_acc,
            "deployed": deployed,
            "new_trees_added": n_new_trees,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

    def _compute_sample_weights(self, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        """
        Compute combined sample weights from 3 sources:
        1. Class balancing - inversely proportional to class frequency
        2. Hard examples - 3x weight on previously misclassified samples
        3. Recency - exponential decay, oldest=0.3x, newest=1.0x
        """
        n = len(y_train)

        # 1. Class balancing weights
        from collections import Counter
        counts = Counter(y_train.values)
        total = len(y_train)
        n_classes = len(counts)
        class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}
        class_w = y_train.map(class_weights).values.astype(float)

        # 2. Hard example weights (3x for previously misclassified)
        hard_w = np.ones(n, dtype=float)
        hard_indices = self._load_hard_examples()
        if hard_indices is not None and len(hard_indices) > 0:
            # Map hard example indices to current training set
            valid_hard = hard_indices[hard_indices < n]
            hard_w[valid_hard] = 3.0
            logger.info(f"Applied 3x weight to {len(valid_hard)} hard examples")

        # 3. Recency weights (exponential from 0.3 to 1.0)
        recency_w = np.linspace(0.3, 1.0, n)

        # Combine multiplicatively
        combined = class_w * hard_w * recency_w

        # Normalize to mean=1
        combined = combined / combined.mean()

        return combined

    def _identify_hard_examples(self, X_train: pd.DataFrame, y_train: pd.Series):
        """After training, predict on training set and save misclassified indices."""
        try:
            predictions = self.model.predict(X_train)
            misclassified = np.where(predictions != y_train.values)[0]

            os.makedirs(os.path.dirname(HARD_EXAMPLES_PATH), exist_ok=True)
            with open(HARD_EXAMPLES_PATH, "wb") as f:
                pickle.dump(misclassified, f)

            logger.info(f"Saved {len(misclassified)} hard examples ({len(misclassified)/len(y_train)*100:.1f}% of training data)")
        except Exception as e:
            logger.warning(f"Failed to save hard examples: {e}")

    @staticmethod
    def _load_hard_examples() -> Optional[np.ndarray]:
        """Load previously identified hard examples."""
        if os.path.exists(HARD_EXAMPLES_PATH):
            try:
                with open(HARD_EXAMPLES_PATH, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

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
        """Get prediction probabilities for all classes."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained!")

        if isinstance(features, pd.DataFrame):
            X = features[self.feature_names]
        else:
            X = features

        return self.model.predict_proba(X)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance rankings."""
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

        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.model.save_model(path)

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

        self.model = xgb.XGBClassifier()
        self.model.load_model(path)

        meta_path = path.replace(".json", "_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.feature_names = meta.get("feature_names", [])
                self.params = meta.get("params", self.params)

        self.is_trained = True
        logger.info(f"Model loaded from {path}")

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate model performance."""
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
