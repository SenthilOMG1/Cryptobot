"""
Walk-Forward Validation
=======================
Rolling-window validation to prevent overfitting.
Model must pass all windows before deployment.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation with sliding windows.

    Splits data into N windows, each with a train and validation portion.
    Model must achieve minimum accuracy on ALL windows to pass.
    """

    def __init__(
        self,
        n_windows: int = 4,
        train_pct: float = 0.40,
        val_pct: float = 0.15,
        min_accuracy: float = 0.36  # Random baseline for 3 classes = 33%
    ):
        """
        Args:
            n_windows: Number of sliding windows
            train_pct: Fraction of total data used for training in each window
            val_pct: Fraction of total data used for validation in each window
            min_accuracy: Minimum accuracy required on each window (>33% = better than random)
        """
        self.n_windows = n_windows
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.min_accuracy = min_accuracy

    def validate_xgboost(
        self,
        model_class,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: dict = None
    ) -> Tuple[bool, Dict]:
        """
        Run walk-forward validation for XGBoost.

        Args:
            model_class: XGBoostPredictor class (not instance)
            X: Full feature DataFrame
            y: Full target Series
            model_params: Optional model parameters override

        Returns:
            Tuple of (passed: bool, results: dict)
        """
        n = len(X)
        window_size = int(n * (self.train_pct + self.val_pct))
        train_size = int(n * self.train_pct)
        step = max(1, (n - window_size) // max(1, self.n_windows - 1))

        results = {
            "n_windows": self.n_windows,
            "total_samples": n,
            "window_results": [],
            "all_passed": True,
            "mean_accuracy": 0.0
        }

        accuracies = []

        for i in range(self.n_windows):
            start = i * step
            end = min(start + window_size, n)
            train_end = min(start + train_size, n)

            if train_end >= end or end - train_end < 50:
                logger.warning(f"Window {i+1}: Not enough data, skipping")
                continue

            X_train = X.iloc[start:train_end]
            y_train = y.iloc[start:train_end]
            X_val = X.iloc[train_end:end]
            y_val = y.iloc[train_end:end]

            try:
                # Create and train a temporary model
                model = model_class()

                # Encode labels for XGBoost
                y_train_enc = y_train.map({-1: 0, 0: 1, 1: 2})
                y_val_enc = y_val.map({-1: 0, 0: 1, 1: 2})

                # Calculate class weights
                from collections import Counter
                counts = Counter(y_train_enc.values)
                total = len(y_train_enc)
                n_classes = len(counts) if counts else 1
                class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}
                sample_weights = y_train_enc.map(class_weights).values

                # Train (using default params from the class)
                import xgboost as xgb
                model_params_clean = {k: v for k, v in model.params.items() if k != "early_stopping_rounds"}
                model.model = xgb.XGBClassifier(**model_params_clean)
                model.model.fit(
                    X_train, y_train_enc,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val_enc)],
                    verbose=False
                )
                model.feature_names = list(X_train.columns)
                model.is_trained = True

                # Evaluate
                val_pred = model.model.predict(X_val)
                accuracy = accuracy_score(y_val_enc, val_pred)
                accuracies.append(accuracy)

                passed = accuracy >= self.min_accuracy
                if not passed:
                    results["all_passed"] = False

                window_result = {
                    "window": i + 1,
                    "train_range": f"{start}-{train_end}",
                    "val_range": f"{train_end}-{end}",
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "accuracy": accuracy,
                    "passed": passed
                }
                results["window_results"].append(window_result)

                logger.info(
                    f"Walk-forward window {i+1}/{self.n_windows}: "
                    f"accuracy={accuracy:.3f} {'PASS' if passed else 'FAIL'}"
                )

            except Exception as e:
                logger.error(f"Walk-forward window {i+1} failed: {e}")
                results["all_passed"] = False
                results["window_results"].append({
                    "window": i + 1,
                    "error": str(e),
                    "passed": False
                })

        if accuracies:
            results["mean_accuracy"] = float(np.mean(accuracies))
            results["min_accuracy_seen"] = float(np.min(accuracies))
            results["max_accuracy_seen"] = float(np.max(accuracies))

        status = "PASSED" if results["all_passed"] else "FAILED"
        logger.info(
            f"Walk-forward validation {status}: "
            f"mean_acc={results['mean_accuracy']:.3f}, "
            f"windows={len(accuracies)}/{self.n_windows}"
        )

        return results["all_passed"], results
