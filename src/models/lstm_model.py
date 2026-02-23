"""
LSTM Sequence Model
===================
Third model in the ensemble - captures sequential patterns
that XGBoost (pointwise) and RL (policy-based) might miss.
"""

import os
import logging
import pickle
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TradingLSTM(nn.Module):
    """LSTM network for price direction prediction."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden_size // 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # Take last timestep
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class TradingSequenceDataset(Dataset):
    """Creates sequences of candles for LSTM training."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int = 48):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.features) - self.seq_length)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length - 1]
        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()


class LSTMPredictor:
    """
    LSTM-based price direction predictor.

    Architecture: LSTM(128) -> LSTM(64) -> Dense(32) -> Dense(3)
    Input: Sequences of 48 candles with normalized features
    Output: BUY (1), HOLD (0), SELL (-1) with confidence
    """

    def __init__(
        self,
        model_path: str = "models/lstm_model.pt",
        seq_length: int = 48,
        hidden_size: int = 128,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ):
        self.model_path = model_path
        self.meta_path = model_path.replace(".pt", "_meta.pkl")
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model: Optional[TradingLSTM] = None
        self.feature_names: List[str] = []
        self.is_trained = False

        # Normalization parameters (saved with model)
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        self.device = torch.device("cpu")

        # Try to load existing model
        if os.path.exists(model_path):
            self.load_model()

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train the LSTM model.

        Args:
            X: Feature DataFrame (chronologically ordered)
            y: Target labels (1=BUY, 0=HOLD, -1=SELL)

        Returns:
            dict with training metrics
        """
        logger.info(f"Training LSTM on {len(X)} samples (seq_length={self.seq_length})")

        self.feature_names = list(X.columns)
        features = X.values.astype(np.float32)

        # Z-score normalization
        self.feature_mean = np.nanmean(features, axis=0)
        self.feature_std = np.nanstd(features, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0  # Avoid division by zero
        features_norm = (features - self.feature_mean) / self.feature_std
        features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=3.0, neginf=-3.0)

        # Encode labels: -1->0, 0->1, 1->2
        targets = y.map({-1: 0, 0: 1, 1: 2}).values.astype(np.int64)

        # Chronological split (80/20)
        split_idx = int(len(features_norm) * 0.8)
        train_features = features_norm[:split_idx]
        train_targets = targets[:split_idx]
        val_features = features_norm[split_idx:]
        val_targets = targets[split_idx:]

        # Create datasets
        train_dataset = TradingSequenceDataset(train_features, train_targets, self.seq_length)
        val_dataset = TradingSequenceDataset(val_features, val_targets, self.seq_length)

        if len(train_dataset) < 100:
            logger.warning(f"Too few training sequences ({len(train_dataset)}), need at least 100")
            return {"error": "Not enough data"}

        # Don't shuffle - LSTM learns temporal patterns, order matters
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Create model
        input_size = features_norm.shape[1]
        self.model = TradingLSTM(input_size, self.hidden_size).to(self.device)

        # Class weights for imbalanced data
        from collections import Counter
        counts = Counter(train_targets)
        total = len(train_targets)
        n_classes = 3
        class_weights = torch.FloatTensor([
            total / (n_classes * counts.get(i, 1)) for i in range(n_classes)
        ]).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            train_acc = train_correct / max(train_total, 1)
            val_acc = val_correct / max(val_total, 1)
            avg_val_loss = val_loss / max(len(val_loader), 1)

            scheduler.step(avg_val_loss)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(
                    f"LSTM Epoch {epoch+1}/{self.epochs}: "
                    f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
                    f"val_loss={avg_val_loss:.4f}"
                )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                self._save_checkpoint()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    # Reload best model
                    self._load_checkpoint()
                    break

        self.is_trained = True
        self.save_model()

        metrics = {
            "train_accuracy": train_acc,
            "validation_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "train_sequences": len(train_dataset),
            "val_sequences": len(val_dataset),
            "features_used": input_size
        }

        logger.info(f"LSTM training complete: val_acc={best_val_acc:.3f}")
        return metrics

    def fine_tune(self, X: pd.DataFrame, y: pd.Series, freeze_early: bool = True) -> dict:
        """
        Incrementally fine-tune the existing LSTM model on new data.

        Key differences from train():
        - Loads existing model weights instead of creating fresh
        - Optionally freezes early LSTM layer to prevent catastrophic forgetting
        - Uses lower learning rate (10x lower than full train)
        - Blends normalization stats with existing ones (EMA)

        Args:
            X: Feature DataFrame (recent data, chronologically ordered)
            y: Target labels (1=BUY, 0=HOLD, -1=SELL)
            freeze_early: If True, freeze lstm1 weights (only train lstm2 + FC layers)

        Returns:
            dict with training metrics
        """
        if not self.is_trained or self.model is None:
            logger.warning("No existing model to fine-tune, falling back to full train")
            return self.train(X, y)

        logger.info(f"Fine-tuning LSTM on {len(X)} samples (freeze_early={freeze_early})")

        # Verify feature compatibility
        new_features = list(X.columns)
        if new_features != self.feature_names:
            missing = set(self.feature_names) - set(new_features)
            extra = set(new_features) - set(self.feature_names)
            if missing:
                logger.warning(f"Fine-tune: {len(missing)} features missing, padding with zeros")
                for col in missing:
                    X = X.copy()
                    X[col] = 0.0
            if extra:
                logger.info(f"Fine-tune: dropping {len(extra)} extra features")
            X = X[self.feature_names]

        features = X.values.astype(np.float32)

        # Blend normalization: EMA with 70% old / 30% new
        # This prevents distribution shift from causing the model to see "alien" inputs
        new_mean = np.nanmean(features, axis=0)
        new_std = np.nanstd(features, axis=0)
        new_std[new_std == 0] = 1.0

        blend_alpha = 0.3  # How much new data influences normalization
        self.feature_mean = (1 - blend_alpha) * self.feature_mean + blend_alpha * new_mean
        self.feature_std = (1 - blend_alpha) * self.feature_std + blend_alpha * new_std

        features_norm = (features - self.feature_mean) / self.feature_std
        features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=3.0, neginf=-3.0)

        # Encode labels
        targets = y.map({-1: 0, 0: 1, 1: 2}).values.astype(np.int64)

        # Chronological split
        split_idx = int(len(features_norm) * 0.8)
        train_features = features_norm[:split_idx]
        train_targets = targets[:split_idx]
        val_features = features_norm[split_idx:]
        val_targets = targets[split_idx:]

        train_dataset = TradingSequenceDataset(train_features, train_targets, self.seq_length)
        val_dataset = TradingSequenceDataset(val_features, val_targets, self.seq_length)

        if len(train_dataset) < 50:
            logger.warning(f"Too few sequences for fine-tune ({len(train_dataset)})")
            return {"error": "Not enough data for fine-tuning"}

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Freeze early layers to prevent catastrophic forgetting
        if freeze_early:
            for param in self.model.lstm1.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Frozen lstm1: training {trainable}/{total_params} params ({trainable/total_params*100:.0f}%)")

        # Class weights
        from collections import Counter
        counts = Counter(train_targets)
        total = len(train_targets)
        class_weights = torch.FloatTensor([
            total / (3 * counts.get(i, 1)) for i in range(3)
        ]).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # Lower LR for fine-tuning: 10x less than full train
        fine_tune_lr = self.learning_rate * 0.1
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=fine_tune_lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        fine_tune_epochs = min(self.epochs, 30)  # Cap at 30 for fine-tune

        for epoch in range(fine_tune_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            train_acc = train_correct / max(train_total, 1)
            val_acc = val_correct / max(val_total, 1)
            avg_val_loss = val_loss / max(len(val_loader), 1)
            scheduler.step(avg_val_loss)

            if epoch % 5 == 0 or epoch == fine_tune_epochs - 1:
                logger.info(
                    f"Fine-tune Epoch {epoch+1}/{fine_tune_epochs}: "
                    f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, val_loss={avg_val_loss:.4f}"
                )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1
                if patience_counter >= 7:
                    logger.info(f"Fine-tune early stopping at epoch {epoch+1}")
                    self._load_checkpoint()
                    break

        # Unfreeze all layers for next time
        if freeze_early:
            for param in self.model.lstm1.parameters():
                param.requires_grad = True

        self.save_model()

        metrics = {
            "mode": "fine_tune",
            "train_accuracy": train_acc,
            "validation_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "train_sequences": len(train_dataset),
            "val_sequences": len(val_dataset),
            "learning_rate": fine_tune_lr,
            "frozen_layers": "lstm1" if freeze_early else "none"
        }

        logger.info(f"LSTM fine-tune complete: val_acc={best_val_acc:.3f}")
        return metrics

    def predict(self, features_df: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict price direction using the last seq_length rows.

        Args:
            features_df: DataFrame with features (needs at least seq_length rows)

        Returns:
            Tuple of (action, confidence)
            action: 1=BUY, 0=HOLD, -1=SELL
            confidence: 0.0 to 1.0
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained!")

        # Get feature columns
        available_cols = [f for f in self.feature_names if f in features_df.columns]
        missing_cols = [f for f in self.feature_names if f not in features_df.columns]
        if missing_cols:
            logger.warning(f"LSTM: Only {len(available_cols)}/{len(self.feature_names)} features available")
            logger.warning(f"Missing {len(missing_cols)} features, using training means as neutral defaults")

        # Prepare full feature matrix with correct columns
        # Use training mean for missing features so they normalize to 0 (neutral)
        # instead of zeros which create phantom bearish/bullish signals
        cols_data = {}
        for i, col in enumerate(self.feature_names):
            if col in features_df.columns:
                cols_data[col] = features_df[col].values
            else:
                # Use training mean so (mean - mean) / std = 0 after normalization
                default_val = self.feature_mean[i] if self.feature_mean is not None else 0.0
                cols_data[col] = np.full(len(features_df), default_val)
        X = pd.DataFrame(cols_data, index=features_df.index)

        features = X.values.astype(np.float32)

        # Use last seq_length rows
        if len(features) < self.seq_length:
            # Pad with zeros at the beginning if not enough data
            padding = np.zeros((self.seq_length - len(features), features.shape[1]), dtype=np.float32)
            features = np.concatenate([padding, features])
        else:
            features = features[-self.seq_length:]

        # Normalize
        features_norm = (features - self.feature_mean) / self.feature_std
        features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=3.0, neginf=-3.0)

        # Predict
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features_norm).unsqueeze(0).to(self.device)
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # Map: class 0=SELL, 1=HOLD, 2=BUY
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # Convert to action
        label_map = {0: -1, 1: 0, 2: 1}
        action = label_map[pred_class]

        return action, confidence

    def save_model(self, path: Optional[str] = None):
        """Save model and metadata."""
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.model is not None:
            torch.save(self.model.state_dict(), path)

        meta_path = path.replace(".pt", "_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "feature_names": self.feature_names,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "seq_length": self.seq_length,
                "hidden_size": self.hidden_size,
                "input_size": len(self.feature_names)
            }, f)

        logger.info(f"LSTM model saved to {path}")

    def load_model(self, path: Optional[str] = None):
        """Load model and metadata."""
        path = path or self.model_path
        meta_path = path.replace(".pt", "_meta.pkl")

        if not os.path.exists(meta_path):
            logger.warning(f"LSTM metadata not found: {meta_path}")
            return

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            self.feature_names = meta.get("feature_names", [])
            self.feature_mean = meta.get("feature_mean")
            self.feature_std = meta.get("feature_std")
            self.seq_length = meta.get("seq_length", self.seq_length)
            self.hidden_size = meta.get("hidden_size", self.hidden_size)
            input_size = meta.get("input_size", len(self.feature_names))

        if os.path.exists(path) and input_size > 0:
            self.model = TradingLSTM(input_size, self.hidden_size).to(self.device)
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.model.eval()
            self.is_trained = True
            logger.info(f"LSTM model loaded from {path}")
        else:
            logger.warning(f"LSTM model file not found: {path}")

    def _save_checkpoint(self):
        """Save best model checkpoint during training."""
        if self.model is not None:
            ckpt_path = self.model_path.replace(".pt", "_best.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(self.model.state_dict(), ckpt_path)

    def _load_checkpoint(self):
        """Load best model checkpoint."""
        ckpt_path = self.model_path.replace(".pt", "_best.pt")
        if os.path.exists(ckpt_path) and self.model is not None:
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
