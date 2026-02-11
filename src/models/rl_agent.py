"""
Reinforcement Learning Trading Agent
====================================
PPO-based agent that learns optimal trading through experience.
This is the second AI brain that votes on trading decisions.
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for crypto trading.

    State: Technical indicators + portfolio state
    Actions: 0=HOLD, 1=BUY, 2=SELL
    Reward: Profit/Loss with penalties for risk
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        initial_balance: float = 1000.0,
        transaction_fee: float = 0.001,  # 0.1% fee
        max_position_pct: float = 0.25,  # Max 25% per trade
    ):
        """
        Initialize trading environment.

        Args:
            df: DataFrame with OHLCV and features
            feature_columns: List of feature column names
            initial_balance: Starting balance in USDT
            transaction_fee: Fee per transaction (0.001 = 0.1%)
            max_position_pct: Maximum position size as fraction of balance
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position_pct = max_position_pct

        # Precompute target labels for reward alignment
        self.has_targets = 'target' in df.columns
        if self.has_targets:
            self.targets = df['target'].values

        # State space: features + [balance_ratio, position_ratio, unrealized_pnl]
        n_features = len(feature_columns) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features,), dtype=np.float32
        )

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Trading state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Crypto holdings
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one trading step.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL

        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self.df.loc[self.current_step, "close"]

        # Execute action
        reward = 0.0
        trade_executed = False

        if action == 1:  # BUY
            if self.balance > 0 and self.position == 0:
                # Calculate position size
                trade_amount = self.balance * self.max_position_pct
                fee = trade_amount * self.transaction_fee
                crypto_bought = (trade_amount - fee) / current_price

                self.position = crypto_bought
                self.balance -= trade_amount
                self.entry_price = current_price
                self.total_trades += 1
                trade_executed = True

        elif action == 2:  # SELL
            if self.position > 0:
                # Sell all position
                sale_value = self.position * current_price
                fee = sale_value * self.transaction_fee
                self.balance += (sale_value - fee)

                # Calculate P&L for reward
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                reward = pnl_pct * 100  # Scale reward

                if pnl_pct > 0:
                    self.winning_trades += 1

                self.position = 0.0
                self.entry_price = 0.0
                self.total_trades += 1
                trade_executed = True

        # Move to next step
        self.current_step += 1

        # Reward alignment with target labels (same signals XGBoost trains on)
        if self.has_targets and self.current_step < len(self.targets):
            target = self.targets[self.current_step - 1]
            # action: 0=HOLD, 1=BUY, 2=SELL; target: 0=HOLD, 1=BUY, -1=SELL
            action_map = {0: 0, 1: 1, 2: -1}  # map RL actions to target labels
            rl_signal = action_map[int(action)]
            if rl_signal == target and target != 0:
                reward += 0.2  # Small bonus for agreeing with direction signal

        # Calculate unrealized P&L for holding positions
        if self.position > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price

            # Penalize large drawdowns
            if unrealized_pnl < -0.05:  # Down more than 5%
                reward -= abs(unrealized_pnl) * 10

        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # Portfolio value for info
        portfolio_value = self.balance + (self.position * current_price if self.position > 0 else 0)

        info = {
            "portfolio_value": portfolio_value,
            "balance": self.balance,
            "position": self.position,
            "current_price": current_price,
            "total_trades": self.total_trades,
            "trade_executed": trade_executed,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1

        row = self.df.iloc[self.current_step]

        # Get features
        features = row[self.feature_columns].values.astype(np.float32)

        # Add portfolio state
        current_price = row["close"]
        portfolio_value = self.balance + (self.position * current_price if self.position > 0 else 0)

        balance_ratio = self.balance / self.initial_balance
        position_ratio = (self.position * current_price) / portfolio_value if portfolio_value > 0 else 0

        unrealized_pnl = 0.0
        if self.position > 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price

        portfolio_state = np.array([balance_ratio, position_ratio, unrealized_pnl], dtype=np.float32)

        # Combine features and portfolio state
        observation = np.concatenate([features, portfolio_state])

        # Handle NaN/Inf
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        return observation

    def render(self):
        """Render current state (for debugging)."""
        current_price = self.df.loc[self.current_step, "close"]
        portfolio_value = self.balance + (self.position * current_price if self.position > 0 else 0)
        print(f"Step {self.current_step}: Price=${current_price:.2f}, Portfolio=${portfolio_value:.2f}")


class RLTradingAgent:
    """
    Reinforcement Learning trading agent using PPO.

    Features:
    - Learns optimal trading through experience
    - Provides action confidence scores
    - Supports continuous learning
    - Auto-saves/loads models
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize RL agent.

        Args:
            model_path: Path to load existing model (optional)
        """
        self.model: Optional[PPO] = None
        self.env: Optional[TradingEnvironment] = None
        self.model_path = model_path or "models/rl_agent.zip"
        self.is_trained = False
        self.feature_columns: list = []

        # PPO hyperparameters (upgraded for VPS hardware)
        self.ppo_params = {
            "learning_rate": 0.0001,
            "n_steps": 4096,
            "batch_size": 128,
            "n_epochs": 15,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.03,  # Slightly lower entropy - bigger network explores naturally
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 0,
            "policy_kwargs": dict(net_arch=[256, 128]),  # Bigger network (was 64, 64)
        }

        # Try to load existing model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        total_timesteps: int = 100000,
        initial_balance: float = 1000.0
    ) -> dict:
        """
        Train the RL agent on historical data.

        Args:
            df: DataFrame with OHLCV and features
            feature_columns: List of feature column names
            total_timesteps: Total training steps
            initial_balance: Starting balance for simulation

        Returns:
            dict with training metrics
        """
        logger.info(f"Training RL agent for {total_timesteps} timesteps")

        self.feature_columns = feature_columns

        # Create environment
        self.env = TradingEnvironment(
            df=df,
            feature_columns=feature_columns,
            initial_balance=initial_balance
        )

        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            **self.ppo_params
        )

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )

        self.is_trained = True

        # Save model first (so it persists even if evaluation fails)
        self.save_model()

        # Evaluate
        metrics = self._evaluate_training(df, feature_columns, initial_balance)

        logger.info(f"RL training complete - Final Return: {metrics['total_return']:.2%}")

        return metrics

    def _evaluate_training(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        initial_balance: float
    ) -> dict:
        """Evaluate trained model performance."""
        env = TradingEnvironment(
            df=df,
            feature_columns=feature_columns,
            initial_balance=initial_balance
        )

        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        final_value = info["portfolio_value"]
        total_return = (final_value - initial_balance) / initial_balance

        return {
            "total_return": total_return,
            "final_portfolio_value": final_value,
            "total_trades": info["total_trades"],
            "total_reward": total_reward
        }

    def decide(self, features: np.ndarray, portfolio_state: dict) -> Tuple[int, float]:
        """
        Make a trading decision.

        Args:
            features: Feature array from FeatureEngine
            portfolio_state: Dict with balance, position, entry_price, current_price

        Returns:
            Tuple of (action, confidence)
            action: 1=BUY, 0=HOLD, -1=SELL
            confidence: 0.0 to 1.0
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained! Call train() first or load a saved model.")

        # Build observation
        balance_ratio = portfolio_state.get("balance", 1000) / 1000
        position_value = portfolio_state.get("position", 0) * portfolio_state.get("current_price", 0)
        portfolio_value = portfolio_state.get("balance", 1000) + position_value
        position_ratio = position_value / portfolio_value if portfolio_value > 0 else 0

        unrealized_pnl = 0.0
        if portfolio_state.get("position", 0) > 0 and portfolio_state.get("entry_price", 0) > 0:
            unrealized_pnl = (
                portfolio_state["current_price"] - portfolio_state["entry_price"]
            ) / portfolio_state["entry_price"]

        # Combine features with portfolio state
        portfolio_features = np.array([balance_ratio, position_ratio, unrealized_pnl], dtype=np.float32)
        observation = np.concatenate([features.flatten(), portfolio_features])

        # Handle NaN/Inf
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        # Get action and value from model
        action, _ = self.model.predict(observation, deterministic=True)

        # Get action probabilities for confidence using policy distribution
        try:
            import torch
            obs_tensor = torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32)
            obs_tensor = obs_tensor.to(self.model.policy.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
                confidence = float(probs[int(action)])
        except Exception as e:
            logger.debug(f"RL confidence fallback: {e}")
            confidence = 0.6  # Default confidence

        # Convert action: 0=HOLD, 1=BUY, 2=SELL -> 0=HOLD, 1=BUY, -1=SELL
        action_map = {0: 0, 1: 1, 2: -1}
        mapped_action = action_map.get(int(action), 0)

        return mapped_action, confidence

    def save_model(self, path: Optional[str] = None):
        """Save model to file."""
        path = path or self.model_path

        if self.model is None:
            raise ValueError("No model to save!")

        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save PPO model
        self.model.save(path.replace(".zip", ""))

        # Save metadata
        import pickle
        meta_path = path.replace(".zip", "_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "feature_columns": self.feature_columns,
                "ppo_params": self.ppo_params
            }, f)

        logger.info(f"RL model saved to {path}")

    def load_model(self, path: Optional[str] = None):
        """Load model from file."""
        path = path or self.model_path
        model_file = path.replace(".zip", "")

        if not os.path.exists(model_file + ".zip"):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load PPO model (needs a dummy env)
        # We'll create a proper env when actually using the model
        self.model = PPO.load(model_file)

        # Load metadata
        import pickle
        meta_path = path.replace(".zip", "_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.feature_columns = meta.get("feature_columns", [])
                self.ppo_params = meta.get("ppo_params", self.ppo_params)

        self.is_trained = True
        logger.info(f"RL model loaded from {path}")
