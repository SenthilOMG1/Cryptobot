"""
Reinforcement Learning Trading Agent
====================================
PPO-based agent that learns optimal trading through experience.
This is the second AI brain that votes on trading decisions.

V2 Upgrades:
- Rich multi-component reward function
- Checkpoint callback for resume after interruption
- Resume training from checkpoint support
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
from collections import deque
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class CheckpointCallback(BaseCallback):
    """Save model checkpoints during training for resume capability."""

    def __init__(self, save_path: str = "models/checkpoints", save_freq: int = 250000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step_path = os.path.join(self.save_path, f"rl_step_{self.n_calls}")
            latest_path = os.path.join(self.save_path, "rl_latest")
            self.model.save(step_path)
            self.model.save(latest_path)
            if self.verbose:
                logger.info(f"RL checkpoint saved at step {self.n_calls}")
        return True


class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for crypto trading.

    State: Technical indicators + portfolio state
    Actions: 0=HOLD, 1=BUY, 2=SELL
    Reward: Rich multi-component reward function
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

        # Precompute trend data for trend-alignment reward
        self._precompute_trend()

        # State space: features + [balance_ratio, position_ratio, unrealized_pnl, side_indicator]
        n_features = len(feature_columns) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features,), dtype=np.float32
        )

        # Action space: 0=HOLD, 1=BUY(long), 2=SELL(short)
        self.action_space = spaces.Discrete(3)

        # Trading state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        # Rich reward tracking state
        self.steps_in_position = 0
        self.recent_trades = deque(maxlen=50)  # (step, pnl_pct) for overtrading detection
        self.episode_returns = deque(maxlen=20)  # Rolling returns for Sharpe
        self.direction_counts = {"long": 0, "short": 0}  # Concentration tracking

    def _precompute_trend(self):
        """Precompute trend direction for each step from price vs SMA50."""
        self.trend_direction = np.zeros(len(self.df))
        if "price_vs_sma50" in self.df.columns:
            pvs = self.df["price_vs_sma50"].values
            self.trend_direction = np.where(pvs > 0.01, 1, np.where(pvs < -0.01, -1, 0))
        elif "sma_50" in self.df.columns:
            close = self.df["close"].values
            sma = self.df["sma_50"].values
            ratio = (close - sma) / np.where(sma > 0, sma, 1)
            self.trend_direction = np.where(ratio > 0.01, 1, np.where(ratio < -0.01, -1, 0))

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        # Reset rich reward state
        self.steps_in_position = 0
        self.recent_trades.clear()
        self.episode_returns.clear()
        self.direction_counts = {"long": 0, "short": 0}

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one trading step with rich reward function."""
        current_price = self.df.loc[self.current_step, "close"]

        reward = 0.0
        trade_executed = False
        trade_pnl = 0.0

        if action == 1:  # BUY
            if self.position_side == -1:
                # Close short position
                pnl_pct = (self.entry_price - current_price) / self.entry_price
                fee = self.position * current_price * self.transaction_fee
                self.balance += self.position * self.entry_price + (pnl_pct * self.position * self.entry_price) - fee
                trade_pnl = pnl_pct
                reward += pnl_pct * 100  # Base P&L reward
                if pnl_pct > 0:
                    self.winning_trades += 1
                self.position = 0.0
                self.position_side = 0
                self.entry_price = 0.0
                self.total_trades += 1
                self.steps_in_position = 0
                trade_executed = True
            elif self.position_side == 0 and self.balance > 0:
                # Open long position
                trade_amount = self.balance * self.max_position_pct
                fee = trade_amount * self.transaction_fee
                self.position = (trade_amount - fee) / current_price
                self.balance -= trade_amount
                self.entry_price = current_price
                self.position_side = 1
                self.total_trades += 1
                self.direction_counts["long"] += 1
                self.steps_in_position = 0
                trade_executed = True

        elif action == 2:  # SELL
            if self.position_side == 1:
                # Close long position
                sale_value = self.position * current_price
                fee = sale_value * self.transaction_fee
                self.balance += (sale_value - fee)
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                trade_pnl = pnl_pct
                reward += pnl_pct * 100  # Base P&L reward
                if pnl_pct > 0:
                    self.winning_trades += 1
                self.position = 0.0
                self.position_side = 0
                self.entry_price = 0.0
                self.total_trades += 1
                self.steps_in_position = 0
                trade_executed = True
            elif self.position_side == 0 and self.balance > 0:
                # Open short position
                trade_amount = self.balance * self.max_position_pct
                fee = trade_amount * self.transaction_fee
                self.position = (trade_amount - fee) / current_price
                self.balance -= trade_amount
                self.entry_price = current_price
                self.position_side = -1
                self.total_trades += 1
                self.direction_counts["short"] += 1
                self.steps_in_position = 0
                trade_executed = True

        # Track position hold time
        if self.position_side != 0:
            self.steps_in_position += 1

        # Record trade for overtrading detection
        if trade_executed:
            self.recent_trades.append((self.current_step, trade_pnl))
            if trade_pnl != 0:
                self.episode_returns.append(trade_pnl)

        # ===== RICH REWARD COMPONENTS =====
        reward += self._calculate_rich_reward(action, trade_executed, trade_pnl, current_price)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # Portfolio value
        if self.position_side == 1:
            pos_value = self.position * current_price
        elif self.position_side == -1:
            pnl = (self.entry_price - current_price) * self.position
            pos_value = self.position * self.entry_price + pnl
        else:
            pos_value = 0
        portfolio_value = self.balance + pos_value

        info = {
            "portfolio_value": portfolio_value,
            "balance": self.balance,
            "position": self.position,
            "position_side": self.position_side,
            "current_price": current_price,
            "total_trades": self.total_trades,
            "trade_executed": trade_executed,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _calculate_rich_reward(self, action: int, trade_executed: bool, trade_pnl: float, current_price: float) -> float:
        """
        Calculate additional reward components beyond base P&L.

        Components:
        1. Trend alignment bonus/penalty
        2. Time penalty for holding losers
        3. Overtrading penalty
        4. Sharpe bonus for consistency
        5. Concentration penalty
        """
        reward = 0.0

        # 1. Trend alignment
        if self.current_step < len(self.trend_direction):
            trend = self.trend_direction[self.current_step]
            if self.position_side != 0:
                # +0.3 for trading with trend, -0.5 for counter-trend
                if self.position_side == trend:
                    reward += 0.3
                elif trend != 0 and self.position_side == -trend:
                    reward -= 0.5

        # 2. Time penalty for holding losers
        if self.position_side != 0 and self.entry_price > 0 and self.steps_in_position > 0:
            if self.position_side == 1:
                unrealized = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized = (self.entry_price - current_price) / self.entry_price

            if unrealized < -0.02:  # Losing > 2%
                # Penalty increases with hold time, capped at 48 steps
                hold_factor = min(self.steps_in_position / 48.0, 1.0)
                reward -= abs(unrealized) * hold_factor * 5.0

        # 3. Overtrading penalty
        if trade_executed:
            # Count trades in last 24 steps
            recent_count = sum(1 for step, _ in self.recent_trades
                             if self.current_step - step < 24)
            if recent_count > 4:
                reward -= 0.3 * (recent_count - 4)

        # 4. Sharpe bonus for consistent returns
        if len(self.episode_returns) >= 5 and trade_pnl != 0:
            returns_arr = np.array(list(self.episode_returns))
            mean_ret = np.mean(returns_arr)
            std_ret = np.std(returns_arr)
            if std_ret > 0:
                sharpe = mean_ret / std_ret
                reward += sharpe * 0.5  # Scale Sharpe contribution

        # 5. Concentration penalty
        total_dir = self.direction_counts["long"] + self.direction_counts["short"]
        if total_dir > 5:
            long_pct = self.direction_counts["long"] / total_dir
            short_pct = self.direction_counts["short"] / total_dir
            if long_pct > 0.80 or short_pct > 0.80:
                reward -= 0.3

        # 6. Target alignment (from original)
        if self.has_targets and self.current_step < len(self.targets):
            target = self.targets[self.current_step]
            action_map = {0: 0, 1: 1, 2: -1}
            rl_signal = action_map[int(action)]
            if rl_signal == target and target != 0:
                reward += 0.2

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1

        row = self.df.iloc[self.current_step]

        # Get features
        features = row[self.feature_columns].values.astype(np.float32)

        # Add portfolio state
        current_price = row["close"]
        if self.position_side == 1:
            pos_value = self.position * current_price
        elif self.position_side == -1:
            pnl = (self.entry_price - current_price) * self.position
            pos_value = self.position * self.entry_price + pnl
        else:
            pos_value = 0
        portfolio_value = self.balance + pos_value

        balance_ratio = self.balance / self.initial_balance
        position_ratio = pos_value / portfolio_value if portfolio_value > 0 else 0

        unrealized_pnl = 0.0
        if self.position > 0 and self.entry_price > 0:
            if self.position_side == 1:
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price

        side_indicator = float(self.position_side)

        portfolio_state = np.array([balance_ratio, position_ratio, unrealized_pnl, side_indicator], dtype=np.float32)

        observation = np.concatenate([features, portfolio_state])
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
    - Rich multi-component reward function
    - Checkpoint saving for training resume
    - Provides action confidence scores
    - Supports continuous learning
    """

    def __init__(self, model_path: Optional[str] = None):
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
            "ent_coef": 0.03,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 0,
            "policy_kwargs": dict(net_arch=[256, 128]),
        }

        # Try to load existing model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        total_timesteps: int = 100000,
        initial_balance: float = 1000.0,
        resume_from: Optional[str] = None
    ) -> dict:
        """
        Train the RL agent on historical data.

        Args:
            df: DataFrame with OHLCV and features
            feature_columns: List of feature column names
            total_timesteps: Total training steps
            initial_balance: Starting balance for simulation
            resume_from: Path to checkpoint to resume from (optional)

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

        # Resume from checkpoint or create new model
        if resume_from and os.path.exists(resume_from + ".zip"):
            logger.info(f"Resuming RL training from checkpoint: {resume_from}")
            self.model = PPO.load(resume_from, env=self.env)
            reset_num_timesteps = False
        else:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                **self.ppo_params
            )
            reset_num_timesteps = True

        # Setup checkpoint callback
        checkpoint_cb = CheckpointCallback(
            save_path="models/checkpoints",
            save_freq=250000,
            verbose=1
        )

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=checkpoint_cb,
            reset_num_timesteps=reset_num_timesteps
        )

        self.is_trained = True

        # Save model
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

        side_indicator = float(portfolio_state.get("side", 0))

        portfolio_features = np.array([balance_ratio, position_ratio, unrealized_pnl, side_indicator], dtype=np.float32)
        observation = np.concatenate([features.flatten(), portfolio_features])
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        # Get action and value from model
        action, _ = self.model.predict(observation, deterministic=True)

        # Get action probabilities for confidence
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
            confidence = 0.6

        # Convert action: 0=HOLD, 1=BUY, 2=SELL -> 0=HOLD, 1=BUY, -1=SELL
        action_map = {0: 0, 1: 1, 2: -1}
        mapped_action = action_map.get(int(action), 0)

        return mapped_action, confidence

    def save_model(self, path: Optional[str] = None):
        """Save model to file."""
        path = path or self.model_path

        if self.model is None:
            raise ValueError("No model to save!")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.model.save(path.replace(".zip", ""))

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

        self.model = PPO.load(model_file)

        import pickle
        meta_path = path.replace(".zip", "_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.feature_columns = meta.get("feature_columns", [])
                self.ppo_params = meta.get("ppo_params", self.ppo_params)

        self.is_trained = True
        logger.info(f"RL model loaded from {path}")
