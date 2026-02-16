#!/usr/bin/env python3
"""
Comprehensive Backtest Simulation
==================================
Loads actual XGBoost + RL models, fetches 30 days of historical data,
and simulates the full trading strategy with ensemble decisions,
dynamic leverage, confidence-based sizing, stop-loss, trailing stop,
and take-profit logic.

Tracks both LONG and SHORT signals on the top 6 pairs.
"""

import os
import sys
import time
import logging
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, "/root/Cryptobot")

from dotenv import load_dotenv
load_dotenv("/root/Cryptobot/.env")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest")

# Suppress noisy loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("okx").setLevel(logging.WARNING)

# ============================================================
# SETTINGS (matching live bot)
# ============================================================
MIN_CONFIDENCE = 0.50
STOP_LOSS_PCT = 0.03       # 3%
TRAILING_STOP_PCT = 0.03   # 3%
TAKE_PROFIT_PCT = 0.20     # 20%
MAX_OPEN_POSITIONS = 8
STARTING_BALANCE = 1000.0  # USDT
TRADING_FEE = 0.001        # 0.1% per side (OKX maker)
BACKTEST_DAYS = 30
PAIRS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT"]


def get_leverage(confidence: float) -> int:
    """Dynamic leverage based on confidence."""
    if confidence >= 0.75:
        return 7
    elif confidence >= 0.65:
        return 5
    elif confidence >= 0.55:
        return 3
    else:
        return 2


def get_position_size_pct(confidence: float) -> float:
    """Position size % of balance based on confidence."""
    if confidence >= 0.75:
        return 20.0
    elif confidence >= 0.65:
        return 15.0
    elif confidence >= 0.55:
        return 10.0
    else:
        return 5.0


@dataclass
class Position:
    """Tracks an open position."""
    pair: str
    side: str           # "long" or "short"
    entry_price: float
    size_usdt: float    # Notional size (including leverage)
    leverage: int
    confidence: float
    entry_time: datetime
    peak_price: float = 0.0    # For trailing stop (longs)
    trough_price: float = 0.0  # For trailing stop (shorts)
    margin_usdt: float = 0.0   # Actual capital locked

    def __post_init__(self):
        self.margin_usdt = self.size_usdt / self.leverage
        if self.side == "long":
            self.peak_price = self.entry_price
        else:
            self.trough_price = self.entry_price


@dataclass
class ClosedTrade:
    """A completed trade."""
    pair: str
    side: str
    entry_price: float
    exit_price: float
    size_usdt: float
    leverage: int
    confidence: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str
    pnl_usdt: float = 0.0
    pnl_pct: float = 0.0
    hold_hours: float = 0.0


# ============================================================
# DATA FETCHING (direct OKX API, no client wrapper)
# ============================================================
def fetch_candles_okx(pair: str, bar: str, limit: int, after: str = None) -> list:
    """Fetch candles from OKX public API."""
    import okx.MarketData as MarketData
    api = MarketData.MarketAPI(flag="0")
    params = {"instId": pair, "bar": bar, "limit": str(min(limit, 300))}
    if after:
        params["after"] = str(after)
    response = api.get_candlesticks(**params)
    if response and response.get("code") == "0":
        return response.get("data", [])
    else:
        code = response.get("code", "?") if response else "None"
        msg = response.get("msg", "") if response else "No response"
        logger.warning(f"OKX API error for {pair} {bar}: [{code}] {msg}")
        return []


def fetch_historical(pair: str, timeframe: str, days: int) -> pd.DataFrame:
    """Fetch historical candles with pagination."""
    bar_map = {"1h": "1H", "4h": "4H", "1d": "1D"}
    bar = bar_map.get(timeframe, timeframe)
    hours_map = {"1h": 1, "4h": 4, "1d": 24}
    hours_per = hours_map.get(timeframe, 1)
    target_candles = int((days * 24) / hours_per) + 50  # Extra for warm-up

    all_candles = []
    after_ts = None
    retries = 0

    while len(all_candles) < target_candles and retries < 20:
        batch = fetch_candles_okx(pair, bar, 300, after=after_ts)
        if not batch:
            retries += 1
            time.sleep(0.5)
            continue

        all_candles.extend(batch)
        after_ts = batch[-1][0]  # Oldest timestamp for pagination
        time.sleep(0.12)  # Rate limit

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=[
        "timestamp", "open", "high", "low", "close",
        "volume", "volume_ccy", "volume_quote", "confirm"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


# ============================================================
# LOAD MODELS
# ============================================================
def load_models():
    """Load the actual XGBoost and RL models."""
    from src.models.xgboost_model import XGBoostPredictor
    from src.models.rl_agent import RLTradingAgent

    xgb_path = "/root/Cryptobot/models/xgboost_model.json"
    rl_path = "/root/Cryptobot/models/rl_agent.zip"

    logger.info("Loading XGBoost model...")
    xgb_model = XGBoostPredictor(model_path=xgb_path)
    if not xgb_model.is_trained:
        raise RuntimeError(f"XGBoost model not loaded from {xgb_path}")
    logger.info(f"  XGBoost loaded: {len(xgb_model.feature_names)} features")

    logger.info("Loading RL agent...")
    rl_agent = RLTradingAgent(model_path=rl_path)
    if not rl_agent.is_trained:
        raise RuntimeError(f"RL agent not loaded from {rl_path}")
    logger.info(f"  RL agent loaded: {len(rl_agent.feature_columns)} features")

    return xgb_model, rl_agent


# ============================================================
# ENSEMBLE LOGIC (matching live ensemble.py)
# ============================================================
def ensemble_decide(xgb_model, rl_agent, features_row: pd.DataFrame,
                    portfolio_state: dict, pair: str) -> Tuple[int, float, str]:
    """
    Run ensemble decision matching the live logic.
    Returns (action, confidence, reasoning).
    action: 1=BUY, -1=SELL, 0=HOLD
    """
    # XGBoost prediction
    try:
        xgb_action, xgb_conf = xgb_model.predict(features_row)
    except Exception as e:
        xgb_action, xgb_conf = 0, 0.5

    # RL prediction
    try:
        feature_cols = xgb_model.feature_names
        feature_array = features_row[feature_cols].values.flatten().astype(np.float32)
        rl_action, rl_conf = rl_agent.decide(feature_array, portfolio_state)
    except Exception as e:
        rl_action, rl_conf = 0, 0.5

    xgb_w, rl_w = 0.5, 0.5
    weighted_conf = xgb_conf * xgb_w + rl_conf * rl_w

    # Both agree BUY
    if xgb_action == 1 and rl_action == 1:
        if weighted_conf >= MIN_CONFIDENCE:
            return 1, weighted_conf, f"Both BUY (xgb={xgb_conf:.2f}, rl={rl_conf:.2f})"
        return 0, 0.0, "Both BUY but low conf"

    # Both agree SELL
    if xgb_action == -1 and rl_action == -1:
        if weighted_conf >= MIN_CONFIDENCE:
            return -1, weighted_conf, f"Both SELL (xgb={xgb_conf:.2f}, rl={rl_conf:.2f})"
        return 0, 0.0, "Both SELL but low conf"

    # Single model override at >80%
    if xgb_action != 0 and xgb_conf > 0.80:
        adj_conf = xgb_conf * 0.8
        if adj_conf >= MIN_CONFIDENCE:
            return xgb_action, adj_conf, f"XGB override ({xgb_conf:.2f})"

    if rl_action != 0 and rl_conf > 0.80:
        adj_conf = rl_conf * 0.8
        if adj_conf >= MIN_CONFIDENCE:
            return rl_action, adj_conf, f"RL override ({rl_conf:.2f})"

    return 0, 0.0, "Disagree/HOLD"


# ============================================================
# BACKTEST ENGINE
# ============================================================
class BacktestEngine:
    """Simulates the full trading strategy on historical data."""

    def __init__(self, xgb_model, rl_agent, starting_balance: float = STARTING_BALANCE):
        self.xgb_model = xgb_model
        self.rl_agent = rl_agent
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.open_positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.decisions_log: List[dict] = []
        self.peak_equity = starting_balance

    def _get_portfolio_state(self, pair: str, current_price: float) -> dict:
        """Build portfolio_state dict for RL agent."""
        pos = self.open_positions.get(pair)
        if pos:
            return {
                "balance": self.balance,
                "position": pos.size_usdt / current_price if current_price > 0 else 0,
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "side": 1 if pos.side == "long" else -1,
            }
        return {
            "balance": self.balance,
            "position": 0,
            "entry_price": 0,
            "current_price": current_price,
            "side": 0,
        }

    def _check_exit_conditions(self, pos: Position, current_price: float,
                                timestamp: datetime) -> Optional[str]:
        """Check stop-loss, trailing stop, and take-profit."""
        if pos.side == "long":
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price

            # Update peak for trailing stop
            if current_price > pos.peak_price:
                pos.peak_price = current_price

            # Stop-loss
            if pnl_pct <= -STOP_LOSS_PCT:
                return "stop_loss"

            # Take-profit
            if pnl_pct >= TAKE_PROFIT_PCT:
                return "take_profit"

            # Trailing stop: price drops TRAILING_STOP_PCT from peak
            trail_drop = (pos.peak_price - current_price) / pos.peak_price
            if trail_drop >= TRAILING_STOP_PCT and pnl_pct > 0:
                return "trailing_stop"

        else:  # short
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price

            # Update trough for trailing stop
            if current_price < pos.trough_price:
                pos.trough_price = current_price

            # Stop-loss
            if pnl_pct <= -STOP_LOSS_PCT:
                return "stop_loss"

            # Take-profit
            if pnl_pct >= TAKE_PROFIT_PCT:
                return "take_profit"

            # Trailing stop: price rises TRAILING_STOP_PCT from trough
            trail_rise = (current_price - pos.trough_price) / pos.trough_price
            if trail_rise >= TRAILING_STOP_PCT and pnl_pct > 0:
                return "trailing_stop"

        return None

    def _close_position(self, pair: str, current_price: float,
                        timestamp: datetime, reason: str):
        """Close a position and record the trade."""
        pos = self.open_positions.pop(pair)

        if pos.side == "long":
            raw_pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        else:
            raw_pnl_pct = (pos.entry_price - current_price) / pos.entry_price

        # P&L with leverage and fees
        leveraged_pnl_pct = raw_pnl_pct * pos.leverage
        fee_cost = pos.size_usdt * TRADING_FEE * 2  # Entry + exit fee
        pnl_usdt = (pos.margin_usdt * leveraged_pnl_pct) - fee_cost

        # Return margin + pnl to balance
        self.balance += pos.margin_usdt + pnl_usdt

        hold_hours = (timestamp - pos.entry_time).total_seconds() / 3600

        trade = ClosedTrade(
            pair=pair,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=current_price,
            size_usdt=pos.size_usdt,
            leverage=pos.leverage,
            confidence=pos.confidence,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            exit_reason=reason,
            pnl_usdt=pnl_usdt,
            pnl_pct=leveraged_pnl_pct * 100,
            hold_hours=hold_hours,
        )
        self.closed_trades.append(trade)

    def _open_position(self, pair: str, side: str, confidence: float,
                       current_price: float, timestamp: datetime):
        """Open a new position."""
        if pair in self.open_positions:
            return  # Already have a position

        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return  # Max positions reached

        size_pct = get_position_size_pct(confidence) / 100.0
        leverage = get_leverage(confidence)
        margin = self.balance * size_pct

        if margin < 5.0:  # Min trade size
            return

        notional = margin * leverage

        pos = Position(
            pair=pair,
            side=side,
            entry_price=current_price,
            size_usdt=notional,
            leverage=leverage,
            confidence=confidence,
            entry_time=timestamp,
        )

        self.balance -= margin
        self.open_positions[pair] = pos

    def _calc_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity (balance + open positions value)."""
        equity = self.balance
        for pair, pos in self.open_positions.items():
            price = prices.get(pair, pos.entry_price)
            if pos.side == "long":
                raw_pnl_pct = (price - pos.entry_price) / pos.entry_price
            else:
                raw_pnl_pct = (pos.entry_price - price) / pos.entry_price
            leveraged_pnl = raw_pnl_pct * pos.leverage
            equity += pos.margin_usdt * (1 + leveraged_pnl)
        return equity

    def run(self, all_data: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Run the full backtest.

        all_data: {pair: {"features_df": df_with_features, "1h": raw_1h_df}}
        The features_df has columns: timestamp, open, high, low, close, volume, + all features
        We iterate candle by candle over the last 30 days.
        """
        logger.info("="*70)
        logger.info("STARTING BACKTEST SIMULATION")
        logger.info(f"  Balance: ${self.starting_balance:.2f}")
        logger.info(f"  Pairs: {list(all_data.keys())}")
        logger.info(f"  Period: last {BACKTEST_DAYS} days")
        logger.info(f"  Min Confidence: {MIN_CONFIDENCE}")
        logger.info(f"  Stop Loss: {STOP_LOSS_PCT*100:.0f}% | Trailing: {TRAILING_STOP_PCT*100:.0f}% | TP: {TAKE_PROFIT_PCT*100:.0f}%")
        logger.info("="*70)

        # Find the common time range (last 30 days of features data)
        cutoff = datetime.utcnow() - timedelta(days=BACKTEST_DAYS)

        # Build per-pair iterators: list of (timestamp, row) for features
        pair_rows = {}
        for pair, data in all_data.items():
            fdf = data["features_df"]
            fdf = fdf[fdf["timestamp"] >= cutoff].reset_index(drop=True)
            if len(fdf) < 10:
                logger.warning(f"  {pair}: only {len(fdf)} rows after cutoff, skipping")
                continue
            pair_rows[pair] = fdf
            logger.info(f"  {pair}: {len(fdf)} candles for backtest")

        if not pair_rows:
            logger.error("No data available for backtest!")
            return

        # Collect all unique timestamps, sort them
        all_timestamps = set()
        for pair, fdf in pair_rows.items():
            all_timestamps.update(fdf["timestamp"].tolist())
        all_timestamps = sorted(all_timestamps)

        logger.info(f"\nSimulating {len(all_timestamps)} hourly candles from {all_timestamps[0]} to {all_timestamps[-1]}")
        logger.info("")

        total_decisions = 0
        signal_count = 0

        for ts in all_timestamps:
            current_prices = {}

            # First: check exit conditions for all open positions
            for pair in list(self.open_positions.keys()):
                if pair not in pair_rows:
                    continue
                fdf = pair_rows[pair]
                row_match = fdf[fdf["timestamp"] == ts]
                if row_match.empty:
                    continue
                price = row_match["close"].iloc[0]
                current_prices[pair] = price

                exit_reason = self._check_exit_conditions(
                    self.open_positions[pair], price, ts
                )
                if exit_reason:
                    self._close_position(pair, price, ts, exit_reason)

            # Second: get ensemble decisions for each pair
            for pair, fdf in pair_rows.items():
                row_match = fdf[fdf["timestamp"] == ts]
                if row_match.empty:
                    continue

                row = row_match.iloc[[0]]  # Keep as DataFrame
                price = row["close"].iloc[0]
                current_prices[pair] = price

                portfolio_state = self._get_portfolio_state(pair, price)

                action, confidence, reasoning = ensemble_decide(
                    self.xgb_model, self.rl_agent, row, portfolio_state, pair
                )

                total_decisions += 1

                self.decisions_log.append({
                    "timestamp": ts,
                    "pair": pair,
                    "action": action,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "price": price,
                })

                if action == 0:
                    continue

                signal_count += 1

                # Handle signals
                if action == 1:  # BUY signal
                    if pair in self.open_positions and self.open_positions[pair].side == "short":
                        # Close short, then open long
                        self._close_position(pair, price, ts, "signal_flip")
                    if pair not in self.open_positions:
                        self._open_position(pair, "long", confidence, price, ts)

                elif action == -1:  # SELL signal
                    if pair in self.open_positions and self.open_positions[pair].side == "long":
                        # Close long, then open short
                        self._close_position(pair, price, ts, "signal_flip")
                    if pair not in self.open_positions:
                        self._open_position(pair, "short", confidence, price, ts)

            # Record equity
            equity = self._calc_equity(current_prices)
            self.equity_curve.append((ts, equity))
            if equity > self.peak_equity:
                self.peak_equity = equity

        # Close any remaining open positions at last known price
        for pair in list(self.open_positions.keys()):
            if pair in current_prices:
                self._close_position(pair, current_prices[pair], all_timestamps[-1], "backtest_end")

        logger.info(f"\nProcessed {total_decisions} decisions, {signal_count} signals generated")

    def report(self):
        """Print comprehensive backtest results."""
        print("\n" + "="*70)
        print("               BACKTEST RESULTS REPORT")
        print("="*70)

        trades = self.closed_trades
        n_trades = len(trades)

        if n_trades == 0:
            print("\n  NO TRADES EXECUTED during backtest period.")
            print("  This could mean:")
            print("    - Models are very conservative at current confidence threshold")
            print("    - Models disagree frequently (ensemble HOLD)")
            self._print_decision_summary()
            return

        # --- Overall P&L ---
        total_pnl = sum(t.pnl_usdt for t in trades)
        total_return = (self.balance - self.starting_balance) / self.starting_balance * 100
        wins = [t for t in trades if t.pnl_usdt > 0]
        losses = [t for t in trades if t.pnl_usdt <= 0]
        win_rate = len(wins) / n_trades * 100

        print(f"\n  Starting Balance:   ${self.starting_balance:>10.2f}")
        print(f"  Final Balance:      ${self.balance:>10.2f}")
        print(f"  Total P&L:          ${total_pnl:>10.2f} ({total_return:+.2f}%)")
        print(f"  Peak Equity:        ${self.peak_equity:>10.2f}")

        # --- Max Drawdown ---
        max_dd = 0
        peak = self.starting_balance
        for ts, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        print(f"  Max Drawdown:       {max_dd*100:>10.2f}%")

        # --- Trade Statistics ---
        print(f"\n  {'='*50}")
        print(f"  TRADE STATISTICS")
        print(f"  {'='*50}")
        print(f"  Total Trades:       {n_trades}")
        print(f"  Winning Trades:     {len(wins)} ({win_rate:.1f}%)")
        print(f"  Losing Trades:      {len(losses)} ({100-win_rate:.1f}%)")

        avg_win = np.mean([t.pnl_usdt for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_usdt for t in losses]) if losses else 0
        print(f"  Avg Win:            ${avg_win:>10.2f}")
        print(f"  Avg Loss:           ${avg_loss:>10.2f}")

        if avg_loss != 0:
            profit_factor = abs(sum(t.pnl_usdt for t in wins) / sum(t.pnl_usdt for t in losses)) if losses else float('inf')
            print(f"  Profit Factor:      {profit_factor:>10.2f}")

        avg_hold = np.mean([t.hold_hours for t in trades])
        median_hold = np.median([t.hold_hours for t in trades])
        print(f"  Avg Hold Time:      {avg_hold:>10.1f} hours")
        print(f"  Median Hold Time:   {median_hold:>10.1f} hours")

        # --- Long vs Short ---
        longs = [t for t in trades if t.side == "long"]
        shorts = [t for t in trades if t.side == "short"]
        print(f"\n  {'='*50}")
        print(f"  LONG vs SHORT BREAKDOWN")
        print(f"  {'='*50}")

        if longs:
            long_wins = len([t for t in longs if t.pnl_usdt > 0])
            long_pnl = sum(t.pnl_usdt for t in longs)
            print(f"  Long Trades:   {len(longs):>4} | Win Rate: {long_wins/len(longs)*100:>5.1f}% | P&L: ${long_pnl:>8.2f}")
        else:
            print(f"  Long Trades:      0")

        if shorts:
            short_wins = len([t for t in shorts if t.pnl_usdt > 0])
            short_pnl = sum(t.pnl_usdt for t in shorts)
            print(f"  Short Trades:  {len(shorts):>4} | Win Rate: {short_wins/len(shorts)*100:>5.1f}% | P&L: ${short_pnl:>8.2f}")
        else:
            print(f"  Short Trades:     0")

        # --- By Exit Reason ---
        print(f"\n  {'='*50}")
        print(f"  EXIT REASONS")
        print(f"  {'='*50}")
        from collections import Counter
        reasons = Counter(t.exit_reason for t in trades)
        for reason, count in reasons.most_common():
            subset = [t for t in trades if t.exit_reason == reason]
            rpnl = sum(t.pnl_usdt for t in subset)
            print(f"  {reason:<20s}: {count:>4} trades | P&L: ${rpnl:>8.2f}")

        # --- By Pair ---
        print(f"\n  {'='*50}")
        print(f"  PER-PAIR PERFORMANCE")
        print(f"  {'='*50}")
        pair_groups = {}
        for t in trades:
            pair_groups.setdefault(t.pair, []).append(t)

        for pair in sorted(pair_groups.keys()):
            pt = pair_groups[pair]
            ppnl = sum(t.pnl_usdt for t in pt)
            pwr = len([t for t in pt if t.pnl_usdt > 0]) / len(pt) * 100
            print(f"  {pair:<12s}: {len(pt):>3} trades | WR: {pwr:>5.1f}% | P&L: ${ppnl:>8.2f}")

        # --- Leverage Distribution ---
        print(f"\n  {'='*50}")
        print(f"  LEVERAGE DISTRIBUTION")
        print(f"  {'='*50}")
        lev_groups = {}
        for t in trades:
            lev_groups.setdefault(t.leverage, []).append(t)
        for lev in sorted(lev_groups.keys()):
            lt = lev_groups[lev]
            lpnl = sum(t.pnl_usdt for t in lt)
            lwr = len([t for t in lt if t.pnl_usdt > 0]) / len(lt) * 100
            print(f"  {lev}x leverage: {len(lt):>3} trades | WR: {lwr:>5.1f}% | P&L: ${lpnl:>8.2f}")

        # --- Top 5 Best and Worst Trades ---
        print(f"\n  {'='*50}")
        print(f"  TOP 5 BEST TRADES")
        print(f"  {'='*50}")
        sorted_by_pnl = sorted(trades, key=lambda t: t.pnl_usdt, reverse=True)
        for t in sorted_by_pnl[:5]:
            print(f"  {t.pair:<10s} {t.side:<5s} {t.leverage}x | "
                  f"${t.pnl_usdt:>+8.2f} ({t.pnl_pct:>+6.2f}%) | "
                  f"{t.hold_hours:>5.1f}h | {t.exit_reason}")

        print(f"\n  {'='*50}")
        print(f"  TOP 5 WORST TRADES")
        print(f"  {'='*50}")
        for t in sorted_by_pnl[-5:]:
            print(f"  {t.pair:<10s} {t.side:<5s} {t.leverage}x | "
                  f"${t.pnl_usdt:>+8.2f} ({t.pnl_pct:>+6.2f}%) | "
                  f"{t.hold_hours:>5.1f}h | {t.exit_reason}")

        self._print_decision_summary()

        print("\n" + "="*70)
        print("  BACKTEST COMPLETE")
        print("="*70 + "\n")

    def _print_decision_summary(self):
        """Print summary of all ensemble decisions."""
        print(f"\n  {'='*50}")
        print(f"  ENSEMBLE DECISION SUMMARY")
        print(f"  {'='*50}")

        total = len(self.decisions_log)
        if total == 0:
            print("  No decisions recorded.")
            return

        buys = len([d for d in self.decisions_log if d["action"] == 1])
        sells = len([d for d in self.decisions_log if d["action"] == -1])
        holds = len([d for d in self.decisions_log if d["action"] == 0])

        print(f"  Total Decisions:    {total}")
        print(f"  BUY signals:        {buys} ({buys/total*100:.1f}%)")
        print(f"  SELL signals:       {sells} ({sells/total*100:.1f}%)")
        print(f"  HOLD (no trade):    {holds} ({holds/total*100:.1f}%)")

        # Confidence distribution for signals
        signals = [d for d in self.decisions_log if d["action"] != 0]
        if signals:
            confs = [d["confidence"] for d in signals]
            print(f"\n  Signal Confidence Stats:")
            print(f"    Mean:   {np.mean(confs):.3f}")
            print(f"    Median: {np.median(confs):.3f}")
            print(f"    Min:    {np.min(confs):.3f}")
            print(f"    Max:    {np.max(confs):.3f}")

        # Reasoning breakdown
        from collections import Counter
        if signals:
            reasons = Counter()
            for d in signals:
                # Simplify reasoning
                r = d["reasoning"]
                if "Both BUY" in r:
                    reasons["Both agree BUY"] += 1
                elif "Both SELL" in r:
                    reasons["Both agree SELL"] += 1
                elif "XGB override" in r:
                    reasons["XGB single override"] += 1
                elif "RL override" in r:
                    reasons["RL single override"] += 1
                else:
                    reasons[r] += 1

            print(f"\n  Signal Sources:")
            for reason, count in reasons.most_common():
                print(f"    {reason:<25s}: {count}")


# ============================================================
# MAIN
# ============================================================
def main():
    start_time = time.time()

    # Load models
    xgb_model, rl_agent = load_models()

    # Feature engine
    from src.data.features import FeatureEngine
    fe = FeatureEngine()

    # Fetch data for each pair
    all_data = {}
    for pair in PAIRS:
        logger.info(f"\nFetching data for {pair}...")

        # Fetch 1H candles (need extra for indicator warm-up: ~250 extra)
        df_1h = fetch_historical(pair, "1h", BACKTEST_DAYS + 15)
        if df_1h.empty or len(df_1h) < 100:
            logger.warning(f"  {pair}: insufficient 1H data ({len(df_1h)} candles), skipping")
            continue
        logger.info(f"  {pair} 1H: {len(df_1h)} candles ({df_1h['timestamp'].iloc[0]} to {df_1h['timestamp'].iloc[-1]})")

        # Fetch 4H candles
        df_4h = fetch_historical(pair, "4h", BACKTEST_DAYS + 30)
        logger.info(f"  {pair} 4H: {len(df_4h)} candles")

        # Fetch 1D candles
        df_1d = fetch_historical(pair, "1d", BACKTEST_DAYS + 60)
        logger.info(f"  {pair} 1D: {len(df_1d)} candles")

        # Calculate multi-timeframe features
        try:
            features_df = fe.calculate_multi_tf_features(df_1h, df_4h, df_1d)
        except Exception as e:
            logger.warning(f"  {pair}: feature calculation failed: {e}")
            continue

        if features_df.empty or len(features_df) < 10:
            logger.warning(f"  {pair}: insufficient features ({len(features_df)} rows), skipping")
            continue

        # Align features to model's expected columns
        # Add any missing features as 0
        for feat in xgb_model.feature_names:
            if feat not in features_df.columns:
                features_df[feat] = 0.0

        logger.info(f"  {pair}: {len(features_df)} feature rows ready, {len(fe.feature_names)} features")
        all_data[pair] = {"features_df": features_df, "1h": df_1h}

    if not all_data:
        logger.error("No pairs have sufficient data. Exiting.")
        return

    # Run backtest
    engine = BacktestEngine(xgb_model, rl_agent, STARTING_BALANCE)
    engine.run(all_data)
    engine.report()

    elapsed = time.time() - start_time
    logger.info(f"Backtest completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
