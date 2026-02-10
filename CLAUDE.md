# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cryptobot is an autonomous AI-powered cryptocurrency trading bot for OKX exchange. It combines XGBoost price prediction with PPO Reinforcement Learning in an ensemble voting system. The bot runs 24/7, self-heals from failures, and auto-retrains models weekly.

## Commands

```bash
# Initial setup
pip install -r requirements.txt
cp .env.example .env  # then configure API keys

# Train models (required before first run)
python train_models.py

# Run the trading bot
python -m src.main

# Run as systemd service
systemctl start cryptobot
systemctl status cryptobot
journalctl -u cryptobot -f  # live logs
```

## Architecture

### Data Flow
```
OKX API → DataCollector → FeatureEngine (50+ indicators)
    → Ensemble (XGBoost + RL vote) → RiskManager → TradeExecutor → OKX
```

### Key Modules

**Entry Point:** `src/main.py` - Infinite loop running every 30 min (configurable). Orchestrates data collection, analysis, risk checks, and execution.

**ML Models (`src/models/`):**
- `xgboost_model.py` - Predicts BUY/HOLD/SELL with confidence score
- `rl_agent.py` - PPO agent trained on custom TradingEnvironment
- `ensemble.py` - Voting system: both models must agree (or one >85% confident)

**Trading (`src/trading/`):**
- `okx_client.py` - OKX API wrapper (market data, orders, balance)
- `executor.py` - Validates with RiskManager, executes, logs to SQLite
- `positions.py` - Tracks open positions, P&L, syncs from exchange

**Data (`src/data/`):**
- `collector.py` - Fetches OHLCV candles from OKX
- `features.py` - Calculates 50+ technical indicators (RSI, MACD, Bollinger, ATR, etc.)

**Risk (`src/risk/manager.py`):**
- Position sizing based on confidence
- Stop-loss (default 8%) and take-profit (default 20%)
- Daily loss circuit breaker (default 12%)
- Max open positions limit

**Autonomous (`src/autonomous/`):**
- `watchdog.py` - Auto-restart on failures
- `retrainer.py` - Weekly model retraining (only deploys if accuracy improves)
- `health.py` - System health monitoring

**Other:**
- `config.py` - Loads from env vars, validates settings
- `dashboard.py` - Flask web dashboard on port 5000
- `security/vault.py` - Fernet encryption for API keys

### Configuration

All settings via environment variables (see `.env.example`):
- `TRADING_MODE` - "live" or "demo"
- `TRADING_PAIRS` - Comma-separated (e.g., "BTC-USDT,ETH-USDT,SOL-USDT")
- `MIN_CONFIDENCE` - Threshold to execute trades (0.0-1.0)
- `MAX_POSITION_PERCENT` - Max % of balance per trade
- `STOP_LOSS_PERCENT`, `TAKE_PROFIT_PERCENT`, `DAILY_LOSS_LIMIT`
- `ANALYSIS_INTERVAL` - Minutes between trading cycles

### Database

SQLite at `data/trades.db`:
- `trades` - Executed trades with entry/exit, P&L
- `decisions` - All ensemble decisions (audit trail)

### Model Files

Saved to `models/`:
- `xgboost_model.json` - Trained XGBoost classifier
- `rl_agent.zip` - Trained PPO agent (stable-baselines3 format)

## Trading Loop (each cycle)

1. Health check
2. Check daily loss limit (pause if exceeded)
3. Update open position prices, check stop-loss/take-profit
4. For each pair: collect data → features → ensemble decision → risk validate → execute
5. Check if models need retraining
6. Sleep until next cycle

## Key Design Decisions

- **Ensemble voting**: Conservative - both XGBoost and RL must agree unless one is >85% confident
- **No withdrawal permissions**: OKX API keys should only have read+trade (security)
- **Position recovery**: Auto-syncs from exchange on startup (crash resilience)
- **Chronological splits**: Training data split by time (no shuffle) to prevent look-ahead bias
- **Fee-aware training**: RL environment includes 0.1% transaction fees
