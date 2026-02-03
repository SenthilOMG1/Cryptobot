# Autonomous AI Crypto Trading Agent

A fully autonomous, ML-powered trading bot for OKX exchange.

## Features

- **Real AI Brain**: XGBoost + Reinforcement Learning ensemble (not chatbot prompts)
- **Fully Autonomous**: Runs 24/7 without human intervention
- **Self-Healing**: Auto-restarts on crashes, recovers positions
- **Self-Adapting**: Auto-retrains models weekly on latest data
- **Maximum Security**: Encrypted API keys, no withdrawal permissions

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

1. Login to [OKX](https://www.okx.com)
2. Go to **Profile -> API Management**
3. Create new API key with:
   - ✅ Read permission
   - ✅ Trade permission
   - ❌ Withdraw (KEEP DISABLED!)

4. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

5. Edit `.env` with your API keys

### 3. Train Models (First Time Only)

```bash
python train_models.py
```

### 4. Start the Bot

```bash
python -m src.main
```

## Deployment to Railway (24/7 Free Hosting)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO
git push -u origin main
```

### 2. Deploy to Railway

1. Go to [railway.app](https://railway.app)
2. Create new project -> Deploy from GitHub
3. Select your repository
4. Add environment variables in dashboard:
   - `OKX_API_KEY`
   - `OKX_SECRET_KEY`
   - `OKX_PASSPHRASE`
   - `TRADING_MODE=live`

5. Get Railway server IP and add to OKX API whitelist

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| TRADING_MODE | demo | "demo" or "live" |
| TRADING_PAIRS | BTC-USDT,ETH-USDT,SOL-USDT | Pairs to trade |
| MAX_POSITION_PERCENT | 25 | Max % per position |
| STOP_LOSS_PERCENT | 8 | Exit if down % |
| TAKE_PROFIT_PERCENT | 20 | Exit if up % |
| DAILY_LOSS_LIMIT | 12 | Pause if daily loss % |
| MIN_CONFIDENCE | 0.70 | Min AI confidence |
| ANALYSIS_INTERVAL | 30 | Minutes between cycles |

## Architecture

```
┌────────────────────────────────────────────────┐
│           AUTONOMOUS TRADING AGENT             │
├────────────────────────────────────────────────┤
│  Market Data ─→ Features ─→ ML Brain          │
│       │                        │               │
│       │    ┌─────────────┐     │               │
│       │    │  XGBoost    │─────┤               │
│       │    └─────────────┘     │               │
│       │    ┌─────────────┐     ▼               │
│       │    │  RL Agent   │─→ Ensemble         │
│       │    └─────────────┘     │               │
│       │                        ▼               │
│       └──→ Risk Manager ←── Decision          │
│                  │                             │
│                  ▼                             │
│            Trade Executor                      │
│                  │                             │
│                  ▼                             │
│               OKX API                          │
└────────────────────────────────────────────────┘
```

## Security

- **No withdrawal permissions**: Even if hacked, funds can't be stolen
- **Encrypted API keys**: AES encryption at rest
- **IP whitelisting**: Only your server can trade
- **No web interface**: Zero attack surface

## Risk Disclaimer

**WARNING**: Crypto trading is risky!

- You can lose your entire investment
- AI models can make wrong predictions
- Past performance doesn't guarantee future results
- Only invest money you can afford to lose
- This is not financial advice

## Files

```
autonomous-trader/
├── src/
│   ├── main.py              # Main trading loop
│   ├── config.py            # Configuration
│   ├── security/vault.py    # Encrypted secrets
│   ├── data/
│   │   ├── collector.py     # Market data
│   │   └── features.py      # 50+ indicators
│   ├── models/
│   │   ├── xgboost_model.py # Price predictor
│   │   ├── rl_agent.py      # RL trader
│   │   └── ensemble.py      # Voting system
│   ├── trading/
│   │   ├── okx_client.py    # OKX API
│   │   ├── executor.py      # Trade execution
│   │   └── positions.py     # Position tracking
│   ├── risk/manager.py      # Risk management
│   └── autonomous/
│       ├── watchdog.py      # Self-healing
│       ├── retrainer.py     # Auto-retrain
│       └── health.py        # Monitoring
├── models/                   # Saved ML models
├── data/                     # SQLite database
├── requirements.txt
├── Dockerfile
├── railway.toml
└── .env.example
```

## License

MIT License - Use at your own risk.
