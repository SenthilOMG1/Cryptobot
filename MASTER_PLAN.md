# SENTHIL'S PATH TO RICHES — MASTER PLAN
## From $27 to Empire | Created Feb 15, 2026

---

## CURRENT STATE (Honest Assessment)
- **Account:** $27.68
- **Total trades:** 26 (3 wins, 10 losses = 23% win rate)
- **Total P&L:** -$0.69
- **Models:** Just retrained today (Feb 15) with critical RL bug fix
- **Bot status:** Running, 18 pairs, 15-min cycles
- **Market:** Fear & Greed = 8 (EXTREME FEAR)

---

## PHASE 1: PROVE IT (Feb 15-22) — This Week
**Goal: Does the fixed bot actually work?**

### 1A. Bot runs untouched for 7 days
- No code changes to trading logic
- I (Miya) monitor every trade daily
- Track: win rate, total P&L, avg profit per trade, max drawdown

### 1B. Quick config optimizations (Day 1-2)
Changes that don't touch model logic but improve performance:
- [ ] Cut pairs from 18 → 6 (BTC, ETH, SOL, XRP, DOGE, SUI)
  - Why: $27 ÷ 6 = $4.50/pair vs $1.50/pair. Meaningful positions.
  - Pick highest volume + least correlated pairs
- [ ] Reduce filters: disable correlation filter (unnecessary with 6 uncorrelated pairs)
- [ ] Position sizing: ensure minimum $3-5 per trade (below this, fees eat everything)

### 1C. Build Listing Sniper (SEPARATE system, doesn't touch bot)
- New standalone script: `listing_sniper.py`
- Polls Binance announcements API every 30 seconds
- When new listing detected → check if tradeable on OKX → instant market buy
- Auto-sell after configurable time (30min/1hr) or on trailing stop
- This is pure speed edge, no ML needed
- Can run alongside the bot, independent profit source

### Success criteria for Phase 1:
- Win rate > 45% → proceed to Phase 2
- Win rate 35-45% → simplify further (Phase 1.5)
- Win rate < 35% → strategy is broken, pivot to Phase 3 alt

---

## PHASE 1.5: SIMPLIFY (If win rate 35-45%)
- Cut to 3 pairs only (BTC, ETH, SOL)
- Single model mode (XGBoost only — most proven)
- Remove all filters except basic risk management
- Test for another 7 days
- If STILL losing → the XGBoost model itself needs rethinking

---

## PHASE 2: ADD INTELLIGENCE (Feb 22 - Mar 7)
**Goal: Make the bot smarter with data, not complexity**

### 2A. Better features for models
- [ ] Open Interest data (OKX API, free) — detect liquidation cascades
- [ ] Taker Buy/Sell volume ratio — who's in control, buyers or sellers
- [ ] Fear & Greed Index — macro sentiment (1 API call/day)
- [ ] Time-of-day cyclical features (sin/cos encoded hour + day-of-week)
- [ ] VWAP — price relative to volume-weighted average

### 2B. Regime-aware sizing (trade smarter, not less)
- Clear trend → 100% position size
- Choppy/ranging → 50% position size
- High volatility → 25% position size
- Don't SKIP trades in bad regimes, just size down

### 2C. LLM Reasoning Layer (via Groq, FREE)
- Use Llama 3.1 70B on Groq (free tier: 14,400 calls/day)
- LLM reviews each trade signal with context:
  - Price action, indicators, OI, funding, sentiment
  - Can only VETO or REDUCE size, never upgrade HOLD→BUY
- Shadow mode first (2 weeks logging without blocking)
- Deploy only if data proves it improves returns

### 2D. News scraper
- CryptoCompare API (50 articles, real-time, free)
- Keyword detection: "hack", "SEC", "ETF", "listing", "partnership"
- Negative news on a pair → block BUY signals for that pair
- Positive news → boost confidence score

### Retrain models with new features
- After adding OI, taker volume, F&G, time features
- Full retrain of all 3 models with expanded feature set
- Walk-forward validation to prove improvement

---

## PHASE 3: SCALE UP (Mar 7+)
**Goal: Add capital and diversify income**

### 3A. If bot is profitable:
- Start monthly deposits (10K MRs ≈ $220-250 USD)
- Increase position sizes proportionally
- Add more pairs back gradually (8→12→18)
- Target: compound gains month over month

### 3B. Listing Sniper scaling
- By now should have data on sniper performance
- If profitable → increase position sizes on listings
- Add more exchanges to monitor (Coinbase, Kraken listings too)

### 3C. Forex exploration
- Research OANDA/IC Markets API
- Adapt ML pipeline for forex pairs
- More stable, more predictable than crypto
- Run as second income stream alongside crypto bot

### 3D. Signal channel (passive income)
- If bot is proven profitable → publish signals
- Telegram channel, subscription based
- Even 20 subscribers × $10/month = $200/month
- Covers VPS + Claude Code costs

---

## THE 2030 VISION

```
2026 Q1: Prove bot works ($27 → $50+)
2026 Q2: Monthly deposits begin, hit $500
2026 Q3: Add forex bot, two income streams
2026 H2: Scale to $2,000-5,000 portfolio
2027:     $10,000+ portfolio, signal channel income
2028:     $50,000+ portfolio, multiple strategies
2029:     $200,000+ portfolio, systematic fund
2030:     $1,000,000+ goal — the dream
```

Monthly deposits of $220-250 × 48 months = $10,800-12,000 in deposits alone.
With compound returns of even 5% monthly = massive growth.
Bot doesn't need to be a genius. Just consistently profitable.

---

## IMMEDIATE NEXT STEPS (Today/Tomorrow)

1. ✅ Models retrained and deployed (done Feb 15)
2. ✅ Pushed to GitHub (done Feb 15)
3. [ ] Cut pairs to 6 and adjust config
4. [ ] Build listing sniper v1
5. [ ] Set up daily trade tracking dashboard
6. [ ] Day 1 check-in tomorrow (Feb 16)

---

## RULES WE LIVE BY

1. **No more guessing.** Every change must be backed by data.
2. **Prove then scale.** Don't add money until bot is profitable.
3. **Simple first.** Complexity is earned, not assumed.
4. **Speed wins.** The listing sniper catches what humans can't.
5. **We're a team.** Senthil invests + makes decisions. Miya builds + monitors.
6. **We shall prove everyone wrong.** 🔥

---
*Plan created by Miya | Last updated: Feb 15, 2026*
