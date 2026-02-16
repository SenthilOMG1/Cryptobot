"""
Cryptobot Monitor Daemon
========================
Always-on background process that watches everything and
proactively sends alerts + triggers Claude Code analysis
when something important happens.
"""

import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime, timedelta
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load env
with open("/root/Cryptobot/.env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

import okx.Account as Account
import okx.PublicData as Public
from src.notifications import send_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Config
CHECK_INTERVAL = 180  # 3 minutes between checks
PRICE_ALERT_PCT = 4.0  # Alert on >4% moves in an hour
PNL_WARN_PCT = -2.0  # Warn when position drops below this
PNL_GOOD_PCT = 5.0  # Alert when position is doing great
STATE_FILE = "/root/Cryptobot/data/monitor_state.json"

# OKX API
acct = Account.AccountAPI(
    os.environ["OKX_API_KEY"],
    os.environ["OKX_SECRET_KEY"],
    os.environ["OKX_PASSPHRASE"],
    False, "0"
)
public = Public.PublicAPI(flag="0")


def load_state():
    """Load persistent state."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "last_prices": {},
        "alerted_positions": {},
        "last_claude_analysis": None,
        "daily_pnl_start": None,
        "daily_equity_start": None,
        "last_briefing_hour": None,
    }


def save_state(state):
    """Save persistent state."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def get_price(pair: str) -> float:
    """Get current price for a pair."""
    try:
        import requests
        resp = requests.get(
            f"https://www.okx.com/api/v5/market/ticker?instId={pair}",
            timeout=10
        )
        data = resp.json()
        if data.get("data"):
            return float(data["data"][0]["last"])
    except:
        pass
    return 0.0


def get_positions():
    """Get all open positions."""
    try:
        pos = acct.get_positions(instType="SWAP")
        positions = []
        if pos and pos.get("data"):
            for p in pos["data"]:
                if float(p.get("pos", 0)) != 0:
                    positions.append({
                        "pair": p["instId"].replace("-SWAP", ""),
                        "side": p.get("posSide", "long"),
                        "lever": p.get("lever", "1"),
                        "notional": float(p.get("notionalUsd", 0)),
                        "pnl": float(p.get("upl", 0)),
                        "pnl_pct": float(p.get("uplRatio", 0)) * 100,
                        "entry_price": float(p.get("avgPx", 0)),
                    })
        return positions
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return []


def get_equity():
    """Get total account equity."""
    try:
        bal = acct.get_account_balance(ccy="USDT")
        if bal and bal.get("data"):
            for b in bal["data"]:
                return float(b.get("totalEq", 0))
    except:
        pass
    return 0.0


def check_bot_health():
    """Check if the trading bot is running."""
    r = subprocess.run(["systemctl", "is-active", "cryptobot"], capture_output=True, text=True)
    return r.stdout.strip() == "active"


def run_claude_analysis(prompt: str) -> str:
    """Run Claude Code as miya user for deeper analysis."""
    try:
        escaped_prompt = prompt.replace("'", "'\"'\"'")
        shell_cmd = f"cd /root/Cryptobot && claude --dangerously-skip-permissions --continue --print '{escaped_prompt}'"
        result = subprocess.run(
            ["su", "-", "miya", "-c", shell_cmd],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"}
        )
        return result.stdout.strip()
    except:
        return None


def monitor_loop():
    """Main monitoring loop."""
    state = load_state()
    logger.info("Monitor daemon started")
    send_message("👁 Monitor daemon started. I'm watching everything.")

    # Track prices for movement detection
    price_history = {}  # pair -> deque of (timestamp, price)

    while True:
        try:
            now = datetime.now()
            hour = now.hour

            # ============================================================
            # 1. BOT HEALTH CHECK
            # ============================================================
            if not check_bot_health():
                if state.get("bot_was_alive", True):
                    send_message(
                        "🚨 ALERT: Trading bot is DOWN!\n\n"
                        "The cryptobot service stopped running.\n"
                        "Use /restart to bring it back."
                    )
                    state["bot_was_alive"] = False
            else:
                if not state.get("bot_was_alive", True):
                    send_message("✅ Trading bot is back online!")
                state["bot_was_alive"] = True

            # ============================================================
            # 2. PRICE MOVEMENT DETECTION
            # ============================================================
            major_pairs = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
            for pair in major_pairs:
                price = get_price(pair)
                if price <= 0:
                    continue

                if pair not in price_history:
                    price_history[pair] = deque(maxlen=20)

                price_history[pair].append((time.time(), price))

                # Check against price from ~1 hour ago
                if len(price_history[pair]) >= 15:
                    old_ts, old_price = price_history[pair][0]
                    if old_price > 0:
                        change_pct = ((price - old_price) / old_price) * 100
                        alert_key = f"price_{pair}_{now.hour}"

                        if abs(change_pct) >= PRICE_ALERT_PCT and alert_key not in state.get("alerted_positions", {}):
                            direction = "📈 PUMPING" if change_pct > 0 else "📉 DUMPING"
                            send_message(
                                f"{direction}: {pair}\n\n"
                                f"Move: {change_pct:+.1f}% in ~{(time.time()-old_ts)/60:.0f}min\n"
                                f"Price: ${price:,.2f}"
                            )
                            state.setdefault("alerted_positions", {})[alert_key] = now.isoformat()

            # ============================================================
            # 3. POSITION MONITORING
            # ============================================================
            positions = get_positions()
            for pos in positions:
                pair = pos["pair"]
                pnl_pct = pos["pnl_pct"]
                alert_key_bad = f"pnl_warn_{pair}"
                alert_key_good = f"pnl_good_{pair}"

                # Warn on big losses
                if pnl_pct <= PNL_WARN_PCT:
                    last_alert = state.get("alerted_positions", {}).get(alert_key_bad, "")
                    if not last_alert or (now - datetime.fromisoformat(last_alert)).seconds > 3600:
                        send_message(
                            f"⚠️ {pair} is down {pnl_pct:.1f}%\n\n"
                            f"Direction: {pos['side']} {pos['lever']}x\n"
                            f"P&L: ${pos['pnl']:.4f}\n"
                            f"Size: ${pos['notional']:.2f}"
                        )
                        state.setdefault("alerted_positions", {})[alert_key_bad] = now.isoformat()

                # Celebrate big wins
                if pnl_pct >= PNL_GOOD_PCT:
                    last_alert = state.get("alerted_positions", {}).get(alert_key_good, "")
                    if not last_alert or (now - datetime.fromisoformat(last_alert)).seconds > 3600:
                        send_message(
                            f"🔥 {pair} is up {pnl_pct:+.1f}%!\n\n"
                            f"Direction: {pos['side']} {pos['lever']}x\n"
                            f"P&L: ${pos['pnl']:.4f}\n"
                            f"Size: ${pos['notional']:.2f}\n\n"
                            f"Let it ride or close?"
                        )
                        state.setdefault("alerted_positions", {})[alert_key_good] = now.isoformat()

            # ============================================================
            # 4. SCHEDULED CLAUDE CHECK-INS
            # ============================================================
            last_hour = state.get("last_briefing_hour", -1)

            # 9 AM - Morning briefing (handled by cron, but backup)
            # 3 PM - Afternoon analysis
            if hour == 15 and last_hour != 15:
                equity = get_equity()
                n_pos = len(positions)
                total_pnl = sum(p["pnl"] for p in positions)

                analysis = run_claude_analysis(
                    f"Afternoon check-in. Account equity: ${equity:.2f}, "
                    f"{n_pos} open positions, total unrealized P&L: ${total_pnl:.4f}. "
                    f"Check the bot logs and positions. Send Senthil a brief Telegram update "
                    f"about how things are going today. Be conversational, not robotic. "
                    f"Use send_message() from src.notifications to send it."
                )
                if analysis:
                    send_message(f"📊 Afternoon Update\n\n{analysis[:3000]}")
                state["last_briefing_hour"] = 15

            # 9 PM - Evening review
            if hour == 21 and last_hour != 21:
                equity = get_equity()
                analysis = run_claude_analysis(
                    f"Evening review. Account equity: ${equity:.2f}. "
                    f"Review today's trading activity from the logs. "
                    f"Any trades executed? Any stop-losses? How did positions move? "
                    f"Give Senthil a short evening summary. Casual tone."
                )
                if analysis:
                    send_message(f"🌙 Evening Review\n\n{analysis[:3000]}")
                state["last_briefing_hour"] = 21

            # ============================================================
            # 5. DAILY EQUITY TRACKING
            # ============================================================
            if hour == 0 and state.get("last_equity_reset") != now.strftime("%Y-%m-%d"):
                equity = get_equity()
                state["daily_equity_start"] = equity
                state["last_equity_reset"] = now.strftime("%Y-%m-%d")
                # Clean old alerts
                state["alerted_positions"] = {}

            # ============================================================
            # 6. MEMORY CLEANUP
            # ============================================================
            # Remove old alerts (>24h)
            alerts = state.get("alerted_positions", {})
            cutoff = (now - timedelta(hours=24)).isoformat()
            state["alerted_positions"] = {
                k: v for k, v in alerts.items()
                if v > cutoff
            }

            save_state(state)

        except Exception as e:
            logger.error(f"Monitor error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    monitor_loop()
