"""
Daily Briefing v2.0
====================
Sends a daily performance report via Telegram.
Tracks trades, P&L, win rate, and account balance.
Runs via cron at 9:00 AM MUT (5:00 UTC) daily.
"""

import os
import sys
import sqlite3
import subprocess
import requests
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load env
with open("/root/Cryptobot/.env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

import okx.Account as Account

TELEGRAM_BOT_TOKEN = "8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY"
TELEGRAM_CHAT_ID = "7997570468"
DB_PATH = "/root/Cryptobot/data/trades.db"
# Day 1 = Feb 15, 2026 (plan start date)
PLAN_START = datetime(2026, 2, 15)


def send_telegram(message: str):
    try:
        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
            data={'chat_id': TELEGRAM_CHAT_ID, 'text': message},
            timeout=10
        )
    except Exception as e:
        print(f"Telegram error: {e}")


def get_account_data():
    """Get account balance and positions."""
    acct = Account.AccountAPI(
        os.environ["OKX_API_KEY"],
        os.environ["OKX_SECRET_KEY"],
        os.environ["OKX_PASSPHRASE"],
        False, "0"
    )

    # Balance
    bal = acct.get_account_balance(ccy="USDT")
    total_eq = 0.0
    avail = 0.0
    if bal and bal.get("data"):
        for b in bal["data"]:
            total_eq = float(b.get("totalEq", 0))
            for d in b.get("details", []):
                avail = float(d.get("availBal", 0))

    # Positions
    positions = []
    total_pnl = 0.0
    pos = acct.get_positions(instType="SWAP")
    if pos and pos.get("data"):
        for p in pos["data"]:
            if float(p.get("pos", 0)) != 0:
                pnl = float(p.get("upl", 0))
                total_pnl += pnl
                positions.append({
                    "pair": p["instId"].replace("-SWAP", ""),
                    "side": "LONG" if float(p.get("pos", 0)) > 0 else "SHORT",
                    "lever": p.get("lever", "?"),
                    "notional": float(p.get("notionalUsd", 0)),
                    "pnl": pnl
                })

    return total_eq, avail, positions, total_pnl


def get_trade_stats(hours: int = 24) -> dict:
    """Get trade stats for the last N hours."""
    stats = {'total': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0.0,
             'best': None, 'worst': None}

    if not Path(DB_PATH).exists():
        return stats

    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        cursor = conn.execute(
            "SELECT pair, side, pnl, timestamp FROM trades "
            "WHERE timestamp > ? AND pnl IS NOT NULL AND pnl != 0 "
            "ORDER BY timestamp DESC", (cutoff,)
        )

        for pair, side, pnl, ts in cursor.fetchall():
            stats['total'] += 1
            stats['total_pnl'] += pnl
            if pnl > 0:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            if stats['best'] is None or pnl > stats['best'][1]:
                stats['best'] = (pair, pnl)
            if stats['worst'] is None or pnl < stats['worst'][1]:
                stats['worst'] = (pair, pnl)

        conn.close()
    except Exception as e:
        print(f"DB error: {e}")

    return stats


def get_all_time_stats() -> dict:
    """Get all-time stats since plan start."""
    stats = {'total': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0.0}

    if not Path(DB_PATH).exists():
        return stats

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute('SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL AND pnl != 0')
        stats['total'] = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM trades WHERE pnl > 0')
        stats['wins'] = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM trades WHERE pnl < 0')
        stats['losses'] = c.fetchone()[0]
        c.execute('SELECT SUM(pnl) FROM trades WHERE pnl IS NOT NULL AND pnl != 0')
        result = c.fetchone()[0]
        stats['total_pnl'] = result if result else 0.0

        conn.close()
    except Exception as e:
        print(f"DB error: {e}")

    return stats


def get_fear_greed() -> str:
    try:
        resp = requests.get('https://api.alternative.me/fng/', timeout=10)
        data = resp.json()['data'][0]
        return f"{data['value']} ({data['value_classification']})"
    except:
        return "N/A"


def get_service_status(name: str) -> str:
    r = subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True)
    return r.stdout.strip()


def build_briefing():
    """Build the daily briefing message."""
    now = datetime.now()
    day_num = (now - PLAN_START).days + 1
    total_eq, avail, positions, unrealized_pnl = get_account_data()
    daily = get_trade_stats(24)
    all_time = get_all_time_stats()
    fng = get_fear_greed()

    daily_wr = (daily['wins'] / daily['total'] * 100) if daily['total'] > 0 else 0
    all_time_wr = (all_time['wins'] / all_time['total'] * 100) if all_time['total'] > 0 else 0

    bot_status = get_service_status("cryptobot")
    sniper_status = get_service_status("listing-sniper")

    msg = f"📊 DAILY BRIEFING — Day {day_num} ({now.strftime('%b %d, %Y')})\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    # Account
    msg += f"💰 Balance: ${total_eq:.2f} (avail: ${avail:.2f})\n"
    msg += f"📈 Unrealized P&L: ${unrealized_pnl:+.4f}\n"
    msg += f"🌡 Fear & Greed: {fng}\n\n"

    # Services
    bot_icon = "🟢" if bot_status == "active" else "🔴"
    sniper_icon = "🟢" if sniper_status == "active" else "🔴"
    msg += f"{bot_icon} Bot: {bot_status} | {sniper_icon} Sniper: {sniper_status}\n\n"

    # Last 24h
    msg += f"📋 Last 24h:\n"
    msg += f"  Trades: {daily['total']} | W: {daily['wins']} L: {daily['losses']} | WR: {daily_wr:.0f}%\n"
    msg += f"  P&L: ${daily['total_pnl']:+.4f}\n"
    if daily['best']:
        msg += f"  Best: {daily['best'][0]} ${daily['best'][1]:+.4f}\n"
    if daily['worst']:
        msg += f"  Worst: {daily['worst'][0]} ${daily['worst'][1]:+.4f}\n"
    msg += "\n"

    # All-time since plan start
    msg += f"📊 All-time ({all_time['total']} trades):\n"
    msg += f"  Win rate: {all_time_wr:.0f}% | P&L: ${all_time['total_pnl']:+.4f}\n\n"

    # Open positions
    if positions:
        msg += f"📍 Open Positions ({len(positions)}):\n"
        for p in positions:
            emoji = "🟢" if p['pnl'] >= 0 else "🔴"
            msg += f"  {emoji} {p['pair']} {p['side']} {p['lever']}x | ${p['pnl']:+.4f}\n"
        msg += "\n"
    else:
        msg += "📍 No open positions\n\n"

    # Phase tracking
    if day_num <= 7:
        msg += f"🎯 Phase 1: PROVE IT (day {day_num}/7)\n"
        msg += f"  Target: >45% win rate\n"
        msg += f"  Current: {all_time_wr:.0f}% {'✅' if all_time_wr >= 45 else '⏳'}\n"
    elif day_num <= 21:
        msg += f"🎯 Phase 2: ADD INTELLIGENCE (day {day_num})\n"
    else:
        msg += f"🎯 Phase 3: SCALE UP (day {day_num})\n"

    msg += "\n— Miya 🤖"
    return msg


if __name__ == "__main__":
    msg = build_briefing()
    send_telegram(msg)
    print("Briefing sent!")
    print(msg)
