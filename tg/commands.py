"""
Telegram Command Handlers — All /slash commands
================================================
"""

import os
import glob
import asyncio
import logging
import subprocess

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from tg.config import (
    auth, owner_only,
    AUTHORIZED_USERS, OWNER_ID, WORKING_DIR, SESSIONS_DIR, SYSTEM_PROMPT,
    _chat_queues, _active_proc, _get_chat_lock,
)
import tg.config as _cfg
from tg.claude_bridge import run_claude, _extract_stream_result
from tg.mem0_bridge import mem0_client, mem0_recall, mem0_store
from tg.helpers import get_sessions, send_response

logger = logging.getLogger(__name__)


@auth
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Cryptobot Assistant\n"
        "━━━━━━━━━━━━━━━━━━━\n\n"
        "Send any message or voice note → Claude Code\n\n"
        "Trading:\n"
        "/positions - Open positions & P&L\n"
        "/pnl - P&L summary\n"
        "/balance - Account equity\n"
        "/market - Live prices (BTC, ETH, SOL...)\n"
        "/trades - Recent trade history\n"
        "/leverage - Leverage on all positions\n"
        "/close [pair] - Close a position\n\n"
        "Bot Control:\n"
        "/status - System & bot status\n"
        "/phone - Phone connection & remote control\n"
        "/services - All running services\n"
        "/config - Bot settings\n"
        "/logs - Recent logs\n"
        "/restart - Restart trading bot\n"
        "/pause - Pause trading (monitor stays)\n"
        "/unpause - Resume trading\n"
        "/stop - Stop trading bot\n"
        "/train - Start model training\n"
        "/steer - Redirect Miya mid-task\n\n"
        "Sessions:\n"
        "/sessions - List Claude sessions\n"
        "/resume - Switch session\n"
        "/new - Fresh session\n"
        "/current - Current session\n\n"
        "Or just chat - everything goes to Claude."
    )


@auth
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    services = {}
    for svc in ["cryptobot", "telegram-bot", "cryptobot-mcp", "miya-brain"]:
        r = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True)
        services[svc] = r.stdout.strip()

    r = subprocess.run(["uptime", "-p"], capture_output=True, text=True)
    uptime = r.stdout.strip()

    r = subprocess.run(["free", "-m"], capture_output=True, text=True)
    mem_lines = r.stdout.strip().split("\n")
    mem_used = mem_lines[1].split()[2] if len(mem_lines) > 1 else "?"
    mem_total = mem_lines[1].split()[1] if len(mem_lines) > 1 else "?"

    status_text = (
        f"System: {uptime}\n"
        f"Memory: {mem_used}MB / {mem_total}MB\n\n"
        f"Services:\n"
    )
    for svc, state in services.items():
        icon = "●" if state == "active" else "○"
        status_text += f"  {icon} {svc}: {state}\n"

    try:
        import requests as _req
        phone = _req.get("http://127.0.0.1:8766/status", timeout=3).json()
        if phone.get("connected"):
            uptime_s = int(phone.get("uptime", 0))
            mins = uptime_s // 60
            status_text += f"\nPhone: Connected ({mins}m uptime)"
        else:
            status_text += "\nPhone: Not connected"
    except Exception:
        status_text += "\nPhone: Brain server offline"

    await update.message.reply_text(status_text)


@auth
async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching positions...")

    try:
        result = subprocess.run(
            ["bash", "-c", """cd /root/Cryptobot && source venv/bin/activate && python3 << 'PYEOF'
import okx.Account as Account
import os

with open("/root/Cryptobot/.env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

acct = Account.AccountAPI(os.environ["OKX_API_KEY"], os.environ["OKX_SECRET_KEY"], os.environ["OKX_PASSPHRASE"], False, "0")

bal = acct.get_account_balance(ccy="USDT")
total_eq = 0
if bal and bal.get("data"):
    for b in bal["data"]:
        total_eq = float(b.get("totalEq", 0))

pos = acct.get_positions(instType="SWAP")
lines = []
total_pnl = 0
if pos and pos.get("data"):
    for p in pos["data"]:
        if float(p.get("pos", 0)) != 0:
            pnl = float(p.get("upl", 0))
            total_pnl += pnl
            lever = p.get("lever", "?")
            notional = float(p.get("notionalUsd", 0))
            pair = p["instId"].replace("-SWAP", "")
            icon = "+" if pnl >= 0 else ""
            lines.append(f"  {pair} short {lever}x | ${notional:.2f} | {icon}${pnl:.4f}")

print(f"Account: ${total_eq:.2f}")
print(f"Positions: {len(lines)}")
print()
for l in lines:
    print(l)
print(f"\\nUnrealized P&L: ${total_pnl:.4f}")
PYEOF"""],
            capture_output=True, text=True, timeout=30
        )
        await update.message.reply_text(result.stdout.strip() if result.stdout.strip() else "Error fetching positions")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@auth
async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        result = subprocess.run(
            ["bash", "-c", """cd /root/Cryptobot && source venv/bin/activate && python3 << 'PYEOF'
import okx.Account as Account
import os

with open("/root/Cryptobot/.env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

acct = Account.AccountAPI(os.environ["OKX_API_KEY"], os.environ["OKX_SECRET_KEY"], os.environ["OKX_PASSPHRASE"], False, "0")
bal = acct.get_account_balance(ccy="USDT")
if bal and bal.get("data"):
    for b in bal["data"]:
        total = float(b.get("totalEq", 0))
        for d in b.get("details", []):
            eq = float(d.get("eq", 0))
            avail = float(d.get("availBal", 0))
            frozen = float(d.get("frozenBal", 0))
            print(f"Total Equity: ${total:.2f}")
            print(f"Available: ${avail:.2f}")
            print(f"In Positions: ${frozen:.2f}")
PYEOF"""],
            capture_output=True, text=True, timeout=15
        )
        await update.message.reply_text(result.stdout.strip() if result.stdout.strip() else "Error")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@auth
async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    result = subprocess.run(
        ["journalctl", "-u", "cryptobot", "--no-pager", "-n", "20"],
        capture_output=True, text=True
    )
    text = result.stdout.strip()[-3500:]
    await update.message.reply_text(f"```\n{text}\n```", parse_mode="Markdown")


@auth
async def cmd_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sessions = get_sessions(10)

    if not sessions:
        await update.message.reply_text("No sessions found.")
        return

    text = "Recent Sessions:\n━━━━━━━━━━━━━━━━━━━\n\n"
    for i, s in enumerate(sessions):
        marker = " ◄ ACTIVE" if s["id"] == _cfg.current_session_id else ""
        text += f"{i+1}. [{s['time']}] {s['size']}{marker}\n   {s['preview']}\n\n"

    text += "Use /resume to switch sessions."
    await update.message.reply_text(text)


@auth
async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sessions = get_sessions(8)
    if not sessions:
        await update.message.reply_text("No sessions found.")
        return

    keyboard = []
    for i, s in enumerate(sessions):
        label = f"{s['time']} | {s['preview'][:35]}"
        keyboard.append([InlineKeyboardButton(label, callback_data=f"resume:{s['id']}")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Pick a session to resume:", reply_markup=reply_markup)


@auth
async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _cfg.current_session_id = None

    loop = asyncio.get_event_loop()
    fresh_msg = "New session started. You are Miya, Senthil's AI assistant connected via Telegram. Say hi to Senthil."
    fresh_parts = ["claude", "--dangerously-skip-permissions",
                   "--verbose", "--output-format", "stream-json",
                   "--append-system-prompt", SYSTEM_PROMPT, fresh_msg]
    escaped = []
    for p in fresh_parts:
        escaped.append("'" + p.replace("'", "'\"'\"'") + "'")
    shell_cmd = f"cd {WORKING_DIR} && {' '.join(escaped)}"
    response = await loop.run_in_executor(None, lambda: subprocess.run(
        ["su", "-", "miya", "-c", shell_cmd],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"}
    ))

    session_files = sorted(
        glob.glob(f"{SESSIONS_DIR}/*.jsonl"),
        key=os.path.getmtime, reverse=True
    )
    if session_files:
        _cfg.current_session_id = os.path.basename(session_files[0]).replace(".jsonl", "")

    output = _extract_stream_result(response.stdout) if response.stdout else "New session started."
    await update.message.reply_text(f"Fresh session started.\n\n{output}")


@auth
async def cmd_current(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _cfg.current_session_id:
        await update.message.reply_text(f"Session: {_cfg.current_session_id[:12]}...\n\nUse /sessions to see all, /new for fresh.")
    else:
        await update.message.reply_text("Using most recent session (--continue mode).\n\nUse /sessions to see all, /new for fresh.")


@owner_only
async def cmd_steer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Interrupt Miya mid-task and redirect her with new guidance."""
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /steer <your guidance>\n\nExample: /steer skip that, focus on signing the APK first")
        return

    guidance = " ".join(args)
    chat_id = update.effective_chat.id
    uid = update.effective_user.id
    user_info = AUTHORIZED_USERS.get(uid, {})
    user_name = user_info.get("name", "Unknown")
    mem0_uid = user_name.lower()

    proc = _active_proc.get(chat_id)
    if proc and proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)
        _active_proc.pop(chat_id, None)
        await update.message.reply_text(f"🔀 Interrupted. Redirecting with: {guidance[:100]}")
    else:
        await update.message.reply_text(f"🔀 No active task — sending guidance directly.")

    queued = _chat_queues.pop(chat_id, [])
    queued_text = ""
    if queued:
        queued_text = "\nAlso, queued messages: " + " | ".join(msg for _, _, msg in queued)

    lock = _get_chat_lock(chat_id)

    for _ in range(10):
        if not lock.locked():
            break
        await asyncio.sleep(0.5)

    async with lock:
        loop = asyncio.get_event_loop()
        memories = await loop.run_in_executor(None, mem0_recall, guidance, mem0_uid)
        prefixed = (
            f"[STEERING from {user_name} — interrupted your current task to redirect you]{memories}: "
            f"{guidance}{queued_text}"
        )
        response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=chat_id))
        loop.run_in_executor(None, mem0_store, guidance, response, mem0_uid)
        await send_response(update, response)


@owner_only
async def cmd_restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Restarting trading bot...")
    result = subprocess.run(["systemctl", "restart", "cryptobot"], capture_output=True, text=True)
    await asyncio.sleep(3)
    r = subprocess.run(["systemctl", "is-active", "cryptobot"], capture_output=True, text=True)
    await update.message.reply_text(f"Cryptobot: {r.stdout.strip()}")


@owner_only
async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Stopping trading bot...")
    subprocess.run(["systemctl", "stop", "cryptobot"], capture_output=True, text=True)
    await update.message.reply_text("Cryptobot stopped. Use /restart to start again.")


@owner_only
async def cmd_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = subprocess.run(["pgrep", "-f", "train_models_v2"], capture_output=True, text=True)
    if r.stdout.strip():
        await update.message.reply_text("Training already in progress!")
        return

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", "training",
         "cd /root/Cryptobot && source venv/bin/activate && python train_models_v2.py 2>&1 | tee models/training_log.txt"],
        capture_output=True, text=True
    )
    await update.message.reply_text("Training launched in tmux.\nCheck progress: send me 'check training'")


@auth
async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Today's P&L summary."""
    try:
        result = subprocess.run(
            ["bash", "-c", """cd /root/Cryptobot && source venv/bin/activate && python3 << 'PYEOF'
import okx.Account as Account
import os

with open("/root/Cryptobot/.env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

acct = Account.AccountAPI(os.environ["OKX_API_KEY"], os.environ["OKX_SECRET_KEY"], os.environ["OKX_PASSPHRASE"], False, "0")

bal = acct.get_account_balance(ccy="USDT")
total_eq = 0
if bal and bal.get("data"):
    for b in bal["data"]:
        total_eq = float(b.get("totalEq", 0))

pos = acct.get_positions(instType="SWAP")
total_pnl = 0
winners = 0
losers = 0
if pos and pos.get("data"):
    for p in pos["data"]:
        if float(p.get("pos", 0)) != 0:
            pnl = float(p.get("upl", 0))
            total_pnl += pnl
            if pnl >= 0:
                winners += 1
            else:
                losers += 1

print(f"P&L Summary")
print(f"━━━━━━━━━━━━━━")
print(f"Equity: ${total_eq:.2f}")
print(f"Unrealized: ${total_pnl:+.4f}")
print(f"Winners: {winners} | Losers: {losers}")
PYEOF"""],
            capture_output=True, text=True, timeout=15
        )
        await update.message.reply_text(result.stdout.strip() if result.stdout.strip() else "Error")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@auth
async def cmd_market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick market prices."""
    try:
        result = subprocess.run(
            ["bash", "-c", """cd /root/Cryptobot && source venv/bin/activate && python3 << 'PYEOF'
import requests

pairs = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT"]
print("Market Prices")
print("━━━━━━━━━━━━━━")
for pair in pairs:
    try:
        resp = requests.get(f"https://www.okx.com/api/v5/market/ticker?instId={pair}", timeout=5)
        data = resp.json()
        if data.get("data"):
            d = data["data"][0]
            price = float(d["last"])
            change = float(d.get("sodUtc8", price))
            pct = ((price - change) / change * 100) if change else 0
            icon = "▲" if pct >= 0 else "▼"
            print(f"  {pair.replace('-USDT',''): <5} ${price:>10,.2f}  {icon} {pct:+.1f}%")
    except:
        pass
PYEOF"""],
            capture_output=True, text=True, timeout=15
        )
        await update.message.reply_text(result.stdout.strip() if result.stdout.strip() else "Error")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@auth
async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Recent trade history."""
    try:
        result = subprocess.run(
            ["bash", "-c", """cd /root/Cryptobot && source venv/bin/activate && python3 << 'PYEOF'
import sqlite3, os
from datetime import datetime, timedelta

db = "/root/Cryptobot/data/trades.db"
if not os.path.exists(db):
    print("No trades database yet.")
else:
    conn = sqlite3.connect(db)
    try:
        cursor = conn.execute("SELECT pair, action, confidence, price, timestamp FROM trades ORDER BY timestamp DESC LIMIT 15")
        trades = cursor.fetchall()
        if trades:
            print(f"Recent Trades ({len(trades)})")
            print("━━━━━━━━━━━━━━")
            for t in trades:
                print(f"  {t[4][:16]} | {t[0]} {t[1]} @ ${t[3]:.2f} ({t[2]:.0%})")
        else:
            print("No trades recorded yet.")
    except Exception as e:
        print(f"Error: {e}")
    conn.close()
PYEOF"""],
            capture_output=True, text=True, timeout=10
        )
        await update.message.reply_text(result.stdout.strip() if result.stdout.strip() else "No trades yet")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@auth
async def cmd_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current bot settings."""
    try:
        result = subprocess.run(
            ["bash", "-c", """cd /root/Cryptobot && grep -E "^(MIN_CONFIDENCE|STOP_LOSS|TRAILING_STOP|TAKE_PROFIT|MAX_OPEN|MAX_POSITION|FUTURES_ENABLED|FUTURES_LEVERAGE|ANALYSIS_INTERVAL|TRADING_PAIRS)" .env | sort"""],
            capture_output=True, text=True, timeout=5
        )
        text = result.stdout.strip()
        await update.message.reply_text(f"Bot Config\n━━━━━━━━━━━━━━\n{text}")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@auth
async def cmd_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show leverage on all positions."""
    try:
        result = subprocess.run(
            ["bash", "-c", """cd /root/Cryptobot && source venv/bin/activate && python3 << 'PYEOF'
import okx.Account as Account
import os

with open("/root/Cryptobot/.env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

acct = Account.AccountAPI(os.environ["OKX_API_KEY"], os.environ["OKX_SECRET_KEY"], os.environ["OKX_PASSPHRASE"], False, "0")
pos = acct.get_positions(instType="SWAP")
if pos and pos.get("data"):
    print("Leverage Overview")
    print("━━━━━━━━━━━━━━")
    for p in pos["data"]:
        if float(p.get("pos", 0)) != 0:
            pair = p["instId"].replace("-SWAP", "")
            lever = p.get("lever", "?")
            pnl_pct = float(p.get("uplRatio", 0)) * 100
            margin = float(p.get("margin", 0))
            print(f"  {pair: <12} {lever}x | margin: ${margin:.2f} | P&L: {pnl_pct:+.1f}%")
else:
    print("No open positions")
PYEOF"""],
            capture_output=True, text=True, timeout=15
        )
        await update.message.reply_text(result.stdout.strip() if result.stdout.strip() else "No positions")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@owner_only
async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Close a specific position. Usage: /close SOL-USDT"""
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /close SOL-USDT\n\nUse /positions to see open pairs.")
        return

    pair = args[0].upper()
    if not pair.endswith("-USDT"):
        pair = pair + "-USDT"

    await update.message.reply_text(f"Closing {pair}... sending to Claude Code")

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, run_claude,
        f"Close the {pair} position. Use the OKX API to close it. Check if it's a futures/swap position and close accordingly."
    )
    await send_response(update, response)


@owner_only
async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pause trading - stop bot but keep monitor running."""
    await update.message.reply_text("Pausing trading bot (monitor stays active)...")
    subprocess.run(["systemctl", "stop", "cryptobot"], capture_output=True, text=True)
    await update.message.reply_text(
        "Trading paused.\n"
        "Monitor still watching positions.\n"
        "Use /unpause to resume trading."
    )


@owner_only
async def cmd_unpause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Resume trading."""
    await update.message.reply_text("Resuming trading bot...")
    subprocess.run(["systemctl", "start", "cryptobot"], capture_output=True, text=True)
    await asyncio.sleep(3)
    r = subprocess.run(["systemctl", "is-active", "cryptobot"], capture_output=True, text=True)
    await update.message.reply_text(f"Cryptobot: {r.stdout.strip()}")


@auth
async def cmd_services(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all running services."""
    services = ["cryptobot", "telegram-bot", "cryptobot-monitor", "cryptobot-mcp"]
    text = "Services\n━━━━━━━━━━━━━━\n"
    for svc in services:
        r = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True)
        state = r.stdout.strip()
        icon = "●" if state == "active" else "○"
        text += f"  {icon} {svc}: {state}\n"
    await update.message.reply_text(text)


@auth
async def cmd_phone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check phone connection status and available tools."""
    try:
        import requests as _req
        phone = _req.get("http://127.0.0.1:8766/status", timeout=3).json()
        if phone.get("connected"):
            uptime_s = int(phone.get("uptime", 0))
            hrs = uptime_s // 3600
            mins = (uptime_s % 3600) // 60
            text = (
                f"Phone Status\n"
                f"━━━━━━━━━━━━━━\n"
                f"● Connected ({hrs}h {mins}m)\n"
                f"Clients: {phone.get('clients', 1)}\n\n"
                f"To control the phone, just tell me what to do!\n"
                f"Examples:\n"
                f"  • \"Open WhatsApp on my phone\"\n"
                f"  • \"Share my location with Bhavya\"\n"
                f"  • \"Set an alarm for 7am\"\n"
                f"  • \"Take a screenshot of my phone\""
            )
        else:
            text = "Phone: Not connected\n\nMake sure MIYA app is open on the phone."
    except Exception:
        text = "Phone brain server is offline.\nCheck: systemctl status miya-brain"
    await update.message.reply_text(text)


@auth
async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Memory commands. /memory, /memory search <query>, /memory count, /memory forget <query>"""
    if not mem0_client:
        await update.message.reply_text("Mem0 not connected. Set MEM0_API_KEY in .env")
        return

    uid = update.effective_user.id
    user_name = AUTHORIZED_USERS.get(uid, {}).get("name", "Unknown")
    mem0_uid = user_name.lower()
    args = context.args or []
    subcmd = args[0].lower() if args else "recent"
    query_text = " ".join(args[1:]) if len(args) > 1 else ""

    try:
        if subcmd == "count":
            raw = mem0_client.get_all(filters={"user_id": mem0_uid}, page=1, page_size=1)
            total = raw.get("count", 0) if isinstance(raw, dict) else "?"
            await update.message.reply_text(f"Total memories for {user_name}: {total}")

        elif subcmd == "forget" and query_text and uid == OWNER_ID:
            raw = mem0_client.search(query_text, filters={"user_id": mem0_uid}, top_k=3)
            results = raw.get("results", []) if isinstance(raw, dict) else raw
            if not results:
                await update.message.reply_text("No matching memories to forget.")
                return
            deleted = 0
            for r in results:
                mem_id = r.get("id")
                if mem_id:
                    mem0_client.delete(mem_id)
                    deleted += 1
            await update.message.reply_text(f"Forgot {deleted} memories matching '{query_text}'")

        elif subcmd == "search" and query_text:
            raw = mem0_client.search(query_text, filters={"user_id": mem0_uid}, top_k=10)
            results = raw.get("results", []) if isinstance(raw, dict) else raw
            if not results:
                await update.message.reply_text(f"No memories matching '{query_text}'")
                return
            text = f"Memories matching '{query_text}'\n━━━━━━━━━━━━━━\n"
            for r in results:
                mem = r.get("memory", "")
                cat = r.get("categories") or []
                source = (r.get("metadata") or {}).get("source", "")
                score = r.get("score", 0)
                tag = f" [{cat[0]}]" if cat else ""
                src = " (app)" if source == "miya_android" else " (tg)" if source == "miya_telegram" else ""
                text += f"• {mem}{tag}{src} ({score:.0%})\n"
            await send_response(update, text)

        else:
            raw = mem0_client.get_all(filters={"user_id": mem0_uid}, page=1, page_size=15)
            results = raw.get("results", []) if isinstance(raw, dict) else raw
            if not results:
                await update.message.reply_text("No memories yet. Chat with me and I'll remember!")
                return
            total = raw.get("count", len(results)) if isinstance(raw, dict) else len(results)
            text = f"Miya's memory ({user_name}) - {total} total\n━━━━━━━━━━━━━━\n"
            for r in results:
                mem = r.get("memory", "")
                cat = r.get("categories") or []
                source = (r.get("metadata") or {}).get("source", "")
                tag = f" [{cat[0]}]" if cat else ""
                src = " (app)" if source == "miya_android" else " (tg)" if source == "miya_telegram" else ""
                text += f"• {mem}{tag}{src}\n"
            text += f"\n/memory search <query>\n/memory count\n/memory forget <query>"
            await send_response(update, text)

    except Exception as e:
        await update.message.reply_text(f"Error: {e}")
