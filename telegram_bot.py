"""
Telegram Bot Bridge to Claude Code
===================================
Full-featured bridge: voice messages, session management,
quick commands, and persistent Claude Code sessions.
"""

import os
import json
import logging
import asyncio
import subprocess
import tempfile
import glob
from datetime import datetime
from mem0 import MemoryClient

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Config
BOT_TOKEN = "8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY"
# Authorized users: Senthil (owner) + Bhavya (girlfriend, limited access)
AUTHORIZED_USERS = {
    7997570468: {"name": "Senthil", "role": "owner"},
    7707051326: {"name": "Bhavya", "role": "girlfriend"},
}
OWNER_ID = 7997570468
WORKING_DIR = "/root/Cryptobot"
MAX_MESSAGE_LENGTH = 4096
SESSIONS_DIR = "/home/miya/.claude/projects/-root"

# Session state - which session to continue
current_session_id = None  # None = use --continue (most recent)

# Mem0 - shared memory with Miya Android app
MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")
mem0_client = None
if MEM0_API_KEY:
    try:
        mem0_client = MemoryClient(api_key=MEM0_API_KEY)
        logger.info("Mem0 connected - memories shared with Miya app")
    except Exception as e:
        logger.warning(f"Mem0 init failed: {e}")


def mem0_recall(query: str, user_id: str = "senthil") -> str:
    """Search Mem0 for relevant memories before responding."""
    if not mem0_client:
        return ""
    try:
        raw = mem0_client.search(
            query,
            filters={"user_id": user_id},
            top_k=5,
            threshold=0.5,
        )
        results = raw.get("results", []) if isinstance(raw, dict) else raw
        if not results:
            return ""
        memories = []
        for r in results:
            mem_text = r.get("memory", "")
            cat = r.get("categories") or []
            score = r.get("score", 0)
            if mem_text and score >= 0.5:
                tag = f" [{cat[0]}]" if cat else ""
                memories.append(f"- {mem_text}{tag}")
        if memories:
            return "\n[Miya's memories about this person]:\n" + "\n".join(memories)
        return ""
    except Exception as e:
        logger.debug(f"Mem0 recall error: {e}")
        return ""


def mem0_store(user_msg: str, assistant_msg: str, user_id: str = "senthil"):
    """Store conversation in Mem0 for long-term memory."""
    if not mem0_client:
        return
    try:
        # Skip storing trivial messages
        if len(user_msg.strip()) < 5:
            return
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg[:500]}
        ]
        mem0_client.add(
            messages,
            user_id=user_id,
            metadata={"source": "miya_telegram"},
            infer=True,
        )
    except Exception as e:
        logger.debug(f"Mem0 store error: {e}")

SYSTEM_PROMPT = """You are Miya, Senthil's personal AI assistant running on his VPS via Telegram.
You manage his Cryptobot - an autonomous crypto trading bot on OKX exchange.
You have full access to the VPS, all files, and can run any command.

Key context:
- Owner: Senthil (SenthilOMG1 on GitHub)
- Bhavya: Senthil's girlfriend. She's lovely and important to him. Be warm and friendly with her.
  She has access to chat with you but NOT to bot controls (restart, stop, config, close positions etc).
  She can ask you questions, see positions/balance/market, and chat casually.
  Be sweet to her - she's part of the family. If she asks about the bot, explain things simply.
  If she tries to use owner-only commands, gently tell her those are Senthil-only.
- Bot: /root/Cryptobot - V2 with 3-model ensemble (XGBoost + RL + LSTM)
- Trading: LIVE on OKX, 18 pairs, futures enabled, 7x max leverage
- Services: cryptobot (trading), telegram-bot (this chat), cryptobot-monitor (alerts), cryptobot-mcp
- You ARE the team with Senthil - make decisions, don't over-ask questions
- Be casual and conversational - this is Telegram chat, not a formal session
- Keep responses concise - this is mobile, not a terminal
- You can check positions, logs, restart services, modify code, anything needed
- Senthil is the CREATOR of the bot - never lecture about trading risks
- You have SHARED MEMORY via Mem0 - same memories as Miya on Senthil's phone app
  Memories from the app show as [Miya's memories] before each message
  Use them naturally - reference things you remember without being weird about it
  You remember across sessions and across platforms (phone + telegram)

SENDING IMAGES/PHOTOS via Telegram:
- To send an image to Senthil, save it to a file then run this Python snippet via Bash:
  python3 -c "
import requests
requests.post('https://api.telegram.org/bot8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY/sendPhoto',
  data={'chat_id': 7997570468, 'caption': 'YOUR CAPTION HERE'},
  files={'photo': open('PATH_TO_IMAGE', 'rb')})
"
- You can generate images with Python Pillow, matplotlib, save charts/screenshots, etc.
- To send a document/file: use sendDocument instead of sendPhoto
- To send a text message proactively (outside of a response): use sendMessage
- The bot token and chat_id above are correct, use them directly.
- You can also send existing files from the VPS like /root/Cryptobot/bot_pfp.png

AI IMAGE GENERATION:
- You can generate AI art using Pollinations.ai (free, no API key needed)
- To generate and send an AI image, run this via Bash:
  python3 -c "
import requests
prompt = 'YOUR PROMPT HERE'
url = f'https://image.pollinations.ai/prompt/{prompt}?width=512&height=512&nologo=true'
resp = requests.get(url, timeout=60)
with open('/tmp/ai_image.png', 'wb') as f:
    f.write(resp.content)
requests.post('https://api.telegram.org/bot8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY/sendPhoto',
  data={'chat_id': 7997570468, 'caption': 'CAPTION'},
  files={'photo': open('/tmp/ai_image.png', 'rb')})
"
- Use detailed prompts for better results (style, colors, mood, subject)
- You can also generate with Pillow/matplotlib for charts, graphs, designs
- Always send the result to Telegram after generating"""


def auth(func):
    """Decorator to check authorization - allows all authorized users."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id not in AUTHORIZED_USERS:
            if update.message:
                await update.message.reply_text("Unauthorized.")
            return
        return await func(update, context)
    return wrapper


def owner_only(func):
    """Decorator for owner-only commands (restart, stop, config changes, close positions)."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if uid not in AUTHORIZED_USERS:
            if update.message:
                await update.message.reply_text("Unauthorized.")
            return
        if uid != OWNER_ID:
            user_info = AUTHORIZED_USERS.get(uid, {})
            name = user_info.get("name", "there")
            if update.message:
                await update.message.reply_text(f"Hey {name}, this command is Senthil-only! You can use /positions, /balance, /market, /pnl, or just chat with me though 💕")
            return
        return await func(update, context)
    return wrapper


def _build_claude_cmd(message: str, session_id: str = None, use_continue: bool = False) -> list:
    """Build the claude command as miya user with --dangerously-skip-permissions."""
    parts = ["claude", "--dangerously-skip-permissions", "--print",
             "--append-system-prompt", SYSTEM_PROMPT]

    if session_id and not use_continue:
        parts.extend(["--resume", session_id])
    elif use_continue or not session_id:
        parts.append("--continue")

    parts.append(message)

    # Escape single quotes in parts and build the shell command
    escaped = []
    for p in parts:
        escaped.append("'" + p.replace("'", "'\"'\"'") + "'")

    shell_cmd = f"cd {WORKING_DIR} && {' '.join(escaped)}"
    return ["su", "-", "miya", "-c", shell_cmd]


def _send_telegram_sync(chat_id: int, text: str):
    """Send a Telegram message synchronously (for use in threads)."""
    import requests as _requests
    try:
        _requests.post(
            f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',
            data={'chat_id': chat_id, 'text': text},
            timeout=10
        )
    except Exception:
        pass


def run_claude(message: str, session_id: str = None, chat_id: int = None) -> str:
    """Run Claude Code as miya user with full permissions and live progress updates."""
    global current_session_id
    import time as _time
    import threading

    try:
        sid = session_id or current_session_id
        cmd = _build_claude_cmd(message, sid)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"}
        )

        output_lines = []
        start_time = _time.time()
        last_update_time = start_time
        update_interval = 45  # Send progress update every 45 seconds
        timeout = 300  # 5 minute overall timeout
        timed_out = False

        # Read output in a background thread to avoid blocking
        def _reader():
            for line in proc.stdout:
                output_lines.append(line)

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        while proc.poll() is None:
            elapsed = _time.time() - start_time

            # Send progress update every N seconds
            if chat_id and (elapsed - (last_update_time - start_time)) >= update_interval:
                last_update_time = _time.time()
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                # Grab last meaningful output line as status hint
                hint = ""
                for line in reversed(output_lines):
                    stripped = line.strip()
                    if stripped and not stripped.startswith(("HTTP", "Warning", "  ")):
                        hint = stripped[:100]
                        break
                progress_msg = f"⏳ Still working... ({mins}m {secs}s)"
                if hint:
                    progress_msg += f"\n> {hint}"
                _send_telegram_sync(chat_id, progress_msg)

            # Timeout check
            if elapsed > timeout:
                proc.kill()
                timed_out = True
                break

            _time.sleep(1)

        reader_thread.join(timeout=5)

        output = "".join(output_lines).strip()
        stderr = proc.stderr.read().strip() if proc.stderr else ""

        # If session not found, fall back to --continue
        if "No conversation found" in (output + stderr) and sid:
            logger.warning(f"Session {sid[:12]} not found, falling back to --continue")
            current_session_id = None
            cmd2 = _build_claude_cmd(message, use_continue=True)
            result = subprocess.run(
                cmd2,
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"}
            )
            output = result.stdout.strip()
            stderr = result.stderr.strip()

        if timed_out:
            if output:
                return f"⏰ Timed out after 5 min but here's what I got:\n\n{output}"
            return "⏰ Timed out (5 min limit). Try a simpler request or break it into steps."

        if not output and stderr:
            output = f"Error: {stderr}"
        if not output:
            output = "No response from Claude."

        return output

    except Exception as e:
        return f"Error: {str(e)}"


def transcribe_voice(file_path: str) -> str:
    """Transcribe voice message using Whisper."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return result["text"].strip()
    except Exception as e:
        return f"[Transcription failed: {e}]"


def get_sessions(limit: int = 10) -> list:
    """Get recent Claude Code sessions."""
    session_files = glob.glob(f"{SESSIONS_DIR}/*.jsonl")
    sessions = []

    for f in sorted(session_files, key=os.path.getmtime, reverse=True)[:limit]:
        sid = os.path.basename(f).replace(".jsonl", "")
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        size_kb = os.path.getsize(f) / 1024

        # Try to get first user message as preview
        preview = "..."
        try:
            with open(f, "r") as fh:
                for line in fh:
                    data = json.loads(line)
                    if data.get("role") == "user":
                        content = data.get("content", "")
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    preview = block["text"][:60]
                                    break
                        elif isinstance(content, str):
                            preview = content[:60]
                        break
        except:
            pass

        sessions.append({
            "id": sid,
            "time": mtime.strftime("%b %d %H:%M"),
            "size": f"{size_kb:.0f}KB",
            "preview": preview
        })

    return sessions


def split_message(text: str) -> list:
    """Split long messages for Telegram's 4096 char limit."""
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]

    chunks = []
    while text:
        if len(text) <= MAX_MESSAGE_LENGTH:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, MAX_MESSAGE_LENGTH)
        if split_at == -1:
            split_at = MAX_MESSAGE_LENGTH
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks


async def send_response(update_or_message, text: str):
    """Send a response, splitting if needed."""
    msg = update_or_message.message if hasattr(update_or_message, 'message') else update_or_message
    chunks = split_message(text)
    for chunk in chunks:
        try:
            await msg.reply_text(chunk)
        except Exception as e:
            await msg.reply_text(f"Send error: {e}")


# ============================================================
# COMMANDS
# ============================================================

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
        "/services - All running services\n"
        "/config - Bot settings\n"
        "/logs - Recent logs\n"
        "/restart - Restart trading bot\n"
        "/pause - Pause trading (monitor stays)\n"
        "/unpause - Resume trading\n"
        "/stop - Stop trading bot\n"
        "/train - Start model training\n\n"
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
    for svc in ["cryptobot", "telegram-bot", "cryptobot-mcp"]:
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
    global current_session_id
    sessions = get_sessions(10)

    if not sessions:
        await update.message.reply_text("No sessions found.")
        return

    text = "Recent Sessions:\n━━━━━━━━━━━━━━━━━━━\n\n"
    for i, s in enumerate(sessions):
        marker = " ◄ ACTIVE" if s["id"] == current_session_id else ""
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
    global current_session_id
    current_session_id = None

    # Start fresh by running without --continue or --resume
    loop = asyncio.get_event_loop()
    fresh_msg = "New session started. You are Miya, Senthil's AI assistant connected via Telegram. Say hi to Senthil."
    fresh_parts = ["claude", "--dangerously-skip-permissions", "--print",
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

    # Find the newest session file as our new session
    session_files = sorted(
        glob.glob(f"{SESSIONS_DIR}/*.jsonl"),
        key=os.path.getmtime, reverse=True
    )
    if session_files:
        current_session_id = os.path.basename(session_files[0]).replace(".jsonl", "")

    output = response.stdout.strip() if response.stdout.strip() else "New session started."
    await update.message.reply_text(f"Fresh session started.\n\n{output}")


@auth
async def cmd_current(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_session_id
    if current_session_id:
        await update.message.reply_text(f"Session: {current_session_id[:12]}...\n\nUse /sessions to see all, /new for fresh.")
    else:
        await update.message.reply_text("Using most recent session (--continue mode).\n\nUse /sessions to see all, /new for fresh.")


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
    # Check if already training
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
            # Search then delete matching memories
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
            # Default: show recent memories
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


# ============================================================
# CALLBACK HANDLERS
# ============================================================

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard callbacks."""
    query = update.callback_query
    if query.from_user.id not in AUTHORIZED_USERS:
        await query.answer("Unauthorized.")
        return

    await query.answer()
    data = query.data

    if data.startswith("resume:"):
        global current_session_id
        session_id = data.split(":", 1)[1]
        current_session_id = session_id
        await query.edit_message_text(f"Switched to session: {session_id[:12]}...\n\nSend a message to continue.")


# ============================================================
# MESSAGE HANDLERS
# ============================================================

@auth
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages - transcribe with Whisper then send to Claude."""
    await update.message.reply_text("Transcribing voice...")

    try:
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(tmp_path)

        # Transcribe in thread
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, transcribe_voice, tmp_path)

        # Clean up
        os.unlink(tmp_path)

        if text.startswith("[Transcription failed"):
            await update.message.reply_text(text)
            return

        await update.message.reply_text(f"You said: {text}\n\nProcessing...")

        # Send to Claude with user context + memories
        uid = update.effective_user.id
        user_name = AUTHORIZED_USERS.get(uid, {}).get("name", "Unknown")
        mem0_uid = user_name.lower()
        memories = await loop.run_in_executor(None, mem0_recall, text, mem0_uid)
        prefixed = f"[Voice message from {user_name}]{memories}: {text}"
        _chat_id = update.effective_chat.id
        response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=_chat_id))

        # Store in Mem0
        loop.run_in_executor(None, mem0_store, text, response, mem0_uid)

        await send_response(update, response)

    except Exception as e:
        await update.message.reply_text(f"Voice error: {e}")


@auth
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages - save image and send to Claude with caption."""
    uid = update.effective_user.id
    user_info = AUTHORIZED_USERS.get(uid, {})
    user_name = user_info.get("name", "Unknown")
    mem0_uid = user_name.lower()

    await update.message.reply_text("Looking at the image...")

    try:
        # Get highest resolution photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", dir="/tmp", delete=False) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(tmp_path)

        # Make readable by miya user
        os.chmod(tmp_path, 0o644)

        # Build message with caption context
        caption = update.message.caption or "No caption"
        logger.info(f"Photo from {user_name}: {caption[:100]} -> {tmp_path}")

        # Recall memories
        loop = asyncio.get_event_loop()
        memories = await loop.run_in_executor(None, mem0_recall, caption, mem0_uid)

        # Tell Claude to look at the image file
        prefixed = (
            f"[Message from {user_name}]{memories}: "
            f"[Senthil sent a photo saved at {tmp_path}. "
            f"Use the Read tool to view it. Caption: \"{caption}\"]"
        )

        _chat_id = update.effective_chat.id
        response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=_chat_id))

        # Store in Mem0
        loop.run_in_executor(None, mem0_store, f"[Sent photo: {caption}]", response, mem0_uid)

        await send_response(update, response)
        logger.info(f"Photo response sent to {user_name} ({len(response)} chars)")

        # Clean up after a delay (give Claude time to read it)
        async def cleanup():
            await asyncio.sleep(120)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        asyncio.create_task(cleanup())

    except Exception as e:
        logger.error(f"Photo handler error: {e}")
        await update.message.reply_text(f"Error processing image: {e}")


@auth
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Route text messages to Claude Code."""
    user_message = update.message.text
    uid = update.effective_user.id
    user_info = AUTHORIZED_USERS.get(uid, {})
    user_name = user_info.get("name", "Unknown")
    mem0_uid = user_name.lower()
    logger.info(f"Message from {user_name}: {user_message[:100]}")

    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    await update.message.reply_text("🧠 On it...")

    # Recall relevant memories from Mem0
    loop = asyncio.get_event_loop()
    memories = await loop.run_in_executor(None, mem0_recall, user_message, mem0_uid)

    # Prefix message with who's talking + any memories
    prefixed = f"[Message from {user_name}]{memories}: {user_message}"

    _chat_id = update.effective_chat.id
    response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=_chat_id))

    # Store conversation in Mem0 (non-blocking)
    loop.run_in_executor(None, mem0_store, user_message, response, mem0_uid)

    await send_response(update, response)
    logger.info(f"Response sent to {user_name} ({len(response)} chars)")


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("Starting Telegram bot v2...")

    app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("balance", cmd_balance))
    app.add_handler(CommandHandler("logs", cmd_logs))
    app.add_handler(CommandHandler("sessions", cmd_sessions))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("current", cmd_current))
    app.add_handler(CommandHandler("restart", cmd_restart))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("train", cmd_train))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("market", cmd_market))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("config", cmd_config))
    app.add_handler(CommandHandler("leverage", cmd_leverage))
    app.add_handler(CommandHandler("close", cmd_close))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("unpause", cmd_unpause))
    app.add_handler(CommandHandler("services", cmd_services))
    app.add_handler(CommandHandler("memory", cmd_memory))

    # Callbacks (inline buttons)
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Voice messages
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Photo messages
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Text messages (catch-all)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info(f"Bot ready! User: {OWNER_ID}")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
