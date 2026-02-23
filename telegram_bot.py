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

# Per-chat concurrency: lock + message queue so users can send while Miya is busy
_chat_locks = {}   # chat_id -> asyncio.Lock
_chat_queues = {}  # chat_id -> list of (user_name, mem0_uid, message_text)
_active_proc = {}  # chat_id -> subprocess.Popen (for /steer to kill)

# Background phone tasks - run independently so they don't block chat
_phone_task_running = False


def _get_chat_lock(chat_id: int):
    """Get or create an asyncio.Lock for a chat."""
    if chat_id not in _chat_locks:
        _chat_locks[chat_id] = asyncio.Lock()
    return _chat_locks[chat_id]

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
- Always send the result to Telegram after generating

PHONE REMOTE CONTROL (via MIYA Android App):
- Senthil's phone runs the MIYA app which connects to the VPS via WebSocket
- You can control the phone remotely through the Push API at http://127.0.0.1:8766
- Check if phone is connected: curl http://127.0.0.1:8766/status
- Execute a tool on the phone: curl -X POST http://127.0.0.1:8766/tool -H 'Content-Type: application/json' -d '{"tool": "tool_name", "params": {"key": "value"}}'
- Send a display message to phone: curl -X POST http://127.0.0.1:8766/push -H 'Content-Type: application/json' -d '{"message": "text to show"}'
- Available phone tools include: open_app, set_timer, set_alarm, phone_tap, phone_swipe, phone_type, take_screenshot, get_clipboard, set_clipboard, send_sms, make_call, get_location, toggle_flashlight, set_volume, get_battery, get_wifi_info, share_location
- Example: To open WhatsApp: curl -X POST http://127.0.0.1:8766/tool -H 'Content-Type: application/json' -d '{"tool": "open_app", "params": {"package": "com.whatsapp"}}'
- Example: To share location: First get location with get_location tool, then open WhatsApp and share it
- IMPORTANT: If the phone is not connected, tell the user and don't try to push commands
- Bhavya can ask you to do phone things too (like "open WhatsApp and share location") - check phone status first, then execute"""


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


def _format_tool_progress(tool_name: str, tool_input: dict) -> str:
    """Format a tool call into a nice Telegram progress message."""
    icons = {
        "Read": "📖", "Write": "✏️", "Edit": "✏️",
        "Bash": "🔧", "Glob": "🔍", "Grep": "🔍",
        "WebFetch": "🌐", "WebSearch": "🌐",
        "Task": "🚀", "TaskCreate": "📋", "TaskUpdate": "✅",
    }
    icon = icons.get(tool_name, "⚙️")

    if tool_name == "Read":
        path = tool_input.get("file_path", "?")
        short = path.rsplit("/", 1)[-1]
        return f"{icon} Reading {short}"
    elif tool_name in ("Write", "Edit"):
        path = tool_input.get("file_path", "?")
        short = path.rsplit("/", 1)[-1]
        return f"{icon} Editing {short}"
    elif tool_name == "Bash":
        desc = tool_input.get("description", "")
        if desc:
            return f"{icon} {desc}"
        cmd = tool_input.get("command", "")[:80]
        return f"{icon} Running: {cmd}"
    elif tool_name in ("Glob", "Grep"):
        pattern = tool_input.get("pattern", "")[:60]
        return f"{icon} Searching: {pattern}"
    elif tool_name == "WebSearch":
        query = tool_input.get("query", "")[:60]
        return f"{icon} Searching: {query}"
    elif tool_name == "WebFetch":
        url = tool_input.get("url", "")[:60]
        return f"{icon} Fetching: {url}"
    elif tool_name == "TaskCreate":
        subject = tool_input.get("subject", "")
        return f"{icon} Task: {subject}"
    elif tool_name == "TaskUpdate":
        status = tool_input.get("status", "")
        if status == "completed":
            return f"✅ Task completed"
        return None  # skip non-completion updates
    else:
        return f"{icon} {tool_name}"


def _build_claude_cmd(message: str, session_id: str = None, use_continue: bool = False) -> list:
    """Build the claude command as miya user with --dangerously-skip-permissions."""
    parts = ["claude", "--dangerously-skip-permissions",
             "--verbose", "--output-format", "stream-json",
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


def _extract_stream_result(raw_output: str) -> str:
    """Extract the final result text from stream-json output."""
    for line in raw_output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if data.get("type") == "result":
                return data.get("result", "")
        except (json.JSONDecodeError, TypeError):
            continue
    return ""


def run_claude(message: str, session_id: str = None, chat_id: int = None) -> str:
    """Run Claude Code as miya user with live tool progress via stream-json."""
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

        # Track active process so /steer can interrupt
        if chat_id:
            _active_proc[chat_id] = proc

        output_lines = []
        result_text = ""
        start_time = _time.time()
        last_progress_time = 0
        progress_min_interval = 5  # Min seconds between progress messages
        timeout = 300
        timed_out = False

        def _process_line(line: str):
            """Parse a stream-json line and send tool progress to Telegram."""
            nonlocal result_text, last_progress_time

            line = line.strip()
            if not line:
                return
            try:
                data = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                return

            msg_type = data.get("type", "")

            # Capture final result
            if msg_type == "result":
                result_text = data.get("result", "")
                return

            # Look for tool calls in assistant messages
            if msg_type == "assistant":
                content = data.get("message", {}).get("content", [])
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_use":
                        continue

                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})

                    # Rate limit progress updates
                    now = _time.time()
                    if chat_id and (now - last_progress_time) >= progress_min_interval:
                        progress_msg = _format_tool_progress(tool_name, tool_input)
                        if progress_msg:
                            last_progress_time = now
                            _send_telegram_sync(chat_id, progress_msg)

        # Read and process output in background thread
        def _reader():
            for line in proc.stdout:
                output_lines.append(line)
                try:
                    _process_line(line)
                except Exception:
                    pass

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        while proc.poll() is None:
            elapsed = _time.time() - start_time
            if elapsed > timeout:
                proc.kill()
                timed_out = True
                break
            _time.sleep(1)

        reader_thread.join(timeout=5)

        # Clear active process tracking
        if chat_id:
            _active_proc.pop(chat_id, None)

        raw_output = "".join(output_lines)
        stderr = proc.stderr.read().strip() if proc.stderr else ""

        # If result wasn't captured from stream, try parsing all output
        if not result_text:
            result_text = _extract_stream_result(raw_output)

        # If session not found, fall back to --continue
        if "No conversation found" in (raw_output + stderr) and sid:
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
            result_text = _extract_stream_result(result.stdout)
            stderr = result.stderr.strip()

        if timed_out:
            if result_text:
                return f"⏰ Timed out after 5 min but here's what I got:\n\n{result_text}"
            return "⏰ Timed out (5 min limit). Try a simpler request or break it into steps."

        if not result_text and stderr:
            result_text = f"Error: {stderr}"
        if not result_text:
            result_text = "No response from Claude."

        return result_text

    except Exception as e:
        return f"Error: {str(e)}"


def phone_tool(tool_name: str, params: dict) -> str:
    """Execute a phone tool via Push API. Fast, doesn't need Claude subprocess."""
    import requests as _req
    try:
        r = _req.post("http://127.0.0.1:8766/tool",
                       json={"tool": tool_name, "params": params}, timeout=30)
        data = r.json()
        return data.get("result", "No result")
    except Exception as e:
        return f"Error: {e}"


def phone_status() -> dict:
    """Check phone connection status."""
    import requests as _req
    try:
        return _req.get("http://127.0.0.1:8766/status", timeout=3).json()
    except Exception:
        return {"connected": False}


PHONE_AGENT_PROMPT = """You are Miya's phone control subagent. You control Senthil's phone via the Push API.
You run in background while Miya chats normally on Telegram.

RULES:
- Execute phone commands via Bash using python3 with requests to http://127.0.0.1:8766/tool
- Available tools: phone_open_app (param: app), phone_tap (params: x, y), phone_type (param: text),
  phone_read_screen, phone_scroll (param: direction), phone_go_back, phone_go_home,
  phone_swipe (params: direction), get_battery, get_location, set_timer (param: minutes),
  set_alarm (params: hour, minute), phone_long_press (params: x, y), phone_wait (param: seconds)
- To execute a tool: python3 -c "import requests; r=requests.post('http://127.0.0.1:8766/tool', json={'tool':'TOOL','params':{}}, timeout=30); print(r.text)"
- Read the screen before tapping to find element coordinates
- Wait 1-2 seconds between actions for the phone to respond
- When typing in WhatsApp/apps: tap the input field FIRST, wait, then type
- Keep it concise - report what you did, not every detail
- If something fails, try a different approach"""


def run_phone_agent(task: str, chat_id: int) -> str:
    """Run a separate Claude subprocess for phone control. Does NOT hold the chat lock."""
    try:
        parts = [
            "claude", "-p", "--dangerously-skip-permissions",
            "--output-format", "stream-json", "--verbose",
            "--append-system-prompt", PHONE_AGENT_PROMPT,
            task
        ]
        escaped = []
        for p in parts:
            escaped.append("'" + p.replace("'", "'\"'\"'") + "'")
        shell_cmd = f"cd {WORKING_DIR} && {' '.join(escaped)}"
        cmd = ["su", "-", "miya", "-c", shell_cmd]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"}
        )

        import time as _time, threading
        output_lines = []
        result_text = ""

        def _process_line(line):
            nonlocal result_text
            line = line.strip()
            if not line:
                return
            try:
                data = json.loads(line)
                if data.get("type") == "result":
                    result_text = data.get("result", "")
            except (json.JSONDecodeError, TypeError):
                pass

        def _reader():
            for line in proc.stdout:
                output_lines.append(line)
                try:
                    _process_line(line)
                except Exception:
                    pass

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        start = _time.time()
        while proc.poll() is None:
            if _time.time() - start > 180:  # 3 min timeout for phone tasks
                proc.kill()
                break
            _time.sleep(1)
        reader.join(timeout=5)

        if not result_text:
            result_text = _extract_stream_result("".join(output_lines))
        return result_text or "Phone task completed."
    except Exception as e:
        return f"Phone agent error: {e}"


# Keywords that suggest phone control requests
_PHONE_KEYWORDS = [
    "open ", "whatsapp", "text ", "message ", "send ", "call ",
    "screenshot", "screen", "phone", "alarm", "timer",
    "brightness", "volume", "flashlight", "location",
    "tap ", "swipe", "type ", "battery",
]


def _is_phone_request(msg: str) -> bool:
    """Check if a message is a phone control request."""
    lower = msg.lower()
    return any(kw in lower for kw in _PHONE_KEYWORDS)


# Whisper model - loaded once at startup for fast transcription
_whisper_model = None

def _get_whisper_model():
    """Load Whisper model once and cache it."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info("Loading Whisper base model...")
        _whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded!")
    return _whisper_model


def transcribe_voice(file_path: str) -> str:
    """Transcribe voice/audio using cached Whisper model."""
    try:
        model = _get_whisper_model()
        result = model.transcribe(file_path)
        return result["text"].strip()
    except Exception as e:
        return f"[Transcription failed: {e}]"


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio track from video using ffmpeg. Returns path to WAV file."""
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    try:
        subprocess.run(
            ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
             '-ar', '16000', '-ac', '1', audio_path, '-y'],
            capture_output=True, timeout=60
        )
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            return audio_path
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
    return ""


def extract_video_frames(video_path: str, max_frames: int = 4) -> list:
    """Extract key frames from video at even intervals. Returns list of image paths."""
    frames = []
    try:
        # Get video duration
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=15
        )
        duration = float(probe.stdout.strip() or '0')
        if duration <= 0:
            return frames

        # Extract frames at even intervals
        interval = duration / (max_frames + 1)
        for i in range(1, max_frames + 1):
            timestamp = interval * i
            frame_path = f"/tmp/frame_{os.getpid()}_{i}.jpg"
            subprocess.run(
                ['ffmpeg', '-ss', str(timestamp), '-i', video_path,
                 '-vframes', '1', '-q:v', '2', frame_path, '-y'],
                capture_output=True, timeout=15
            )
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                os.chmod(frame_path, 0o644)
                frames.append(frame_path)
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
    return frames


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

    # Check phone connection
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

    # Find the newest session file as our new session
    session_files = sorted(
        glob.glob(f"{SESSIONS_DIR}/*.jsonl"),
        key=os.path.getmtime, reverse=True
    )
    if session_files:
        current_session_id = os.path.basename(session_files[0]).replace(".jsonl", "")

    output = _extract_stream_result(response.stdout) if response.stdout else "New session started."
    await update.message.reply_text(f"Fresh session started.\n\n{output}")


@auth
async def cmd_current(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_session_id
    if current_session_id:
        await update.message.reply_text(f"Session: {current_session_id[:12]}...\n\nUse /sessions to see all, /new for fresh.")
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

    # Kill the active Claude process if running
    proc = _active_proc.get(chat_id)
    if proc and proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)
        _active_proc.pop(chat_id, None)
        await update.message.reply_text(f"🔀 Interrupted. Redirecting with: {guidance[:100]}")
    else:
        await update.message.reply_text(f"🔀 No active task — sending guidance directly.")

    # Clear any queued messages and combine with steer
    queued = _chat_queues.pop(chat_id, [])
    queued_text = ""
    if queued:
        queued_text = "\nAlso, queued messages: " + " | ".join(msg for _, _, msg in queued)

    # Release the lock if held (the killed process's handler will release it,
    # but we send the steer as a fresh call)
    lock = _get_chat_lock(chat_id)

    # Wait briefly for lock to release after kill
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
        voice = update.message.voice or update.message.audio
        if not voice:
            await update.message.reply_text("No audio found in message.")
            return
        file = await context.bot.get_file(voice.file_id)

        suffix = ".ogg" if update.message.voice else ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
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

        # Send to Claude with user context + memories
        uid = update.effective_user.id
        user_name = AUTHORIZED_USERS.get(uid, {}).get("name", "Unknown")
        mem0_uid = user_name.lower()
        _chat_id = update.effective_chat.id
        lock = _get_chat_lock(_chat_id)

        # Queue if Miya is busy
        if lock.locked():
            _chat_queues.setdefault(_chat_id, []).append((user_name, mem0_uid, f"[Voice: {text}]"))
            await update.message.reply_text(f"🎤 Transcribed: \"{text[:100]}\"\n📬 Queued — she's busy.")
            return

        await update.message.reply_text(f"You said: {text}\n\nProcessing...")

        async with lock:
            memories = await loop.run_in_executor(None, mem0_recall, text, mem0_uid)
            prefixed = f"[Voice message from {user_name}]{memories}: {text}"
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

        _chat_id = update.effective_chat.id
        lock = _get_chat_lock(_chat_id)
        loop = asyncio.get_event_loop()

        if lock.locked():
            _chat_queues.setdefault(_chat_id, []).append((user_name, mem0_uid, f"[Photo saved at {tmp_path}, caption: \"{caption}\"]"))
            await update.message.reply_text("📬 Photo saved & queued — she's busy.")
            return

        async with lock:
            memories = await loop.run_in_executor(None, mem0_recall, caption, mem0_uid)
            prefixed = (
                f"[Message from {user_name}]{memories}: "
                f"[Senthil sent a photo saved at {tmp_path}. "
                f"Use the Read tool to view it. Caption: \"{caption}\"]"
            )

            response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=_chat_id))
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
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle video/video note/animation messages - extract audio + frames, send to Claude."""
    uid = update.effective_user.id
    user_info = AUTHORIZED_USERS.get(uid, {})
    user_name = user_info.get("name", "Unknown")
    mem0_uid = user_name.lower()

    await update.message.reply_text("Processing video...")

    try:
        # Get the video file (supports video, video_note, animation)
        video = update.message.video or update.message.video_note or update.message.animation
        if not video:
            await update.message.reply_text("No video found.")
            return

        file = await context.bot.get_file(video.file_id)

        with tempfile.NamedTemporaryFile(suffix=".mp4", dir="/tmp", delete=False) as tmp:
            video_path = tmp.name
            await file.download_to_drive(video_path)

        os.chmod(video_path, 0o644)
        caption = update.message.caption or ""
        loop = asyncio.get_event_loop()

        # Extract audio and transcribe
        audio_path = await loop.run_in_executor(None, extract_audio_from_video, video_path)
        transcript = ""
        if audio_path:
            transcript = await loop.run_in_executor(None, transcribe_voice, audio_path)
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        # Extract key frames for visual context
        frame_paths = await loop.run_in_executor(None, extract_video_frames, video_path)

        # Build the message for Claude
        parts = []
        if transcript and not transcript.startswith("[Transcription failed"):
            parts.append(f"Audio transcript: \"{transcript}\"")
            await update.message.reply_text(f"Audio: {transcript[:200]}{'...' if len(transcript) > 200 else ''}\n\nAnalyzing...")
        else:
            await update.message.reply_text("No audio detected. Analyzing frames...")

        if frame_paths:
            parts.append(f"Video frames saved at: {', '.join(frame_paths)} (use Read tool to view them)")
        if caption:
            parts.append(f"Caption: \"{caption}\"")

        video_desc = ". ".join(parts) if parts else "Video had no extractable audio or frames"

        _chat_id = update.effective_chat.id
        lock = _get_chat_lock(_chat_id)

        if lock.locked():
            _chat_queues.setdefault(_chat_id, []).append((user_name, mem0_uid, f"[Video: {video_desc}]"))
            await update.message.reply_text("📬 Video processed & queued — she's busy.")
            return

        async with lock:
            query = caption or transcript or "sent a video"
            memories = await loop.run_in_executor(None, mem0_recall, query, mem0_uid)

            prefixed = (
                f"[Message from {user_name}]{memories}: "
                f"[Senthil sent a video. {video_desc}]"
            )

            response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=_chat_id))
            loop.run_in_executor(None, mem0_store, f"[Sent video: {caption or transcript[:60] or 'no caption'}]", response, mem0_uid)

            await send_response(update, response)

        # Clean up after delay
        async def cleanup():
            await asyncio.sleep(120)
            for f in [video_path] + frame_paths:
                try:
                    os.unlink(f)
                except OSError:
                    pass
        asyncio.create_task(cleanup())

    except Exception as e:
        logger.error(f"Video handler error: {e}")
        await update.message.reply_text(f"Error processing video: {e}")


@auth
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document messages - audio files, videos sent as files, etc."""
    uid = update.effective_user.id
    user_info = AUTHORIZED_USERS.get(uid, {})
    user_name = user_info.get("name", "Unknown")
    mem0_uid = user_name.lower()

    doc = update.message.document
    if not doc:
        return

    mime = doc.mime_type or ""
    filename = doc.file_name or "unknown"

    # Audio files -> transcribe
    if mime.startswith("audio/"):
        await update.message.reply_text(f"Transcribing {filename}...")
        try:
            file = await context.bot.get_file(doc.file_id)
            suffix = os.path.splitext(filename)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, dir="/tmp", delete=False) as tmp:
                tmp_path = tmp.name
                await file.download_to_drive(tmp_path)

            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, transcribe_voice, tmp_path)
            os.unlink(tmp_path)

            if text.startswith("[Transcription failed"):
                await update.message.reply_text(text)
                return

            await update.message.reply_text(f"Audio: {text[:200]}{'...' if len(text) > 200 else ''}\n\nProcessing...")

            caption = update.message.caption or ""
            memories = await loop.run_in_executor(None, mem0_recall, text, mem0_uid)
            prefixed = f"[Message from {user_name}]{memories}: [Sent audio file '{filename}'. Transcript: \"{text}\"{f'. Caption: {caption}' if caption else ''}]"
            _chat_id = update.effective_chat.id
            response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=_chat_id))
            loop.run_in_executor(None, mem0_store, text, response, mem0_uid)
            await send_response(update, response)
            return
        except Exception as e:
            await update.message.reply_text(f"Audio processing error: {e}")
            return

    # Video files -> same as video handler
    if mime.startswith("video/"):
        await update.message.reply_text(f"Processing video {filename}...")
        try:
            file = await context.bot.get_file(doc.file_id)
            suffix = os.path.splitext(filename)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, dir="/tmp", delete=False) as tmp:
                video_path = tmp.name
                await file.download_to_drive(video_path)

            os.chmod(video_path, 0o644)
            loop = asyncio.get_event_loop()

            audio_path = await loop.run_in_executor(None, extract_audio_from_video, video_path)
            transcript = ""
            if audio_path:
                transcript = await loop.run_in_executor(None, transcribe_voice, audio_path)
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

            frame_paths = await loop.run_in_executor(None, extract_video_frames, video_path)

            parts = []
            if transcript and not transcript.startswith("[Transcription failed"):
                parts.append(f"Audio transcript: \"{transcript}\"")
            if frame_paths:
                parts.append(f"Video frames saved at: {', '.join(frame_paths)} (use Read tool to view them)")

            caption = update.message.caption or ""
            if caption:
                parts.append(f"Caption: \"{caption}\"")

            video_desc = ". ".join(parts) if parts else "No extractable content"
            query = caption or transcript or "sent a video"
            memories = await loop.run_in_executor(None, mem0_recall, query, mem0_uid)
            prefixed = f"[Message from {user_name}]{memories}: [Sent video file '{filename}'. {video_desc}]"
            _chat_id = update.effective_chat.id
            response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=_chat_id))
            loop.run_in_executor(None, mem0_store, f"[Sent video: {caption or filename}]", response, mem0_uid)
            await send_response(update, response)

            async def cleanup():
                await asyncio.sleep(120)
                for f in [video_path] + frame_paths:
                    try:
                        os.unlink(f)
                    except OSError:
                        pass
            asyncio.create_task(cleanup())
            return
        except Exception as e:
            await update.message.reply_text(f"Video processing error: {e}")
            return

    # Other documents - just pass filename and caption to Claude
    caption = update.message.caption or ""
    _chat_id = update.effective_chat.id
    lock = _get_chat_lock(_chat_id)
    loop = asyncio.get_event_loop()

    if lock.locked():
        _chat_queues.setdefault(_chat_id, []).append((user_name, mem0_uid, f"[File: '{filename}', caption: \"{caption}\"]"))
        await update.message.reply_text("📬 File noted & queued — she's busy.")
        return

    async with lock:
        memories = await loop.run_in_executor(None, mem0_recall, caption or filename, mem0_uid)
        prefixed = f"[Message from {user_name}]{memories}: [Sent a file: '{filename}' (type: {mime}). Caption: \"{caption}\"]"
        response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=_chat_id))
        await send_response(update, response)


@auth
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Route text messages to Claude Code. Queues messages if Miya is busy."""
    user_message = update.message.text
    uid = update.effective_user.id
    user_info = AUTHORIZED_USERS.get(uid, {})
    user_name = user_info.get("name", "Unknown")
    mem0_uid = user_name.lower()
    chat_id = update.effective_chat.id
    logger.info(f"Message from {user_name}: {user_message[:100]}")

    lock = _get_chat_lock(chat_id)

    # If Miya is busy processing, queue the message but don't block
    if lock.locked():
        _chat_queues.setdefault(chat_id, []).append((user_name, mem0_uid, user_message))
        count = len(_chat_queues[chat_id])
        # Don't spam "queued" messages - they know she's working
        if count <= 1:
            await update.message.reply_text("📬 Got it — I'll get to this right after.")
        return

    # Check if this is a phone control request - run as background subagent
    if _is_phone_request(user_message):
        status = phone_status()
        if not status.get("connected"):
            await update.message.reply_text("Phone's not connected rn. Make sure MIYA app is open.")
            return

        await update.message.reply_text(f"📱 On it — controlling phone in background. Keep chatting!")
        loop = asyncio.get_event_loop()

        # Run phone agent in background - doesn't hold any lock
        async def _bg_phone():
            try:
                memories = await loop.run_in_executor(None, mem0_recall, user_message, mem0_uid)
                task = f"[Phone control request from {user_name}]{memories}: {user_message}"
                result = await loop.run_in_executor(None, run_phone_agent, task, chat_id)
                loop.run_in_executor(None, mem0_store, user_message, result, mem0_uid)
                # Send result back to Telegram
                for chunk in split_message(f"📱 {result}"):
                    _send_telegram_sync(chat_id, chunk)
            except Exception as e:
                _send_telegram_sync(chat_id, f"📱 Phone task failed: {e}")

        asyncio.create_task(_bg_phone())
        return

    async with lock:
        # Send typing action
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        await update.message.reply_text("🧠 On it...")

        # Recall relevant memories from Mem0
        loop = asyncio.get_event_loop()
        memories = await loop.run_in_executor(None, mem0_recall, user_message, mem0_uid)

        # Prefix message with who's talking + any memories
        prefixed = f"[Message from {user_name}]{memories}: {user_message}"

        response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=chat_id))

        # Store conversation in Mem0 (non-blocking)
        loop.run_in_executor(None, mem0_store, user_message, response, mem0_uid)

        await send_response(update, response)
        logger.info(f"Response sent to {user_name} ({len(response)} chars)")

        # Drain queued messages — combine and send as one follow-up
        while _chat_queues.get(chat_id):
            queued = _chat_queues.pop(chat_id, [])
            combined_texts = [msg for _, _, msg in queued]
            combined = "\n".join(combined_texts)
            q_user = queued[0][0]  # user_name from first queued msg
            q_mem0 = queued[0][1]  # mem0_uid from first queued msg
            logger.info(f"Processing {len(queued)} queued messages from {q_user}")

            memories = await loop.run_in_executor(None, mem0_recall, combined, q_mem0)
            prefixed = f"[Follow-up from {q_user} — {len(queued)} messages sent while you were working]{memories}: {combined}"

            response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=chat_id))
            loop.run_in_executor(None, mem0_store, combined, response, q_mem0)
            await send_response(update, response)


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("Starting Telegram bot v2...")

    app = Application.builder().token(BOT_TOKEN).concurrent_updates(True).build()

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
    app.add_handler(CommandHandler("steer", cmd_steer))
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
    app.add_handler(CommandHandler("phone", cmd_phone))
    app.add_handler(CommandHandler("memory", cmd_memory))

    # Callbacks (inline buttons)
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Voice messages
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Video messages (video, video notes/circles, GIFs)
    app.add_handler(MessageHandler(
        filters.VIDEO | filters.VIDEO_NOTE | filters.ANIMATION, handle_video
    ))

    # Photo messages
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Document messages (audio files, video files, other docs)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Text messages (catch-all)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info(f"Bot ready! User: {OWNER_ID}")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
