"""
Phone Remote Control — Push API bridge for MIYA Android App
============================================================
"""

import os
import json
import logging
import subprocess
import time as _time
import threading

from tg.config import WORKING_DIR
from tg.claude_bridge import _extract_stream_result

logger = logging.getLogger(__name__)

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

# Keywords that suggest phone control requests
_PHONE_KEYWORDS = [
    "open ", "whatsapp", "text ", "message ", "send ", "call ",
    "screenshot", "screen", "phone", "alarm", "timer",
    "brightness", "volume", "flashlight", "location",
    "tap ", "swipe", "type ", "battery",
]


def phone_tool(tool_name: str, params: dict) -> str:
    """Execute a phone tool via Push API."""
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
            if _time.time() - start > 180:
                proc.kill()
                break
            _time.sleep(1)
        reader.join(timeout=5)

        if not result_text:
            result_text = _extract_stream_result("".join(output_lines))
        return result_text or "Phone task completed."
    except Exception as e:
        return f"Phone agent error: {e}"


def is_phone_request(msg: str) -> bool:
    """Check if a message is a phone control request."""
    lower = msg.lower()
    return any(kw in lower for kw in _PHONE_KEYWORDS)
