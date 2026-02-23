"""
Claude Code Bridge — Run Claude as subprocess with stream-json
==============================================================
"""

import os
import json
import logging
import subprocess
import time as _time
import threading

from tg.config import (
    WORKING_DIR, SYSTEM_PROMPT, BOT_TOKEN,
    current_session_id, _active_proc,
)

logger = logging.getLogger(__name__)


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
        return None
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
    import tg.config as _cfg

    try:
        sid = session_id or _cfg.current_session_id
        cmd = _build_claude_cmd(message, sid)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"}
        )

        if chat_id:
            _active_proc[chat_id] = proc

        output_lines = []
        result_text = ""
        start_time = _time.time()
        last_progress_time = 0
        progress_min_interval = 5
        timeout = 300
        timed_out = False

        def _process_line(line: str):
            nonlocal result_text, last_progress_time
            line = line.strip()
            if not line:
                return
            try:
                data = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                return

            msg_type = data.get("type", "")

            if msg_type == "result":
                result_text = data.get("result", "")
                return

            if msg_type == "assistant":
                content = data.get("message", {}).get("content", [])
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_use":
                        continue

                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})

                    now = _time.time()
                    if chat_id and (now - last_progress_time) >= progress_min_interval:
                        progress_msg = _format_tool_progress(tool_name, tool_input)
                        if progress_msg:
                            last_progress_time = now
                            _send_telegram_sync(chat_id, progress_msg)

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

        if chat_id:
            _active_proc.pop(chat_id, None)

        raw_output = "".join(output_lines)
        stderr = proc.stderr.read().strip() if proc.stderr else ""

        if not result_text:
            result_text = _extract_stream_result(raw_output)

        if "No conversation found" in (raw_output + stderr) and sid:
            logger.warning(f"Session {sid[:12]} not found, falling back to --continue")
            _cfg.current_session_id = None
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
