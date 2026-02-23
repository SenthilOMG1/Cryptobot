"""
Telegram Bot Package
====================
Split from 1,878-line monolith into focused modules:
  config.py       — Constants, auth decorators, system prompt
  mem0_bridge.py  — Shared memory with Miya Android app
  claude_bridge.py — Run Claude Code subprocess with stream-json
  phone.py        — Phone remote control via Push API
  helpers.py      — Whisper, message splitting, session listing
  commands.py     — All /slash command handlers
  media.py        — Voice, photo, video, document, text handlers
"""
