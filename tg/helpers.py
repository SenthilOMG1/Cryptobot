"""
Telegram Helpers — Whisper, message splitting, session listing, media extraction
================================================================================
"""

import os
import json
import glob
import logging
import subprocess
from datetime import datetime

from tg.config import MAX_MESSAGE_LENGTH, SESSIONS_DIR

logger = logging.getLogger(__name__)

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
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=15
        )
        duration = float(probe.stdout.strip() or '0')
        if duration <= 0:
            return frames

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
