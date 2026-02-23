"""
Media & Message Handlers — Voice, Photo, Video, Document, Text
==============================================================
"""

import os
import asyncio
import logging
import tempfile

from telegram import Update
from telegram.ext import ContextTypes

from tg.config import (
    auth, AUTHORIZED_USERS,
    _chat_queues, _get_chat_lock,
)
import tg.config as _cfg
from tg.claude_bridge import run_claude, _send_telegram_sync
from tg.mem0_bridge import mem0_recall, mem0_store
from tg.phone import phone_status, run_phone_agent, is_phone_request
from tg.helpers import (
    transcribe_voice, extract_audio_from_video, extract_video_frames,
    split_message, send_response,
)

logger = logging.getLogger(__name__)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard callbacks."""
    query = update.callback_query
    if query.from_user.id not in AUTHORIZED_USERS:
        await query.answer("Unauthorized.")
        return

    await query.answer()
    data = query.data

    if data.startswith("resume:"):
        session_id = data.split(":", 1)[1]
        _cfg.current_session_id = session_id
        await query.edit_message_text(f"Switched to session: {session_id[:12]}...\n\nSend a message to continue.")


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

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, transcribe_voice, tmp_path)

        os.unlink(tmp_path)

        if text.startswith("[Transcription failed"):
            await update.message.reply_text(text)
            return

        uid = update.effective_user.id
        user_name = AUTHORIZED_USERS.get(uid, {}).get("name", "Unknown")
        mem0_uid = user_name.lower()
        _chat_id = update.effective_chat.id
        lock = _get_chat_lock(_chat_id)

        if lock.locked():
            _chat_queues.setdefault(_chat_id, []).append((user_name, mem0_uid, f"[Voice: {text}]"))
            await update.message.reply_text(f"🎤 Transcribed: \"{text[:100]}\"\n📬 Queued — she's busy.")
            return

        await update.message.reply_text(f"You said: {text}\n\nProcessing...")

        async with lock:
            memories = await loop.run_in_executor(None, mem0_recall, text, mem0_uid)
            prefixed = f"[Voice message from {user_name}]{memories}: {text}"
            response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=_chat_id))

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
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        with tempfile.NamedTemporaryFile(suffix=".jpg", dir="/tmp", delete=False) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(tmp_path)

        os.chmod(tmp_path, 0o644)

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

    if lock.locked():
        _chat_queues.setdefault(chat_id, []).append((user_name, mem0_uid, user_message))
        count = len(_chat_queues[chat_id])
        if count <= 1:
            await update.message.reply_text("📬 Got it — I'll get to this right after.")
        return

    if is_phone_request(user_message):
        status = phone_status()
        if not status.get("connected"):
            await update.message.reply_text("Phone's not connected rn. Make sure MIYA app is open.")
            return

        await update.message.reply_text(f"📱 On it — controlling phone in background. Keep chatting!")
        loop = asyncio.get_event_loop()

        async def _bg_phone():
            try:
                memories = await loop.run_in_executor(None, mem0_recall, user_message, mem0_uid)
                task = f"[Phone control request from {user_name}]{memories}: {user_message}"
                result = await loop.run_in_executor(None, run_phone_agent, task, chat_id)
                loop.run_in_executor(None, mem0_store, user_message, result, mem0_uid)
                for chunk in split_message(f"📱 {result}"):
                    _send_telegram_sync(chat_id, chunk)
            except Exception as e:
                _send_telegram_sync(chat_id, f"📱 Phone task failed: {e}")

        asyncio.create_task(_bg_phone())
        return

    async with lock:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        await update.message.reply_text("🧠 On it...")

        loop = asyncio.get_event_loop()
        memories = await loop.run_in_executor(None, mem0_recall, user_message, mem0_uid)

        prefixed = f"[Message from {user_name}]{memories}: {user_message}"

        response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=chat_id))

        loop.run_in_executor(None, mem0_store, user_message, response, mem0_uid)

        await send_response(update, response)
        logger.info(f"Response sent to {user_name} ({len(response)} chars)")

        while _chat_queues.get(chat_id):
            queued = _chat_queues.pop(chat_id, [])
            combined_texts = [msg for _, _, msg in queued]
            combined = "\n".join(combined_texts)
            q_user = queued[0][0]
            q_mem0 = queued[0][1]
            logger.info(f"Processing {len(queued)} queued messages from {q_user}")

            memories = await loop.run_in_executor(None, mem0_recall, combined, q_mem0)
            prefixed = f"[Follow-up from {q_user} — {len(queued)} messages sent while you were working]{memories}: {combined}"

            response = await loop.run_in_executor(None, lambda: run_claude(prefixed, chat_id=chat_id))
            loop.run_in_executor(None, mem0_store, combined, response, q_mem0)
            await send_response(update, response)
