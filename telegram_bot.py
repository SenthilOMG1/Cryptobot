"""
Telegram Bot Bridge to Claude Code — Entry Point
=================================================
Full-featured bridge: voice messages, session management,
quick commands, and persistent Claude Code sessions.
"""

import logging

from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters,
)

from tg.config import BOT_TOKEN, OWNER_ID
from tg.commands import (
    cmd_start, cmd_status, cmd_positions, cmd_balance, cmd_logs,
    cmd_sessions, cmd_resume, cmd_new, cmd_current, cmd_steer,
    cmd_restart, cmd_stop, cmd_train, cmd_pnl, cmd_market,
    cmd_trades, cmd_config, cmd_leverage, cmd_close,
    cmd_pause, cmd_unpause, cmd_services, cmd_phone, cmd_memory,
)
from tg.media import (
    handle_callback, handle_voice, handle_photo,
    handle_video, handle_document, handle_message,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


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
