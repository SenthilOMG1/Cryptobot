"""
Telegram Notifications
======================
Send proactive messages to the user via Telegram.
Used by the trading bot for alerts and by cron for daily briefings.
"""

import logging
import requests

logger = logging.getLogger(__name__)

BOT_TOKEN = "8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY"
CHAT_ID = 7997570468
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


def send_message(text: str, silent: bool = False) -> bool:
    """Send a message to the user via Telegram."""
    try:
        resp = requests.post(
            f"{API_URL}/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": text,
                "disable_notification": silent
            },
            timeout=10
        )
        if resp.json().get("ok"):
            return True
        else:
            logger.error(f"Telegram send failed: {resp.json()}")
            return False
    except Exception as e:
        logger.error(f"Telegram notification error: {e}")
        return False


def notify_trade(pair: str, action: str, direction: str, leverage: int,
                 confidence: float, price: float, size_usd: float):
    """Notify when a trade is executed."""
    icon = "🟢" if action == "BUY" or direction == "long" else "🔴"
    send_message(
        f"{icon} TRADE EXECUTED\n\n"
        f"Pair: {pair}\n"
        f"Direction: {direction.upper()}\n"
        f"Leverage: {leverage}x\n"
        f"Confidence: {confidence:.1%}\n"
        f"Price: ${price:.4f}\n"
        f"Size: ${size_usd:.2f}"
    )


def notify_close(pair: str, direction: str, pnl_pct: float, pnl_usd: float, reason: str):
    """Notify when a position is closed."""
    icon = "💰" if pnl_pct >= 0 else "💸"
    send_message(
        f"{icon} POSITION CLOSED\n\n"
        f"Pair: {pair}\n"
        f"Direction: {direction}\n"
        f"P&L: {pnl_pct:+.2f}% (${pnl_usd:+.4f})\n"
        f"Reason: {reason}"
    )


def notify_stoploss(pair: str, direction: str, pnl_pct: float):
    """Notify when stop-loss is triggered."""
    send_message(
        f"🛑 STOP-LOSS HIT\n\n"
        f"Pair: {pair}\n"
        f"Direction: {direction}\n"
        f"Loss: {pnl_pct:.2f}%"
    )


def notify_takeprofit(pair: str, direction: str, pnl_pct: float):
    """Notify when take-profit is hit."""
    send_message(
        f"🎯 TAKE-PROFIT HIT\n\n"
        f"Pair: {pair}\n"
        f"Direction: {direction}\n"
        f"Profit: +{pnl_pct:.2f}%"
    )


def notify_alert(title: str, message: str):
    """Send a general alert."""
    send_message(f"⚠️ {title}\n\n{message}")


def notify_daily_limit():
    """Notify when daily loss limit is reached."""
    send_message(
        "🚨 DAILY LOSS LIMIT REACHED\n\n"
        "Trading paused until next cycle.\n"
        "All positions monitored, no new trades."
    )
