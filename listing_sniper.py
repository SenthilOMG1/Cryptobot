#!/usr/bin/env python3
"""
LISTING SNIPER v1.0
====================
Detects new coin listings on Binance and auto-buys on OKX.

How it works:
1. Polls Binance announcements API every 30 seconds
2. When a new listing is detected, checks if it's tradeable on OKX
3. If tradeable → instant market buy
4. Auto-sells after configurable time or on trailing stop

Runs as a standalone service alongside the main trading bot.
"""

import os
import re
import json
import time
import logging
import hashlib
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SNIPER] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/root/Cryptobot/data/sniper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('sniper')

# ============================================================
# CONFIGURATION
# ============================================================
TELEGRAM_BOT_TOKEN = "8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY"
TELEGRAM_CHAT_ID = "7997570468"

# How much USDT to spend per snipe
SNIPE_AMOUNT_USDT = 5.0  # $5 per listing snipe

# Auto-sell settings
AUTO_SELL_AFTER_MINUTES = 60  # Sell after 60 min if no trailing stop hit
TRAILING_STOP_PERCENT = 10.0  # 10% trailing stop from peak
TAKE_PROFIT_PERCENT = 50.0    # Take profit at 50% gain

# Poll interval
POLL_INTERVAL_SECONDS = 30

# State file to track seen announcements
STATE_FILE = '/root/Cryptobot/data/sniper_state.json'

# OKX API credentials from .env
OKX_API_KEY = os.getenv('OKX_API_KEY', '')
OKX_SECRET_KEY = os.getenv('OKX_SECRET_KEY', '')
OKX_PASSPHRASE = os.getenv('OKX_PASSPHRASE', '')


# ============================================================
# TELEGRAM NOTIFICATIONS
# ============================================================
def send_telegram(message: str):
    """Send a message to Telegram."""
    try:
        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
            data={'chat_id': TELEGRAM_CHAT_ID, 'text': message},
            timeout=10
        )
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


# ============================================================
# STATE MANAGEMENT
# ============================================================
def load_state() -> dict:
    """Load seen announcements and active snipes."""
    if Path(STATE_FILE).exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'seen_announcement_ids': [],
        'seen_okx_instruments': [],
        'active_snipes': [],
        'completed_snipes': [],
        'last_check': None
    }


def save_state(state: dict):
    """Save state to disk."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


# ============================================================
# BINANCE ANNOUNCEMENT SCRAPER
# ============================================================
def check_binance_listings() -> list:
    """
    Check Binance announcements for new listings.
    Returns list of coin symbols from new listing announcements.
    """
    new_coins = []

    try:
        # Binance announcement API
        url = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
        params = {
            'type': 1,
            'catalogId': 48,  # New listings category
            'pageNo': 1,
            'pageSize': 10
        }

        resp = requests.get(url, params=params, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })

        if resp.status_code != 200:
            logger.warning(f"Binance API returned {resp.status_code}")
            return []

        data = resp.json()
        articles = data.get('data', {}).get('catalogs', [{}])

        if articles:
            articles = articles[0].get('articles', [])

        if not articles:
            articles = data.get('data', {}).get('articles', [])

        for article in articles:
            title = article.get('title', '')
            article_id = str(article.get('id', article.get('code', '')))

            # Look for "Binance Will List" pattern
            # Examples:
            #   "Binance Will List Espresso (ESP)"
            #   "Binance Will List XYZ (XYZ) with Seed Tag Applied"
            patterns = [
                r'Binance Will List\s+[\w\s]+\((\w+)\)',
                r'Binance (?:Will|to) List\s+(\w+)\s',
                r'New Listing:\s*(\w+)',
            ]

            for pattern in patterns:
                match = re.search(pattern, title, re.IGNORECASE)
                if match:
                    symbol = match.group(1).upper()
                    new_coins.append({
                        'symbol': symbol,
                        'title': title,
                        'article_id': article_id,
                        'timestamp': article.get('releaseDate', int(time.time() * 1000))
                    })
                    break

    except Exception as e:
        logger.error(f"Error checking Binance listings: {e}")

    return new_coins


# ============================================================
# OKX INSTRUMENT CHECKER
# ============================================================
def get_okx_instruments() -> dict:
    """Get all tradeable instruments on OKX. Returns {symbol: inst_id}."""
    instruments = {}

    try:
        # Check spot instruments
        resp = requests.get(
            'https://www.okx.com/api/v5/public/instruments',
            params={'instType': 'SPOT'},
            timeout=15
        )

        if resp.status_code == 200:
            data = resp.json()
            for inst in data.get('data', []):
                inst_id = inst['instId']
                base = inst_id.split('-')[0]
                if inst_id.endswith('-USDT'):
                    instruments[base] = {
                        'inst_id': inst_id,
                        'min_size': inst.get('minSz', '0'),
                        'lot_size': inst.get('lotSz', '0'),
                        'list_time': inst.get('listTime', '0'),
                        'state': inst.get('state', 'live')
                    }
    except Exception as e:
        logger.error(f"Error fetching OKX instruments: {e}")

    return instruments


def check_new_okx_listings(state: dict) -> list:
    """
    Check OKX for newly listed instruments that weren't there before.
    This catches listings even without Binance announcements.
    """
    new_listings = []

    try:
        instruments = get_okx_instruments()
        current_symbols = set(instruments.keys())
        known_symbols = set(state.get('seen_okx_instruments', []))

        if known_symbols:  # Only detect new ones if we have a baseline
            new_symbols = current_symbols - known_symbols
            for sym in new_symbols:
                info = instruments[sym]
                if info['state'] == 'live':
                    new_listings.append({
                        'symbol': sym,
                        'inst_id': info['inst_id'],
                        'source': 'okx_direct'
                    })
                    logger.info(f"New OKX listing detected: {sym}")

        # Update baseline
        state['seen_okx_instruments'] = list(current_symbols)

    except Exception as e:
        logger.error(f"Error checking OKX listings: {e}")

    return new_listings


# ============================================================
# OKX TRADING (using direct API for speed)
# ============================================================
def place_snipe_buy(symbol: str, usdt_amount: float) -> dict:
    """
    Place an instant market buy on OKX for a newly listed coin.
    Uses the bot's existing OKX client for authenticated trading.
    """
    try:
        # Import and use the bot's OKX client
        import sys
        sys.path.insert(0, '/root/Cryptobot')
        from dotenv import load_dotenv
        load_dotenv('/root/Cryptobot/.env')

        from src.security.vault import SecureVault
        from src.trading.okx_client import SecureOKXClient

        vault = SecureVault()
        client = SecureOKXClient(vault, demo_mode=False)

        inst_id = f"{symbol}-USDT"

        # Check if instrument exists and get min size
        try:
            info = client.get_instrument_info(inst_id)
            min_size = float(info.get('minSz', 0))
            logger.info(f"Instrument {inst_id}: minSz={min_size}")
        except Exception as e:
            logger.warning(f"{inst_id} not found on OKX spot: {e}")
            return {'success': False, 'error': f'Instrument not found: {e}'}

        # Check balance
        balance = client.get_usdt_balance()
        if balance < usdt_amount:
            logger.warning(f"Insufficient balance: ${balance:.2f} < ${usdt_amount:.2f}")
            return {'success': False, 'error': f'Insufficient balance: ${balance:.2f}'}

        # Place market buy
        logger.info(f"SNIPING: Buying ${usdt_amount} of {inst_id}")
        result = client.place_market_buy(inst_id, usdt_amount)

        if result:
            logger.info(f"SNIPE SUCCESS: {result}")
            return {
                'success': True,
                'order_id': result.get('ordId', ''),
                'symbol': symbol,
                'inst_id': inst_id,
                'amount_usdt': usdt_amount,
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {'success': False, 'error': 'Order returned empty result'}

    except Exception as e:
        logger.error(f"Snipe buy failed for {symbol}: {e}")
        return {'success': False, 'error': str(e)}


def check_snipe_exit(snipe: dict) -> str:
    """
    Check if an active snipe should be sold.
    Returns: 'hold', 'take_profit', 'trailing_stop', 'time_exit'
    """
    try:
        symbol = snipe['symbol']
        inst_id = f"{symbol}-USDT"
        entry_time = datetime.fromisoformat(snipe['timestamp'])

        # Get current price
        resp = requests.get(
            'https://www.okx.com/api/v5/market/ticker',
            params={'instId': inst_id},
            timeout=10
        )

        if resp.status_code != 200:
            return 'hold'

        data = resp.json()
        tickers = data.get('data', [])
        if not tickers:
            return 'hold'

        current_price = float(tickers[0]['last'])
        entry_price = snipe.get('entry_price', current_price)
        peak_price = snipe.get('peak_price', entry_price)

        # Update peak
        if current_price > peak_price:
            snipe['peak_price'] = current_price
            peak_price = current_price

        # Update current price
        snipe['current_price'] = current_price

        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        drawdown_from_peak = ((peak_price - current_price) / peak_price) * 100

        snipe['pnl_pct'] = pnl_pct

        # Take profit
        if pnl_pct >= TAKE_PROFIT_PERCENT:
            return 'take_profit'

        # Trailing stop (only if we're up at least 5%)
        if pnl_pct > 5.0 and drawdown_from_peak >= TRAILING_STOP_PERCENT:
            return 'trailing_stop'

        # Time exit
        if datetime.now() - entry_time > timedelta(minutes=AUTO_SELL_AFTER_MINUTES):
            return 'time_exit'

        # Hard stop loss at -20%
        if pnl_pct <= -20.0:
            return 'stop_loss'

        return 'hold'

    except Exception as e:
        logger.error(f"Error checking snipe exit: {e}")
        return 'hold'


def execute_snipe_sell(snipe: dict, reason: str) -> bool:
    """Sell a sniped position."""
    try:
        import sys
        sys.path.insert(0, '/root/Cryptobot')
        from dotenv import load_dotenv
        load_dotenv('/root/Cryptobot/.env')

        from src.security.vault import SecureVault
        from src.trading.okx_client import SecureOKXClient

        vault = SecureVault()
        client = SecureOKXClient(vault, demo_mode=False)

        inst_id = f"{snipe['symbol']}-USDT"
        amount = snipe.get('amount_coins', 0)

        if amount <= 0:
            # Try to get position from order
            try:
                order = client.get_order(inst_id, snipe.get('order_id', ''))
                amount = float(order.get('fillSz', 0))
            except:
                logger.error(f"Can't determine position size for {inst_id}")
                return False

        result = client.place_market_sell(inst_id, amount)

        if result:
            pnl_pct = snipe.get('pnl_pct', 0)
            logger.info(f"SNIPE SOLD: {inst_id} | Reason: {reason} | P&L: {pnl_pct:+.1f}%")
            send_telegram(
                f"{'🟢' if pnl_pct > 0 else '🔴'} Snipe Exit: {snipe['symbol']}\n"
                f"Reason: {reason}\n"
                f"P&L: {pnl_pct:+.1f}%\n"
                f"Entry: ${snipe.get('entry_price', 0):.6f}\n"
                f"Exit: ${snipe.get('current_price', 0):.6f}"
            )
            return True

        return False

    except Exception as e:
        logger.error(f"Snipe sell failed: {e}")
        return False


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    """Main listing sniper loop."""
    logger.info("=" * 50)
    logger.info("LISTING SNIPER v1.0 STARTING")
    logger.info("=" * 50)
    logger.info(f"Snipe amount: ${SNIPE_AMOUNT_USDT}")
    logger.info(f"Auto-sell after: {AUTO_SELL_AFTER_MINUTES} min")
    logger.info(f"Trailing stop: {TRAILING_STOP_PERCENT}%")
    logger.info(f"Take profit: {TAKE_PROFIT_PERCENT}%")
    logger.info(f"Poll interval: {POLL_INTERVAL_SECONDS}s")

    send_telegram(
        "🎯 Listing Sniper v1.0 ACTIVE\n"
        f"Watching for new listings on Binance + OKX\n"
        f"Snipe size: ${SNIPE_AMOUNT_USDT}\n"
        f"Auto-sell: {AUTO_SELL_AFTER_MINUTES}min / {TRAILING_STOP_PERCENT}% trailing stop"
    )

    # Load state
    state = load_state()

    # Initialize OKX instrument baseline on first run
    if not state.get('seen_okx_instruments'):
        logger.info("Building OKX instrument baseline (first run)...")
        instruments = get_okx_instruments()
        state['seen_okx_instruments'] = list(instruments.keys())
        logger.info(f"Baseline: {len(instruments)} instruments tracked")
        save_state(state)

    # Initialize seen Binance announcement IDs
    if not state.get('seen_announcement_ids'):
        logger.info("Building Binance announcement baseline...")
        existing = check_binance_listings()
        state['seen_announcement_ids'] = [a['article_id'] for a in existing]
        logger.info(f"Baseline: {len(existing)} existing announcements tracked")
        save_state(state)

    cycle = 0

    while True:
        try:
            cycle += 1

            # ---- CHECK FOR NEW BINANCE LISTINGS ----
            listings = check_binance_listings()
            for listing in listings:
                if listing['article_id'] not in state['seen_announcement_ids']:
                    # NEW LISTING DETECTED!
                    symbol = listing['symbol']
                    logger.info(f"!!! NEW BINANCE LISTING: {symbol} !!!")
                    send_telegram(
                        f"🚨 NEW BINANCE LISTING DETECTED!\n"
                        f"Coin: {symbol}\n"
                        f"Title: {listing['title']}\n"
                        f"Checking OKX availability..."
                    )

                    # Check if tradeable on OKX
                    okx_instruments = get_okx_instruments()
                    if symbol in okx_instruments:
                        info = okx_instruments[symbol]
                        if info['state'] == 'live':
                            # SNIPE IT!
                            result = place_snipe_buy(symbol, SNIPE_AMOUNT_USDT)
                            if result.get('success'):
                                # Get entry price
                                try:
                                    resp = requests.get(
                                        'https://www.okx.com/api/v5/market/ticker',
                                        params={'instId': f'{symbol}-USDT'},
                                        timeout=10
                                    )
                                    price_data = resp.json().get('data', [{}])
                                    entry_price = float(price_data[0].get('last', 0)) if price_data else 0
                                except:
                                    entry_price = 0

                                snipe_record = {
                                    **result,
                                    'entry_price': entry_price,
                                    'peak_price': entry_price,
                                    'source': 'binance_listing'
                                }
                                state['active_snipes'].append(snipe_record)

                                send_telegram(
                                    f"🎯 SNIPED! Bought ${SNIPE_AMOUNT_USDT} of {symbol}\n"
                                    f"Entry: ${entry_price:.6f}\n"
                                    f"Trailing stop: {TRAILING_STOP_PERCENT}%\n"
                                    f"Take profit: {TAKE_PROFIT_PERCENT}%\n"
                                    f"Max hold: {AUTO_SELL_AFTER_MINUTES}min"
                                )
                            else:
                                send_telegram(
                                    f"❌ Snipe failed for {symbol}: {result.get('error', 'Unknown')}"
                                )
                        else:
                            send_telegram(f"⏳ {symbol} on OKX but not live yet (state: {info['state']})")
                    else:
                        send_telegram(f"❌ {symbol} not available on OKX (yet)")

                    state['seen_announcement_ids'].append(listing['article_id'])

            # ---- CHECK FOR NEW OKX LISTINGS (direct) ----
            if cycle % 10 == 0:  # Check every 10 cycles (5 min)
                new_okx = check_new_okx_listings(state)
                for listing in new_okx:
                    symbol = listing['symbol']
                    send_telegram(
                        f"🆕 New OKX listing detected: {symbol}\n"
                        f"Pair: {listing['inst_id']}\n"
                        f"Attempting snipe..."
                    )

                    result = place_snipe_buy(symbol, SNIPE_AMOUNT_USDT)
                    if result.get('success'):
                        try:
                            resp = requests.get(
                                'https://www.okx.com/api/v5/market/ticker',
                                params={'instId': f'{symbol}-USDT'},
                                timeout=10
                            )
                            price_data = resp.json().get('data', [{}])
                            entry_price = float(price_data[0].get('last', 0)) if price_data else 0
                        except:
                            entry_price = 0

                        snipe_record = {
                            **result,
                            'entry_price': entry_price,
                            'peak_price': entry_price,
                            'source': 'okx_direct'
                        }
                        state['active_snipes'].append(snipe_record)

                        send_telegram(
                            f"🎯 SNIPED new OKX listing! ${SNIPE_AMOUNT_USDT} of {symbol}\n"
                            f"Entry: ${entry_price:.6f}"
                        )

            # ---- MANAGE ACTIVE SNIPES ----
            for snipe in list(state.get('active_snipes', [])):
                action = check_snipe_exit(snipe)

                if action != 'hold':
                    success = execute_snipe_sell(snipe, action)
                    if success:
                        snipe['exit_reason'] = action
                        snipe['exit_time'] = datetime.now().isoformat()
                        state['completed_snipes'].append(snipe)
                        state['active_snipes'].remove(snipe)

            # ---- SAVE STATE ----
            state['last_check'] = datetime.now().isoformat()
            save_state(state)

            # Log status periodically
            if cycle % 60 == 0:  # Every 30 min
                active = len(state.get('active_snipes', []))
                completed = len(state.get('completed_snipes', []))
                logger.info(f"Sniper status: {active} active, {completed} completed, cycle {cycle}")

            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Sniper stopped by user")
            send_telegram("🛑 Listing Sniper stopped")
            save_state(state)
            break
        except Exception as e:
            logger.error(f"Sniper error: {e}")
            time.sleep(60)  # Wait a minute on error


if __name__ == '__main__':
    # Load env
    from dotenv import load_dotenv
    load_dotenv('/root/Cryptobot/.env')

    # Ensure data dir exists
    Path('/root/Cryptobot/data').mkdir(exist_ok=True)

    main()
