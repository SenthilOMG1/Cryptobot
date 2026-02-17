#!/usr/bin/env python3
"""
LISTING SNIPER v2.0
====================
Detects new coin listings on Binance and auto-buys on OKX futures.

How it works:
1. Polls Binance announcements API every 30 seconds
2. Monitors OKX for new SWAP instrument listings every 5 minutes
3. When detected, opens LONG on OKX futures (isolated margin)
4. Server-side stop loss on OKX + software trailing stop + auto-sell

v2.0 fixes over v1.0:
- Uses SWAP/futures instead of spot (spot orders were failing)
- Expanded regex patterns (catches 5/5 listing formats vs 1/5)
- Isolated margin (no conflicts with main bot)
- Server-side stop loss on OKX
- Atomic state file writes (crash-safe)
- Actual fill price from order (not stale ticker)
- OKX clients initialized once at startup
- Duplicate snipe protection
- State pruning (prevents unbounded growth)
- Proper exception handling (no bare except)
"""

import os
import re
import json
import time
import logging
import tempfile
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Load env FIRST before anything else
from dotenv import load_dotenv
load_dotenv('/root/Cryptobot/.env')

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
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '7997570468')

SNIPE_AMOUNT_USDT = 5.0       # $5 per listing snipe
LEVERAGE = 3                   # 3x isolated leverage
AUTO_SELL_AFTER_MINUTES = 60   # Sell after 60 min if no trailing stop hit
TRAILING_STOP_PERCENT = 10.0   # 10% trailing stop from peak
TAKE_PROFIT_PERCENT = 50.0     # Take profit at 50% gain
STOP_LOSS_PERCENT = 15.0       # Hard stop loss at -15%
POLL_INTERVAL_SECONDS = 30
MAX_ACTIVE_SNIPES = 3          # Max concurrent snipe positions

STATE_FILE = '/root/Cryptobot/data/sniper_state.json'

# Pairs the main bot trades - don't snipe these
MAIN_BOT_PAIRS = {'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'SUI'}

# OKX API clients (initialized once in main)
_trade_api = None
_acct_api = None
_pub_api = None


# ============================================================
# OKX INITIALIZATION
# ============================================================
def _init_okx():
    """Initialize OKX API clients once."""
    global _trade_api, _acct_api, _pub_api
    import okx.Trade as Trade
    import okx.Account as Account
    import okx.PublicData as PublicData

    api_key = os.environ.get('OKX_API_KEY', '')
    secret = os.environ.get('OKX_SECRET_KEY', '')
    passphrase = os.environ.get('OKX_PASSPHRASE', '')

    _trade_api = Trade.TradeAPI(api_key, secret, passphrase, False, '0')
    _acct_api = Account.AccountAPI(api_key, secret, passphrase, False, '0')
    _pub_api = PublicData.PublicAPI("", "", "", False, "0")
    logger.info("OKX API clients initialized")


# ============================================================
# TELEGRAM NOTIFICATIONS
# ============================================================
def send_telegram(message: str):
    """Send a message to Telegram."""
    try:
        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
            data={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'},
            timeout=10
        )
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


# ============================================================
# STATE MANAGEMENT (atomic writes, corruption recovery)
# ============================================================
def _default_state() -> dict:
    return {
        'seen_announcement_ids': [],
        'seen_okx_instruments': [],
        'active_snipes': [],
        'completed_snipes': [],
        'last_check': None
    }


def load_state() -> dict:
    """Load state with corruption recovery."""
    if not Path(STATE_FILE).exists():
        return _default_state()
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        # Ensure all keys exist
        default = _default_state()
        for key in default:
            if key not in state:
                state[key] = default[key]
        return state
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"State file corrupt: {e}")
        # Rename corrupt file for debugging
        corrupt_path = f"{STATE_FILE}.corrupt.{int(time.time())}"
        try:
            os.rename(STATE_FILE, corrupt_path)
            logger.info(f"Corrupt state moved to {corrupt_path}")
        except Exception:
            pass
        return _default_state()


def save_state(state: dict):
    """Atomic state save via tempfile + os.replace."""
    state_dir = os.path.dirname(STATE_FILE)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix='.tmp')
        with os.fdopen(fd, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp_path, STATE_FILE)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ============================================================
# BINANCE ANNOUNCEMENT SCRAPER
# ============================================================
LISTING_PATTERNS = [
    r'Binance Will List\s+[\w\s]+\((\w+)\)',          # "Binance Will List Espresso (ESP)"
    r'Binance Will Add\s+[\w\s]+\((\w+)\)',            # "Binance Will Add Espresso (ESP) on Earn..."
    r'Binance (?:Will|to) List\s+(\w+)\s',             # "Binance Will List XYZ ..."
    r'New Listing:\s*(\w+)',                             # "New Listing: XYZ"
    r'USDⓈ-Margined\s+(\w+?)USDT\s+Perpetual',        # "Futures Will Launch XYZUSDT Perpetual"
    r'Will Launch.*?(\w+)USDT Perpetual',               # Alternate futures format
]


def check_binance_listings() -> list:
    """Check Binance announcements for new listings."""
    new_coins = []

    try:
        url = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
        params = {
            'type': 1,
            'catalogId': 48,  # New listings category
            'pageNo': 1,
            'pageSize': 20
        }

        resp = requests.get(url, params=params, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0'
        })

        if resp.status_code != 200:
            if resp.status_code in (429, 418):
                logger.warning(f"Binance rate limited ({resp.status_code}), backing off")
                time.sleep(60)
            else:
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

            for pattern in LISTING_PATTERNS:
                match = re.search(pattern, title, re.IGNORECASE)
                if match:
                    symbol = match.group(1).upper()
                    # Skip non-coin matches
                    if len(symbol) < 2 or len(symbol) > 10:
                        continue
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
# OKX INSTRUMENT CHECKER (SWAP/futures)
# ============================================================
def get_okx_swap_instruments() -> dict:
    """Get all tradeable SWAP instruments on OKX. Returns {symbol: info}."""
    instruments = {}

    try:
        resp = requests.get(
            'https://www.okx.com/api/v5/public/instruments',
            params={'instType': 'SWAP'},
            timeout=15
        )

        if resp.status_code == 200:
            data = resp.json()
            for inst in data.get('data', []):
                inst_id = inst['instId']
                if inst_id.endswith('-USDT-SWAP'):
                    base = inst_id.replace('-USDT-SWAP', '')
                    instruments[base] = {
                        'inst_id': inst_id,
                        'ct_val': float(inst.get('ctVal', 1)),
                        'min_sz': float(inst.get('minSz', 1)),
                        'lot_sz': float(inst.get('lotSz', 1)),
                        'state': inst.get('state', 'live')
                    }
    except Exception as e:
        logger.error(f"Error fetching OKX instruments: {e}")

    return instruments


def check_new_okx_listings(state: dict) -> list:
    """Check OKX for newly listed SWAP instruments."""
    new_listings = []

    try:
        instruments = get_okx_swap_instruments()
        current_symbols = set(instruments.keys())
        known_symbols = set(state.get('seen_okx_instruments', []))

        if known_symbols:
            new_symbols = current_symbols - known_symbols
            for sym in new_symbols:
                info = instruments[sym]
                if info['state'] == 'live':
                    new_listings.append({
                        'symbol': sym,
                        'inst_id': info['inst_id'],
                        'source': 'okx_direct'
                    })
                    logger.info(f"New OKX SWAP listing detected: {sym}")

        state['seen_okx_instruments'] = list(current_symbols)

    except Exception as e:
        logger.error(f"Error checking OKX listings: {e}")

    return new_listings


# ============================================================
# OKX FUTURES TRADING
# ============================================================
def _get_available_balance() -> float:
    """Get available USDT balance."""
    try:
        result = _acct_api.get_account_balance(ccy='USDT')
        if result.get('code') == '0' and result['data']:
            for detail in result['data'][0].get('details', []):
                if detail['ccy'] == 'USDT':
                    return float(detail.get('availBal', 0))
    except Exception as e:
        logger.error(f"Balance check failed: {e}")
    return 0


def _check_existing_position(swap_id: str) -> bool:
    """Check if we already have a position on this instrument."""
    try:
        result = _acct_api.get_positions(instId=swap_id)
        if result.get('code') == '0':
            for p in result.get('data', []):
                if abs(float(p.get('pos', 0))) > 0:
                    return True
    except Exception as e:
        logger.error(f"Position check failed: {e}")
    return False


def place_snipe_buy(symbol: str, state: dict) -> dict:
    """
    Open a LONG futures position on OKX for a newly listed coin.
    Uses isolated margin to avoid conflicts with main bot.
    """
    swap_id = f"{symbol}-USDT-SWAP"

    try:
        # Duplicate check
        active_symbols = {s['symbol'] for s in state.get('active_snipes', [])}
        if symbol in active_symbols:
            logger.info(f"Already have active snipe for {symbol}, skipping")
            return {'success': False, 'error': 'Duplicate snipe'}

        # Main bot pair check
        if symbol in MAIN_BOT_PAIRS:
            logger.info(f"{symbol} is a main bot pair, skipping")
            return {'success': False, 'error': 'Main bot pair'}

        # Max active check
        if len(state.get('active_snipes', [])) >= MAX_ACTIVE_SNIPES:
            logger.info(f"Max active snipes ({MAX_ACTIVE_SNIPES}) reached, skipping")
            return {'success': False, 'error': 'Max snipes reached'}

        # Balance check
        balance = _get_available_balance()
        needed = SNIPE_AMOUNT_USDT * 1.1  # 10% buffer for fees
        if balance < needed:
            logger.warning(f"Insufficient balance: ${balance:.2f} < ${needed:.2f}")
            return {'success': False, 'error': f'Insufficient balance: ${balance:.2f}'}

        # Existing position check
        if _check_existing_position(swap_id):
            logger.info(f"Already have position on {swap_id}, skipping")
            return {'success': False, 'error': 'Existing position'}

        # Get instrument info
        inst_resp = _pub_api.get_instruments(instType="SWAP", instId=swap_id)
        if inst_resp.get('code') != '0' or not inst_resp.get('data'):
            logger.warning(f"{swap_id} not available on OKX futures")
            return {'success': False, 'error': 'Instrument not found'}

        inst = inst_resp['data'][0]
        ct_val = float(inst.get('ctVal', 1))
        min_sz = float(inst.get('minSz', 1))
        lot_sz = float(inst.get('lotSz', 1))

        # Get current price for sizing
        ticker_resp = requests.get(
            'https://www.okx.com/api/v5/market/ticker',
            params={'instId': swap_id},
            timeout=10
        )
        ticker_data = ticker_resp.json().get('data', [])
        if not ticker_data:
            return {'success': False, 'error': 'No ticker data'}
        current_price = float(ticker_data[0]['last'])

        # Calculate contract size
        effective_usdt = SNIPE_AMOUNT_USDT * LEVERAGE
        raw_contracts = effective_usdt / (ct_val * current_price)
        if lot_sz > 0:
            num_contracts = round(int(raw_contracts / lot_sz) * lot_sz, 8)
        else:
            num_contracts = int(raw_contracts)

        if num_contracts < min_sz:
            logger.warning(f"{symbol}: calculated {num_contracts} contracts < min {min_sz}")
            return {'success': False, 'error': f'Below min size: {num_contracts} < {min_sz}'}

        # Set leverage (isolated)
        lev_result = _acct_api.set_leverage(
            instId=swap_id, lever=str(LEVERAGE), mgnMode="isolated"
        )
        if lev_result.get('code') != '0':
            logger.error(f"Failed to set leverage for {swap_id}: {lev_result}")
            return {'success': False, 'error': f'Leverage set failed: {lev_result.get("msg", "")}'}

        # Place market buy (open long)
        logger.info(f"SNIPING: Buying {num_contracts} contracts of {swap_id} @ ~${current_price}")
        order_result = _trade_api.place_order(
            instId=swap_id,
            tdMode="isolated",
            side="buy",
            ordType="market",
            sz=str(num_contracts),
            posSide="net"
        )

        if order_result.get('code') != '0':
            error_msg = order_result.get('data', [{}])[0].get('sMsg', '') if order_result.get('data') else order_result.get('msg', '')
            logger.error(f"Snipe order failed: {error_msg}")
            return {'success': False, 'error': f'Order failed: {error_msg}'}

        order_id = order_result['data'][0].get('ordId', '')

        # Get actual fill price
        time.sleep(0.5)
        actual_price = current_price  # fallback
        try:
            order_info = _trade_api.get_order(instId=swap_id, ordId=order_id)
            if order_info.get('code') == '0' and order_info.get('data'):
                fill_px = order_info['data'][0].get('avgPx', '')
                if fill_px:
                    actual_price = float(fill_px)
        except Exception as e:
            logger.warning(f"Couldn't get fill price, using ticker: {e}")

        # Place server-side stop loss
        sl_price = round(actual_price * (1 - STOP_LOSS_PERCENT / 100), 8)
        sl_order_id = ""
        try:
            sl_result = _trade_api.place_algo_order(
                instId=swap_id,
                tdMode="isolated",
                side="sell",
                ordType="conditional",
                sz=str(num_contracts),
                posSide="net",
                slTriggerPx=str(sl_price),
                slOrdPx="-1"  # market price on trigger
            )
            if sl_result.get('code') == '0' and sl_result.get('data'):
                sl_order_id = sl_result['data'][0].get('algoId', '')
                logger.info(f"Server-side SL placed for {symbol} at ${sl_price}")
            else:
                logger.warning(f"SL order failed: {sl_result}")
        except Exception as e:
            logger.warning(f"Failed to place SL: {e}")

        logger.info(f"SNIPE SUCCESS: {symbol} | {num_contracts} contracts @ ${actual_price:.6f}")

        return {
            'success': True,
            'order_id': order_id,
            'sl_order_id': sl_order_id,
            'symbol': symbol,
            'inst_id': swap_id,
            'num_contracts': num_contracts,
            'entry_price': actual_price,
            'peak_price': actual_price,
            'amount_usdt': SNIPE_AMOUNT_USDT,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Snipe buy failed for {symbol}: {e}")
        return {'success': False, 'error': str(e)}


def check_snipe_exit(snipe: dict) -> str:
    """Check if an active snipe should be closed. Returns exit reason or 'hold'."""
    try:
        swap_id = snipe['inst_id']
        entry_time = datetime.fromisoformat(snipe['timestamp'])
        entry_price = snipe.get('entry_price', 0)

        if entry_price <= 0:
            return 'hold'

        # Get current price
        resp = requests.get(
            'https://www.okx.com/api/v5/market/ticker',
            params={'instId': swap_id},
            timeout=10
        )
        if resp.status_code != 200:
            return 'hold'

        data = resp.json().get('data', [])
        if not data:
            return 'hold'

        current_price = float(data[0]['last'])
        peak_price = snipe.get('peak_price', entry_price)

        # Update peak
        if current_price > peak_price:
            snipe['peak_price'] = current_price
            peak_price = current_price

        snipe['current_price'] = current_price
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        drawdown_from_peak = ((peak_price - current_price) / peak_price) * 100 if peak_price > 0 else 0
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

        # Note: hard stop loss is handled server-side on OKX
        # But check in case server SL wasn't placed
        if pnl_pct <= -(STOP_LOSS_PERCENT + 2):  # 2% buffer past SL
            return 'stop_loss'

        return 'hold'

    except Exception as e:
        logger.error(f"Error checking snipe exit: {e}")
        return 'hold'


def execute_snipe_close(snipe: dict, reason: str) -> bool:
    """Close a sniped futures position."""
    swap_id = snipe['inst_id']
    symbol = snipe['symbol']

    try:
        # Cancel server-side SL first
        sl_order_id = snipe.get('sl_order_id', '')
        if sl_order_id:
            try:
                _trade_api.cancel_algo_order([{
                    'instId': swap_id,
                    'algoId': sl_order_id
                }])
            except Exception as e:
                logger.warning(f"Failed to cancel SL for {symbol}: {e}")

        # Get actual position size from exchange
        pos_result = _acct_api.get_positions(instId=swap_id)
        actual_sz = 0
        if pos_result.get('code') == '0':
            for p in pos_result.get('data', []):
                pos_val = float(p.get('pos', 0))
                if pos_val > 0:  # Long position
                    actual_sz = pos_val
                    break

        if actual_sz <= 0:
            logger.warning(f"No position found for {swap_id}, may have been stopped out")
            return True  # Position already closed (SL hit)

        # Close position (sell to close long)
        result = _trade_api.place_order(
            instId=swap_id,
            tdMode="isolated",
            side="sell",
            ordType="market",
            sz=str(actual_sz),
            posSide="net"
        )

        if result.get('code') == '0':
            pnl_pct = snipe.get('pnl_pct', 0)
            logger.info(f"SNIPE CLOSED: {symbol} | Reason: {reason} | P&L: {pnl_pct:+.1f}%")
            send_telegram(
                f"{'🟢' if pnl_pct > 0 else '🔴'} <b>Snipe Exit: {symbol}</b>\n"
                f"Reason: {reason}\n"
                f"P&L: {pnl_pct:+.1f}%\n"
                f"Entry: ${snipe.get('entry_price', 0):.6f}\n"
                f"Exit: ${snipe.get('current_price', 0):.6f}"
            )
            return True
        else:
            error_msg = result.get('data', [{}])[0].get('sMsg', '') if result.get('data') else result.get('msg', '')
            logger.error(f"Close failed for {symbol}: {error_msg}")
            return False

    except Exception as e:
        logger.error(f"Snipe close failed for {symbol}: {e}")
        return False


def _reconcile_positions(state: dict):
    """On startup, remove active snipes that have no matching exchange position."""
    active = state.get('active_snipes', [])
    if not active:
        return

    to_remove = []
    for snipe in active:
        swap_id = snipe.get('inst_id', '')
        if not swap_id:
            to_remove.append(snipe)
            continue
        try:
            pos = _acct_api.get_positions(instId=swap_id)
            has_pos = False
            if pos.get('code') == '0':
                for p in pos.get('data', []):
                    if float(p.get('pos', 0)) > 0:
                        has_pos = True
                        break
            if not has_pos:
                logger.warning(f"Orphan snipe {snipe.get('symbol', '?')}: no position on exchange, removing")
                to_remove.append(snipe)
        except Exception as e:
            logger.warning(f"Reconcile check failed for {swap_id}: {e}")

    for s in to_remove:
        s['exit_reason'] = 'orphan_reconciled'
        s['exit_time'] = datetime.now().isoformat()
        state.setdefault('completed_snipes', []).append(s)
        active.remove(s)


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    """Main listing sniper loop."""
    logger.info("=" * 50)
    logger.info("LISTING SNIPER v2.0 STARTING")
    logger.info("=" * 50)
    logger.info(f"Snipe amount: ${SNIPE_AMOUNT_USDT} @ {LEVERAGE}x ISOLATED leverage")
    logger.info(f"Max concurrent: {MAX_ACTIVE_SNIPES}")
    logger.info(f"Auto-sell after: {AUTO_SELL_AFTER_MINUTES} min")
    logger.info(f"Trailing stop: {TRAILING_STOP_PERCENT}% | SL: {STOP_LOSS_PERCENT}% | TP: {TAKE_PROFIT_PERCENT}%")
    logger.info(f"Poll interval: {POLL_INTERVAL_SECONDS}s")
    logger.info(f"Excluded pairs (main bot): {MAIN_BOT_PAIRS}")

    # Initialize OKX
    _init_okx()

    send_telegram(
        "🎯 <b>Listing Sniper v2.0 ACTIVE</b>\n"
        f"Watching Binance + OKX for new listings\n"
        f"Snipe: ${SNIPE_AMOUNT_USDT} @ {LEVERAGE}x isolated\n"
        f"SL: {STOP_LOSS_PERCENT}% | TP: {TAKE_PROFIT_PERCENT}% | Trail: {TRAILING_STOP_PERCENT}%"
    )

    # Load state
    state = load_state()

    # Reconcile orphan positions
    _reconcile_positions(state)

    # Initialize OKX SWAP instrument baseline on first run
    if not state.get('seen_okx_instruments'):
        logger.info("Building OKX SWAP instrument baseline (first run)...")
        instruments = get_okx_swap_instruments()
        state['seen_okx_instruments'] = list(instruments.keys())
        logger.info(f"Baseline: {len(instruments)} SWAP instruments tracked")
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
            seen_ids = set(state['seen_announcement_ids'])

            for listing in listings:
                if listing['article_id'] not in seen_ids:
                    symbol = listing['symbol']
                    logger.info(f"!!! NEW BINANCE LISTING: {symbol} !!!")
                    send_telegram(
                        f"🚨 <b>NEW BINANCE LISTING DETECTED!</b>\n"
                        f"Coin: {symbol}\n"
                        f"Title: {listing['title']}\n"
                        f"Checking OKX futures..."
                    )

                    state['seen_announcement_ids'].append(listing['article_id'])

                    # Skip main bot pairs
                    if symbol in MAIN_BOT_PAIRS:
                        send_telegram(f"⏭ {symbol} is a main bot pair, skipping snipe")
                        continue

                    # Check if tradeable on OKX futures
                    okx_instruments = get_okx_swap_instruments()
                    if symbol in okx_instruments and okx_instruments[symbol]['state'] == 'live':
                        result = place_snipe_buy(symbol, state)
                        if result.get('success'):
                            state['active_snipes'].append(result)
                            send_telegram(
                                f"🎯 <b>SNIPED!</b> {symbol}\n"
                                f"Contracts: {result['num_contracts']}\n"
                                f"Entry: ${result['entry_price']:.6f}\n"
                                f"SL: {STOP_LOSS_PERCENT}% | TP: {TAKE_PROFIT_PERCENT}%\n"
                                f"Max hold: {AUTO_SELL_AFTER_MINUTES}min"
                            )
                        else:
                            error = result.get('error', 'Unknown')
                            if error not in ('Duplicate snipe', 'Main bot pair'):
                                send_telegram(f"❌ Snipe failed for {symbol}: {error}")
                    else:
                        send_telegram(f"❌ {symbol} not available on OKX futures (yet)")

            # ---- CHECK FOR NEW OKX SWAP LISTINGS (direct) ----
            if cycle % 10 == 0:  # Every ~5 min
                new_okx = check_new_okx_listings(state)
                for listing in new_okx:
                    symbol = listing['symbol']
                    if symbol in MAIN_BOT_PAIRS:
                        continue

                    send_telegram(
                        f"🆕 <b>New OKX futures listing:</b> {symbol}\n"
                        f"Pair: {listing['inst_id']}\n"
                        f"Attempting snipe..."
                    )

                    result = place_snipe_buy(symbol, state)
                    if result.get('success'):
                        state['active_snipes'].append(result)
                        send_telegram(
                            f"🎯 <b>SNIPED new listing!</b> {symbol}\n"
                            f"Entry: ${result['entry_price']:.6f}"
                        )

            # ---- MANAGE ACTIVE SNIPES ----
            closed_order_ids = []
            for snipe in list(state.get('active_snipes', [])):
                action = check_snipe_exit(snipe)

                if action != 'hold':
                    success = execute_snipe_close(snipe, action)
                    if success:
                        snipe['exit_reason'] = action
                        snipe['exit_time'] = datetime.now().isoformat()
                        state['completed_snipes'].append(snipe)
                        closed_order_ids.append(snipe.get('order_id'))

            # Remove closed snipes by order_id (safe, no list-during-iterate issues)
            if closed_order_ids:
                state['active_snipes'] = [
                    s for s in state['active_snipes']
                    if s.get('order_id') not in closed_order_ids
                ]

            # ---- PRUNE STATE ----
            if len(state['seen_announcement_ids']) > 500:
                state['seen_announcement_ids'] = state['seen_announcement_ids'][-500:]
            if len(state['completed_snipes']) > 100:
                state['completed_snipes'] = state['completed_snipes'][-100:]

            # ---- SAVE STATE ----
            state['last_check'] = datetime.now().isoformat()
            save_state(state)

            # Log status periodically
            if cycle % 60 == 0:  # Every ~30 min
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
            logger.error(f"Sniper error: {e}", exc_info=True)
            time.sleep(60)


if __name__ == '__main__':
    Path('/root/Cryptobot/data').mkdir(exist_ok=True)
    main()
