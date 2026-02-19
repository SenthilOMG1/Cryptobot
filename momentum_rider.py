#!/usr/bin/env python3
"""
MOMENTUM RIDER v2.0
====================
Autonomous momentum scanner + auto-trader.

v2.0 STRATEGY: Buy the DIP of the pump, not the top.

Old (v1): detect pump → buy immediately → bought the top → lost money
New (v2): detect pump → track it → wait for pullback → confirm with volume + green candle → buy

Key improvements over v1.1:
1. DIP ENTRY: After detecting a pump, waits for a 2-3% pullback from high
2. VOLUME CONFIRM: Requires recent volume >2x average (filters fake pumps)
3. CANDLE CONFIRM: Waits for a green 5min candle (pump still alive after dip)
4. PAPER MODE: Can run without placing real orders to validate strategy

All v1.1 infrastructure preserved:
- ISOLATED margin, server-side SL, atomic state, fill price, balance checks
"""

import os
import json
import time
import logging
import tempfile
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MOMENTUM] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/root/Cryptobot/data/momentum.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('momentum')

# ============================================================
# CONFIGURATION
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8382776877:AAH8R-Tt0rUU_tHGFN3dTPnatesmnpjcdFY')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '7997570468')

# Detection thresholds
PUMP_THRESHOLD_24H = 8.0     # Minimum 8% gain in 24h to consider
PUMP_THRESHOLD_SHORT = 5.0   # Or 5% in short window (15-20min)
MIN_VOLUME_USDT = 100000     # Minimum 24h volume to avoid illiquid coins
MAX_CONCURRENT_RIDES = 3     # Max simultaneous momentum trades
RIDE_AMOUNT_USDT = 3.0       # $3 per momentum trade
MIN_BALANCE_USDT = 5.0       # Don't trade if balance below this

# v2: Dip entry parameters
DIP_ENTRY_PCT = 2.5          # Wait for 2.5% pullback from detected high before buying
DIP_MAX_WAIT_MIN = 30        # Max time to wait for dip (abandon if no pullback)
VOLUME_SPIKE_RATIO = 1.5     # Recent volume must be >1.5x average
GREEN_CANDLE_REQUIRED = True # Require a green 5min candle before entry

# Paper mode: set to True to run without real trades (validation mode)
PAPER_MODE = True

# Exit settings
TRAILING_STOP_PCT = 3.0      # 3% trailing stop from peak (tighter)
TRAILING_ACTIVATE_PCT = 2.0  # Activate trailing stop after 2% gain
STOP_LOSS_PCT = 5.0          # Hard stop at -5% price (= -15% on margin at 3x)
MAX_HOLD_MINUTES = 60        # Auto-sell after 60 min
TAKE_PROFIT_PCT = 15.0       # Take profit at 15%

# Scan interval
SCAN_INTERVAL_SECONDS = 120  # Scan every 2 minutes (faster to catch dips)

# Cooldown: don't re-enter same coin within 2 hours
COOLDOWN_MINUTES = 120

# State file
STATE_FILE = '/root/Cryptobot/data/momentum_state.json'
MAX_COMPLETED_RIDES = 100    # Keep last N completed rides

# Blacklist: don't trade stablecoins, wrapped tokens, or main bot pairs
BLACKLIST = {'USDC', 'USDT', 'DAI', 'TUSD', 'BUSD', 'FDUSD', 'WBTC', 'WETH',
             'STETH', 'RETH', 'CBETH', 'PYUSD', 'USDP', 'GUSD', 'FRAX'}

# Main bot pairs - NEVER trade these to avoid position conflicts
MAIN_BOT_PAIRS = {'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'SUI'}

# OKX API clients (initialized once at startup)
_trade_api = None
_acct_api = None
_pub_api = None


def _init_okx():
    """Initialize OKX API clients once."""
    global _trade_api, _acct_api, _pub_api
    if _trade_api is not None:
        return

    import okx.Trade as Trade
    import okx.Account as Account
    import okx.PublicData as PublicData

    api_key = os.environ.get('OKX_API_KEY', '')
    secret = os.environ.get('OKX_SECRET_KEY', '')
    passphrase = os.environ.get('OKX_PASSPHRASE', '')

    _trade_api = Trade.TradeAPI(api_key, secret, passphrase, False, '0')
    _acct_api = Account.AccountAPI(api_key, secret, passphrase, False, '0')
    _pub_api = PublicData.PublicAPI("", "", "", False, "0")


# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message: str):
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
def _fresh_state() -> dict:
    return {
        'active_rides': [],
        'watchlist': [],       # v2: pumps waiting for dip entry
        'completed_rides': [],
        'cooldowns': {},
        'price_cache': {},
        'paper_trades': [],    # v2: paper mode tracked trades
        'stats': {'total_trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0}
    }


def load_state() -> dict:
    if Path(STATE_FILE).exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Corrupt state file, starting fresh: {e}")
            corrupt_path = STATE_FILE + f'.corrupt.{int(time.time())}'
            os.rename(STATE_FILE, corrupt_path)
    return _fresh_state()


def save_state(state: dict):
    """Atomic state save - write to temp file then rename."""
    try:
        state_dir = os.path.dirname(STATE_FILE)
        fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix='.tmp')
        with os.fdopen(fd, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_FILE)
    except Exception as e:
        logger.error(f"State save failed: {e}")


# ============================================================
# MARKET SCANNER
# ============================================================
def get_all_tickers() -> list:
    """Get all USDT swap tickers from OKX (trade on futures, detect on futures)."""
    try:
        resp = requests.get(
            'https://www.okx.com/api/v5/market/tickers',
            params={'instType': 'SWAP'},
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            tickers = []
            for t in data.get('data', []):
                inst_id = t.get('instId', '')
                if inst_id.endswith('-USDT-SWAP'):
                    symbol = inst_id.replace('-USDT-SWAP', '')
                    if symbol in BLACKLIST or symbol in MAIN_BOT_PAIRS:
                        continue
                    try:
                        last = float(t.get('last', 0) or 0)
                        open24h = float(t.get('open24h', 0) or 0)
                        high24h = float(t.get('high24h', 0) or 0)
                        low24h = float(t.get('low24h', 0) or 0)
                        vol24h = float(t.get('volCcy24h', 0) or 0)
                    except (ValueError, TypeError):
                        continue

                    change24h = ((last - open24h) / open24h * 100) if open24h > 0 else 0
                    tickers.append({
                        'symbol': symbol,
                        'instId': inst_id,
                        'last': last,
                        'open24h': open24h,
                        'high24h': high24h,
                        'low24h': low24h,
                        'vol24h': vol24h,
                        'change24h': change24h
                    })
            return tickers
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
    return []


def detect_pumps(tickers: list, state: dict) -> list:
    """
    v2: Detect pumps AND dumps, add to watchlist (DON'T trade immediately).

    Stage 1: Find coins with strong momentum in either direction
      - PUMP: >8% up 24h or >5% up short-term → direction='long' (buy the dip)
      - DUMP: >8% down 24h or >5% down short-term → direction='short' (short the bounce)
    Stage 2 (in check_watchlist_entries): Wait for pullback + volume + candle confirm
    """
    new_watchlist_candidates = []
    now = time.time()
    cache = state.get('price_cache', {})
    cooldowns = state.get('cooldowns', {})
    active_symbols = {r['symbol'] for r in state.get('active_rides', [])}
    watchlist_symbols = {w['symbol'] for w in state.get('watchlist', [])}

    for t in tickers:
        symbol = t['symbol']

        if symbol in active_symbols or symbol in watchlist_symbols:
            continue

        # Check cooldown
        if symbol in cooldowns:
            try:
                expiry = datetime.fromisoformat(cooldowns[symbol])
                if datetime.now() < expiry:
                    continue
                else:
                    del cooldowns[symbol]
            except (ValueError, TypeError):
                del cooldowns[symbol]

        if t['vol24h'] < MIN_VOLUME_USDT:
            continue

        price = t['last']
        if price <= 0:
            continue

        detected = False
        direction = None  # 'long' or 'short'
        trigger = ""

        # Short-term detection from price cache
        if symbol in cache:
            cached = cache[symbol]
            cached_price = cached.get('price', 0)
            cached_time = cached.get('time', 0)
            elapsed_min = (now - cached_time) / 60

            if cached_price > 0 and elapsed_min > 0:
                change_pct = ((price - cached_price) / cached_price) * 100
                if 3 <= elapsed_min <= 20:
                    if change_pct >= PUMP_THRESHOLD_SHORT:
                        detected = True
                        direction = 'long'
                        trigger = f'+{change_pct:.1f}% in {elapsed_min:.0f}min'
                    elif change_pct <= -PUMP_THRESHOLD_SHORT:
                        detected = True
                        direction = 'short'
                        trigger = f'{change_pct:.1f}% in {elapsed_min:.0f}min'

        # 24h pump detection (near high = long opportunity)
        if not detected and t['change24h'] >= PUMP_THRESHOLD_24H and t['high24h'] > 0:
            from_high = ((t['high24h'] - price) / t['high24h']) * 100
            if from_high <= 5:  # Within 5% of 24h high
                detected = True
                direction = 'long'
                trigger = f'+{t["change24h"]:.1f}% 24h (near high)'

        # 24h dump detection (near low = short opportunity)
        if not detected and t['change24h'] <= -PUMP_THRESHOLD_24H and t['low24h'] > 0:
            from_low = ((price - t['low24h']) / t['low24h']) * 100
            if from_low <= 5:  # Within 5% of 24h low
                detected = True
                direction = 'short'
                trigger = f'{t["change24h"]:.1f}% 24h (near low)'

        if detected and direction:
            candidate = {
                'symbol': symbol,
                'instId': t['instId'],
                'detected_price': price,
                'vol24h': t['vol24h'],
                'change24h': t['change24h'],
                'trigger': trigger,
                'detected_at': datetime.now().isoformat(),
                'direction': direction,
                'status': 'watching',
            }
            # Track extremes: high for longs (buy dip from high), low for shorts (short bounce from low)
            if direction == 'long':
                candidate['high_price'] = max(price, t.get('high24h', price))
            else:
                candidate['low_price'] = min(price, t.get('low24h', price))

            new_watchlist_candidates.append(candidate)

        cache[symbol] = {'price': price, 'time': now}

    # Prune old cache entries (>2 hours old)
    cutoff = now - 7200
    state['price_cache'] = {k: v for k, v in cache.items() if v.get('time', 0) > cutoff}
    state['cooldowns'] = cooldowns

    new_watchlist_candidates.sort(key=lambda x: abs(x['change24h']), reverse=True)
    return new_watchlist_candidates


def _get_5min_candles(inst_id: str) -> list:
    """Fetch recent 5-minute candles for volume and candle color checks."""
    try:
        resp = requests.get(
            'https://www.okx.com/api/v5/market/candles',
            params={'instId': inst_id, 'bar': '5m', 'limit': '12'},  # Last hour
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            # OKX returns newest first, reverse to chronological
            candles = []
            for d in reversed(data):
                candles.append({
                    'open': float(d[1]),
                    'high': float(d[2]),
                    'low': float(d[3]),
                    'close': float(d[4]),
                    'vol': float(d[5]),
                })
            return candles
    except Exception as e:
        logger.error(f"Error fetching 5min candles for {inst_id}: {e}")
    return []


def check_watchlist_entries(tickers: list, state: dict) -> list:
    """
    v2 CORE: Check watchlist items for entry conditions (both long and short).

    LONG entry (after pump detected):
    1. Price dipped >DIP_ENTRY_PCT from high → 2. Green candle → 3. Volume spike

    SHORT entry (after dump detected):
    1. Price bounced >DIP_ENTRY_PCT from low → 2. Red candle → 3. Volume spike

    Returns list of entries ready to trade.
    """
    ready_to_enter = []
    ticker_map = {t['instId']: t for t in tickers}

    for watch in list(state.get('watchlist', [])):
        symbol = watch['symbol']
        inst_id = watch['instId']
        direction = watch.get('direction', 'long')

        # Check expiry
        try:
            detected_at = datetime.fromisoformat(watch['detected_at'])
            age_min = (datetime.now() - detected_at).total_seconds() / 60
        except (ValueError, TypeError):
            age_min = 999

        if age_min > DIP_MAX_WAIT_MIN:
            logger.info(f"Watchlist expired: {symbol} ({direction}, no entry in {DIP_MAX_WAIT_MIN}min)")
            state['watchlist'].remove(watch)
            continue

        # Get current price
        t = ticker_map.get(inst_id)
        if not t:
            continue

        current_price = t['last']

        if direction == 'long':
            # LONG: track high, wait for dip
            high_price = watch.get('high_price', current_price)
            if current_price > high_price:
                watch['high_price'] = current_price
                high_price = current_price

            pullback_pct = ((high_price - current_price) / high_price) * 100 if high_price > 0 else 0
            if pullback_pct < DIP_ENTRY_PCT:
                continue

            logger.info(f"Watchlist {symbol} LONG: dipped {pullback_pct:.1f}% from high ${high_price:.6f}")
            need_green = True  # Buy on green candle (bounce confirmation)

        else:
            # SHORT: track low, wait for bounce
            low_price = watch.get('low_price', current_price)
            if current_price < low_price:
                watch['low_price'] = current_price
                low_price = current_price

            pullback_pct = ((current_price - low_price) / low_price) * 100 if low_price > 0 else 0
            if pullback_pct < DIP_ENTRY_PCT:
                continue

            logger.info(f"Watchlist {symbol} SHORT: bounced {pullback_pct:.1f}% from low ${low_price:.6f}")
            need_green = False  # Short on red candle (rejection confirmation)

        # Check candle + volume confirmation
        candles = _get_5min_candles(inst_id)
        if not candles or len(candles) < 4:
            continue

        latest = candles[-1]
        is_green = latest['close'] > latest['open']

        # Candle color check: green for longs, red for shorts
        if GREEN_CANDLE_REQUIRED:
            if need_green and not is_green:
                logger.debug(f"Watchlist {symbol}: dipped but candle is red, waiting for green...")
                continue
            if not need_green and is_green:
                logger.debug(f"Watchlist {symbol}: bounced but candle is green, waiting for red...")
                continue

        # Volume check (same for both directions)
        recent_vol = sum(c['vol'] for c in candles[-2:])
        avg_vol = sum(c['vol'] for c in candles[:-2]) / max(len(candles) - 2, 1)
        vol_ratio = recent_vol / (2 * avg_vol) if avg_vol > 0 else 0

        if vol_ratio < VOLUME_SPIKE_RATIO:
            logger.debug(f"Watchlist {symbol}: volume too low ({vol_ratio:.1f}x < {VOLUME_SPIKE_RATIO}x)")
            continue

        # ALL CONDITIONS MET
        dir_label = "LONG" if direction == 'long' else "SHORT"
        candle_label = "green" if is_green else "red"
        action_label = "DIP BUY" if direction == 'long' else "BOUNCE SHORT"
        logger.info(
            f"ENTRY SIGNAL: {symbol} {dir_label} | pullback={pullback_pct:.1f}% | "
            f"candle={candle_label} | vol={vol_ratio:.1f}x"
        )

        ready_to_enter.append({
            'symbol': symbol,
            'instId': inst_id,
            'price': current_price,
            'direction': direction,
            'change_pct': watch['change24h'],
            'vol24h': watch['vol24h'],
            'change24h': watch['change24h'],
            'pullback_pct': pullback_pct,
            'vol_ratio': vol_ratio,
            'trigger': f"{action_label}: {watch['trigger']} → pullback {pullback_pct:.1f}%, vol {vol_ratio:.1f}x",
        })

        state['watchlist'].remove(watch)

    return ready_to_enter


# ============================================================
# TRADING
# ============================================================
def _check_existing_position(swap_id: str) -> bool:
    """Check if there's already a position on this instrument."""
    _init_okx()
    try:
        pos = _acct_api.get_positions(instId=swap_id)
        if pos and pos.get('data'):
            for p in pos['data']:
                if float(p.get('pos', 0)) != 0:
                    return True
    except Exception:
        pass
    return False


def _get_available_balance() -> float:
    """Get available USDT balance."""
    _init_okx()
    try:
        bal = _acct_api.get_account_balance(ccy='USDT')
        if bal and bal.get('data'):
            for b in bal['data']:
                for d in b.get('details', []):
                    return float(d.get('availBal', 0))
    except Exception:
        pass
    return 0


def open_momentum_trade(pump: dict, state: dict) -> bool:
    """Open a momentum trade (long or short). Supports paper mode."""
    symbol = pump['symbol']
    swap_id = pump['instId']
    direction = pump.get('direction', 'long')

    # PAPER MODE: track what would happen without real orders
    if PAPER_MODE:
        paper_trade = {
            'symbol': symbol,
            'instId': swap_id,
            'direction': direction,
            'entry_price': pump['price'],
            'peak_price': pump['price'],
            'trough_price': pump['price'],
            'current_price': pump['price'],
            'contracts': 0,
            'leverage': 3,
            'margin_usdt': RIDE_AMOUNT_USDT,
            'trigger': pump['trigger'],
            'change_at_entry': pump['change_pct'],
            'timestamp': datetime.now().isoformat(),
            'pnl': 0,
            'pnl_pct': 0,
            'paper': True,
        }
        state['active_rides'].append(paper_trade)
        state['stats']['total_trades'] += 1

        dir_emoji = "📈" if direction == 'long' else "📉"
        dir_label = "LONG" if direction == 'long' else "SHORT"
        logger.info(f"PAPER {dir_label}: {symbol} @ ${pump['price']:.6f} | {pump['trigger']}")
        send_telegram(
            f"📝 PAPER {dir_label}: {symbol} {dir_emoji}\n"
            f"Signal: {pump['trigger']}\n"
            f"Entry: ${pump['price']:.6f}\n"
            f"(Paper mode - no real order)\n"
            f"Vol 24h: ${pump['vol24h']:,.0f}"
        )
        return True

    _init_okx()

    try:
        # Check balance
        avail = _get_available_balance()
        if avail < MIN_BALANCE_USDT:
            logger.info(f"Skipping {symbol}: available ${avail:.2f} < ${MIN_BALANCE_USDT}")
            return False

        # Check for existing position (prevent conflicts)
        if _check_existing_position(swap_id):
            logger.info(f"Skipping {symbol}: existing position on {swap_id}")
            return False

        # Check if futures available and get instrument info
        try:
            inst_resp = _pub_api.get_instruments(instType="SWAP", instId=swap_id)
            if not inst_resp or not inst_resp.get('data'):
                logger.info(f"{swap_id} not available for futures")
                return False
            inst = inst_resp['data'][0]
            ct_val = float(inst.get('ctVal', 1))
            min_sz = float(inst.get('minSz', 1))
            lot_sz = float(inst.get('lotSz', 1))
        except Exception:
            return False

        # Set leverage to 3x with ISOLATED margin (doesn't affect main bot)
        leverage = 3
        try:
            _acct_api.set_leverage(instId=swap_id, lever=str(leverage), mgnMode="isolated")
        except Exception as e:
            logger.warning(f"Leverage set failed for {swap_id}: {e}, skipping trade")
            return False  # Don't trade at unknown leverage

        # Calculate contracts
        effective_usdt = RIDE_AMOUNT_USDT * leverage
        price = pump['price']
        raw_contracts = effective_usdt / (ct_val * price)

        if lot_sz > 0:
            num_contracts = round(int(raw_contracts / lot_sz) * lot_sz, 8)
        else:
            num_contracts = int(raw_contracts)

        if num_contracts < min_sz:
            logger.info(f"{symbol}: {num_contracts} contracts < min {min_sz}")
            return False

        # Direction determines order side
        open_side = "buy" if direction == 'long' else "sell"
        close_side = "sell" if direction == 'long' else "buy"
        dir_label = "LONG" if direction == 'long' else "SHORT"

        # Place market order with ISOLATED margin
        result = _trade_api.place_order(
            instId=swap_id,
            tdMode="isolated",
            side=open_side,
            ordType="market",
            sz=str(num_contracts),
            posSide="net"
        )

        if not (result and result.get('data') and result['data'][0].get('ordId')):
            error = result.get('data', [{}])[0].get('sMsg', 'Unknown') if result else 'No response'
            logger.warning(f"Momentum {dir_label} failed for {symbol}: {error}")
            return False

        order_id = result['data'][0]['ordId']

        # Get actual fill price (wait briefly for fill)
        time.sleep(0.5)
        actual_price = price  # fallback
        try:
            order_info = _trade_api.get_order(instId=swap_id, ordId=order_id)
            if order_info and order_info.get('data'):
                fill_px = order_info['data'][0].get('avgPx', '')
                if fill_px:
                    actual_price = float(fill_px)
        except Exception:
            pass

        # Place server-side stop loss (direction-aware)
        if direction == 'long':
            sl_price = round(actual_price * (1 - STOP_LOSS_PCT / 100), 8)
        else:
            sl_price = round(actual_price * (1 + STOP_LOSS_PCT / 100), 8)

        sl_order_id = ""
        try:
            sl_result = _trade_api.place_algo_order(
                instId=swap_id,
                tdMode="isolated",
                side=close_side,
                ordType="conditional",
                sz=str(num_contracts),
                posSide="net",
                slTriggerPx=str(sl_price),
                slOrdPx="-1"  # market price on trigger
            )
            if sl_result and sl_result.get('data'):
                sl_order_id = sl_result['data'][0].get('algoId', '')
                logger.info(f"Server-side SL placed for {symbol} {dir_label} at ${sl_price:.6f}")
        except Exception as e:
            logger.warning(f"Server SL failed for {symbol}: {e} (software SL still active)")

        logger.info(f"MOMENTUM {dir_label}: {symbol} | {num_contracts} contracts @ ${actual_price:.6f} | {pump['trigger']}")

        ride = {
            'symbol': symbol,
            'instId': swap_id,
            'direction': direction,
            'order_id': order_id,
            'sl_order_id': sl_order_id,
            'entry_price': actual_price,
            'peak_price': actual_price,
            'trough_price': actual_price,
            'current_price': actual_price,
            'contracts': num_contracts,
            'leverage': leverage,
            'margin_usdt': RIDE_AMOUNT_USDT,
            'trigger': pump['trigger'],
            'change_at_entry': pump['change_pct'],
            'timestamp': datetime.now().isoformat(),
            'pnl': 0,
            'pnl_pct': 0
        }
        state['active_rides'].append(ride)
        state['stats']['total_trades'] += 1

        dir_emoji = "🚀" if direction == 'long' else "🔻"
        send_telegram(
            f"{dir_emoji} MOMENTUM {dir_label}: {symbol}\n"
            f"Signal: {pump['trigger']}\n"
            f"Entry: ${actual_price:.6f}\n"
            f"Size: {num_contracts} contracts (3x isolated)\n"
            f"Margin: ${RIDE_AMOUNT_USDT}\n"
            f"Vol 24h: ${pump['vol24h']:,.0f}\n"
            f"Server SL: ${sl_price:.6f} ({STOP_LOSS_PCT}%)\n"
            f"Trail: {TRAILING_STOP_PCT}% | TP: {TAKE_PROFIT_PCT}%"
        )
        return True

    except Exception as e:
        logger.error(f"Error opening momentum trade for {symbol}: {e}")
        return False


def check_ride_exit(ride: dict) -> str:
    """Check if a momentum ride should be closed (works for both long and short)."""
    try:
        swap_id = ride['instId']
        entry_time = datetime.fromisoformat(ride['timestamp'])
        entry_price = ride['entry_price']
        direction = ride.get('direction', 'long')

        # Get current price from swap ticker
        resp = requests.get(
            'https://www.okx.com/api/v5/market/ticker',
            params={'instId': swap_id},
            timeout=10
        )
        if resp.status_code != 200:
            return 'hold'

        data = resp.json()
        tickers = data.get('data', [])
        if not tickers:
            return 'hold'

        current_price = float(tickers[0]['last'])
        ride['current_price'] = current_price

        # Track extremes
        peak_price = ride.get('peak_price', entry_price)
        trough_price = ride.get('trough_price', entry_price)
        if current_price > peak_price:
            ride['peak_price'] = current_price
            peak_price = current_price
        if current_price < trough_price:
            ride['trough_price'] = current_price
            trough_price = current_price

        # P&L calculation (direction-aware)
        if direction == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            # Trailing: how far has price fallen from peak?
            drawback_pct = ((peak_price - current_price) / peak_price) * 100 if peak_price > 0 else 0
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            # Trailing: how far has price risen from trough?
            drawback_pct = ((current_price - trough_price) / trough_price) * 100 if trough_price > 0 else 0

        ride['pnl_pct'] = pnl_pct
        ride['pnl'] = (pnl_pct / 100) * ride.get('leverage', 3) * RIDE_AMOUNT_USDT

        # Take profit
        if pnl_pct >= TAKE_PROFIT_PCT:
            return 'take_profit'

        # Trailing stop (activate after TRAILING_ACTIVATE_PCT profit)
        if pnl_pct > TRAILING_ACTIVATE_PCT and drawback_pct >= TRAILING_STOP_PCT:
            return 'trailing_stop'

        # Software stop loss (backup for server-side SL)
        if pnl_pct <= -STOP_LOSS_PCT:
            return 'stop_loss'

        # Time exit
        if datetime.now() - entry_time > timedelta(minutes=MAX_HOLD_MINUTES):
            return 'time_exit'

        return 'hold'

    except Exception as e:
        logger.error(f"Error checking ride exit for {ride.get('symbol')}: {e}")
        return 'hold'


def close_ride(ride: dict, reason: str, state: dict) -> bool:
    """Close a momentum ride (long or short). Supports paper mode."""
    direction = ride.get('direction', 'long')
    dir_label = "LONG" if direction == 'long' else "SHORT"

    # PAPER MODE: just log it
    if ride.get('paper', False):
        pnl_pct = ride.get('pnl_pct', 0)
        pnl_usd = ride.get('pnl', 0)
        icon = '🟢' if pnl_pct > 0 else '🔴'

        if pnl_pct > 0:
            state['stats']['wins'] += 1
        else:
            state['stats']['losses'] += 1
        state['stats']['total_pnl'] += pnl_usd

        state['cooldowns'][ride['symbol']] = (
            datetime.now() + timedelta(minutes=COOLDOWN_MINUTES)
        ).isoformat()

        try:
            entry_dt = datetime.fromisoformat(ride['timestamp'])
            hold_mins = (datetime.now() - entry_dt).total_seconds() / 60
            hold_str = f"{hold_mins:.0f}min"
        except Exception:
            hold_str = "?"

        ride['exit_reason'] = reason
        ride['exit_time'] = datetime.now().isoformat()
        ride['exit_price'] = ride['current_price']
        ride['hold_minutes'] = hold_str
        state['completed_rides'].append(ride)
        state['active_rides'].remove(ride)

        if len(state['completed_rides']) > MAX_COMPLETED_RIDES:
            state['completed_rides'] = state['completed_rides'][-MAX_COMPLETED_RIDES:]

        stats = state['stats']
        wr = (stats['wins'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0

        logger.info(f"PAPER CLOSE: {ride['symbol']} {dir_label} | {reason} | PnL: {pnl_pct:+.1f}%")
        send_telegram(
            f"{icon} PAPER EXIT: {ride['symbol']} ({dir_label})\n"
            f"Reason: {reason}\n"
            f"Entry: ${ride['entry_price']:.6f}\n"
            f"Exit: ${ride['current_price']:.6f}\n"
            f"PnL: {pnl_pct:+.1f}% (${pnl_usd:+.2f})\n"
            f"Hold: {hold_str}\n"
            f"─────\n"
            f"Stats: {stats['wins']}W/{stats['losses']}L ({wr:.0f}%) | Total: ${stats['total_pnl']:+.2f}"
        )
        return True

    _init_okx()
    try:
        swap_id = ride['instId']
        close_side = "sell" if direction == 'long' else "buy"

        # Get actual position size from exchange
        actual_sz = None
        try:
            pos = _acct_api.get_positions(instId=swap_id)
            if pos and pos.get('data'):
                for p in pos['data']:
                    pos_sz = float(p.get('pos', 0))
                    # Long = positive pos, Short = negative pos
                    if (direction == 'long' and pos_sz > 0) or (direction == 'short' and pos_sz < 0):
                        actual_sz = str(abs(pos_sz))
                        break
        except Exception:
            pass

        if actual_sz is None or float(actual_sz) == 0:
            logger.warning(f"No {dir_label} position found for {ride['symbol']}, removing from tracking")
            state['active_rides'].remove(ride)
            return True

        # Cancel server-side stop loss if exists
        if ride.get('sl_order_id'):
            try:
                _trade_api.cancel_algo_order([{'instId': swap_id, 'algoId': ride['sl_order_id']}])
            except Exception:
                pass

        # Close position using actual size and correct side
        result = _trade_api.place_order(
            instId=swap_id,
            tdMode="isolated",
            side=close_side,
            ordType="market",
            sz=actual_sz,
            posSide="net"
        )

        if result and result.get('data') and result['data'][0].get('ordId'):
            pnl_pct = ride.get('pnl_pct', 0)
            pnl_usd = ride.get('pnl', 0)
            icon = '🟢' if pnl_pct > 0 else '🔴'

            if pnl_pct > 0:
                state['stats']['wins'] += 1
            else:
                state['stats']['losses'] += 1
            state['stats']['total_pnl'] += pnl_usd

            state['cooldowns'][ride['symbol']] = (
                datetime.now() + timedelta(minutes=COOLDOWN_MINUTES)
            ).isoformat()

            try:
                entry_dt = datetime.fromisoformat(ride['timestamp'])
                hold_mins = (datetime.now() - entry_dt).total_seconds() / 60
                hold_str = f"{hold_mins:.0f} min"
            except Exception:
                hold_str = "?"

            ride['exit_reason'] = reason
            ride['exit_time'] = datetime.now().isoformat()
            ride['exit_price'] = ride['current_price']
            state['completed_rides'].append(ride)
            state['active_rides'].remove(ride)

            if len(state['completed_rides']) > MAX_COMPLETED_RIDES:
                state['completed_rides'] = state['completed_rides'][-MAX_COMPLETED_RIDES:]

            stats = state['stats']
            win_rate = (stats['wins'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0

            logger.info(f"RIDE CLOSED: {ride['symbol']} {dir_label} | {reason} | PnL: {pnl_pct:+.1f}%")

            send_telegram(
                f"{icon} MOMENTUM EXIT: {ride['symbol']} ({dir_label})\n"
                f"Reason: {reason}\n"
                f"Entry: ${ride['entry_price']:.6f}\n"
                f"Exit: ${ride['current_price']:.6f}\n"
                f"PnL: {pnl_pct:+.1f}% (${pnl_usd:+.2f})\n"
                f"Hold time: {hold_str}\n"
                f"─────\n"
                f"Stats: {stats['wins']}W/{stats['losses']}L ({win_rate:.0f}%) | Total: ${stats['total_pnl']:+.2f}"
            )
            return True
        else:
            error = result.get('data', [{}])[0].get('sMsg', 'Unknown') if result else 'No response'
            logger.error(f"Close ride failed for {ride['symbol']}: {error}")
            return False

    except Exception as e:
        logger.error(f"Error closing ride for {ride.get('symbol')}: {e}")
        return False


def _reconcile_positions(state: dict):
    """On startup, check that active rides have matching positions on exchange."""
    _init_okx()
    for ride in list(state.get('active_rides', [])):
        try:
            pos = _acct_api.get_positions(instId=ride['instId'])
            has_pos = False
            if pos and pos.get('data'):
                for p in pos['data']:
                    if float(p.get('pos', 0)) > 0:
                        has_pos = True
                        break
            if not has_pos:
                logger.warning(f"Orphan ride {ride['symbol']}: no position on exchange, removing")
                state['active_rides'].remove(ride)
        except Exception:
            pass


# ============================================================
# MAIN LOOP
# ============================================================
def run_momentum_scanner(state: dict = None):
    """
    v2: Two-stage momentum scanner.

    Stage 1: Detect pumps → add to watchlist
    Stage 2: Check watchlist for dip entry conditions → trade
    """
    if state is None:
        state = load_state()

    # Ensure v2 state fields exist
    if 'watchlist' not in state:
        state['watchlist'] = []
    if 'paper_trades' not in state:
        state['paper_trades'] = []

    try:
        tickers = get_all_tickers()
        if not tickers:
            return state

        # Build ticker lookup
        ticker_map = {}
        for t in tickers:
            ticker_map[t['instId']] = t

        # Stage 1: Detect new pumps/dumps → add to watchlist
        new_candidates = detect_pumps(tickers, state)
        for candidate in new_candidates:
            if len(state['watchlist']) < 10:  # Cap watchlist size
                state['watchlist'].append(candidate)
                d = candidate.get('direction', 'long').upper()
                action = "dip" if candidate.get('direction') == 'long' else "bounce"
                logger.info(
                    f"WATCHLIST +{candidate['symbol']} ({d}): {candidate['trigger']} "
                    f"(waiting for {DIP_ENTRY_PCT}% {action})"
                )

        # Stage 2: Check watchlist for dip entry signals
        ready_entries = check_watchlist_entries(tickers, state)

        active_count = len(state.get('active_rides', []))
        for entry in ready_entries:
            if active_count >= MAX_CONCURRENT_RIDES:
                break

            logger.info(f"DIP ENTRY: {entry['symbol']} {entry['trigger']}")
            if open_momentum_trade(entry, state):
                active_count += 1

        # Check active rides for exit
        for ride in list(state.get('active_rides', [])):
            action = check_ride_exit(ride)
            if action != 'hold':
                close_ride(ride, action, state)

        save_state(state)

    except Exception as e:
        logger.error(f"Momentum scanner error: {e}")

    return state


def main():
    """Standalone momentum rider loop."""
    mode_str = "PAPER MODE" if PAPER_MODE else "LIVE"
    logger.info("=" * 50)
    logger.info(f"MOMENTUM RIDER v2.0 STARTING ({mode_str})")
    logger.info("=" * 50)
    logger.info(f"Strategy: Detect pump → watchlist → wait for {DIP_ENTRY_PCT}% dip → confirm → enter")
    logger.info(f"Pump threshold: >{PUMP_THRESHOLD_24H}% (24h) or >{PUMP_THRESHOLD_SHORT}% (short)")
    logger.info(f"Entry: dip {DIP_ENTRY_PCT}% + vol {VOLUME_SPIKE_RATIO}x + green candle={GREEN_CANDLE_REQUIRED}")
    logger.info(f"Ride size: ${RIDE_AMOUNT_USDT} @ 3x ISOLATED leverage")
    logger.info(f"Max concurrent: {MAX_CONCURRENT_RIDES}")
    logger.info(f"Trailing stop: {TRAILING_STOP_PCT}% | SL: {STOP_LOSS_PCT}% | TP: {TAKE_PROFIT_PCT}%")
    logger.info(f"Excluded pairs (main bot): {MAIN_BOT_PAIRS}")

    if not PAPER_MODE:
        _init_okx()

    state = load_state()

    # Reset v1 state if upgrading
    if state.get('active_rides') and not PAPER_MODE:
        _reconcile_positions(state)

    send_telegram(
        f"🏄 Momentum Rider v2.0 ({mode_str})\n"
        f"Strategy: dip-buy (wait for pullback)\n"
        f"Scan: every {SCAN_INTERVAL_SECONDS // 60}min\n"
        f"Detect: >{PUMP_THRESHOLD_24H}% 24h or >{PUMP_THRESHOLD_SHORT}% short\n"
        f"Entry: {DIP_ENTRY_PCT}% dip + {VOLUME_SPIKE_RATIO}x vol + green candle\n"
        f"Ride: ${RIDE_AMOUNT_USDT} @ 3x | SL: {STOP_LOSS_PCT}% | Trail: {TRAILING_STOP_PCT}% | TP: {TAKE_PROFIT_PCT}%\n"
        f"Max rides: {MAX_CONCURRENT_RIDES}\n"
        f"Excluded: {', '.join(sorted(MAIN_BOT_PAIRS))}"
    )

    cycle = 0

    while True:
        try:
            cycle += 1
            state = run_momentum_scanner(state)

            # Status log every ~24 min (12 cycles × 2 min)
            if cycle % 12 == 0:
                active = len(state.get('active_rides', []))
                watching = len(state.get('watchlist', []))
                stats = state.get('stats', {})
                logger.info(
                    f"Momentum v2: {active} active | {watching} watching | "
                    f"{stats.get('total_trades', 0)} trades | "
                    f"${stats.get('total_pnl', 0):+.2f} total PnL"
                )

            time.sleep(SCAN_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Momentum rider stopped")
            send_telegram("🛑 Momentum Rider v2.0 stopped")
            save_state(state)
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(60)


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv('/root/Cryptobot/.env')
    Path('/root/Cryptobot/data').mkdir(exist_ok=True)
    main()
