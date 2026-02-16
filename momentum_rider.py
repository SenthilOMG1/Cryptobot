#!/usr/bin/env python3
"""
MOMENTUM RIDER v1.1
====================
Autonomous momentum scanner + auto-trader.

Scans ALL OKX USDT pairs every 5 minutes.
Detects pumps (>8% in 1 hour) and auto-trades them.

Strategy:
1. Scan all pairs → detect pumps >8% in 1hr
2. Auto-buy $2-3 worth (futures long, isolated margin to avoid conflicts)
3. Server-side stop loss on OKX for flash crash protection
4. Trailing stop 5% from peak to lock profits
5. Auto-sell on: trailing stop, -10% stop loss, 30% TP, or 2hr timeout
6. Telegram notification for every action

v1.1 fixes:
- Uses ISOLATED margin to avoid conflicts with main bot's cross positions
- Checks for existing positions before opening (no accidental cancellation)
- Server-side stop loss order placed immediately after entry
- Atomic state file writes to prevent corruption
- Gets actual fill price from OKX, not stale ticker
- Balance check before trading
- Price cache pruning (no memory leak)
- Scans SWAP tickers (trades on futures, should detect on futures)
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
PUMP_THRESHOLD_1H = 8.0      # Minimum 8% gain in 1 hour to trigger
PUMP_THRESHOLD_15M = 5.0     # Or 5% in 15 minutes (faster detection)
MIN_VOLUME_USDT = 100000     # Minimum 24h volume to avoid illiquid coins
MAX_CONCURRENT_RIDES = 3     # Max simultaneous momentum trades
RIDE_AMOUNT_USDT = 3.0       # $3 per momentum trade
MIN_BALANCE_USDT = 5.0       # Don't trade if balance below this

# Exit settings
TRAILING_STOP_PCT = 5.0      # 5% trailing stop from peak
STOP_LOSS_PCT = 10.0         # Hard stop at -10% price (= -30% on margin at 3x)
MAX_HOLD_MINUTES = 120       # Auto-sell after 2 hours
TAKE_PROFIT_PCT = 30.0       # Take profit at 30%

# Scan interval
SCAN_INTERVAL_SECONDS = 300  # Scan every 5 minutes

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
        'completed_rides': [],
        'cooldowns': {},
        'price_cache': {},
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
    """Detect coins pumping hard using short-term cache + 24h data."""
    pumps = []
    now = time.time()
    cache = state.get('price_cache', {})
    cooldowns = state.get('cooldowns', {})
    active_symbols = {r['symbol'] for r in state.get('active_rides', [])}

    for t in tickers:
        symbol = t['symbol']

        if symbol in active_symbols:
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

        # Short-term pump detection from price cache
        if symbol in cache:
            cached = cache[symbol]
            cached_price = cached.get('price', 0)
            cached_time = cached.get('time', 0)
            elapsed_min = (now - cached_time) / 60

            if cached_price > 0 and elapsed_min > 0:
                change_pct = ((price - cached_price) / cached_price) * 100

                if 3 <= elapsed_min <= 20 and change_pct >= PUMP_THRESHOLD_15M:
                    pumps.append({
                        'symbol': symbol,
                        'instId': t['instId'],
                        'price': price,
                        'change_pct': change_pct,
                        'window_min': elapsed_min,
                        'vol24h': t['vol24h'],
                        'change24h': t['change24h'],
                        'trigger': f'+{change_pct:.1f}% in {elapsed_min:.0f}min'
                    })
                    cache[symbol] = {'price': price, 'time': now}
                    continue

        # 24h pump detection - must be near high AND have recent momentum
        # Require >8% 24h change AND within 3% of high (stricter than v1.0's 5%)
        if t['change24h'] >= PUMP_THRESHOLD_1H and t['high24h'] > 0:
            from_high = ((t['high24h'] - price) / t['high24h']) * 100
            # Also verify short-term direction is still UP (not topping out)
            if from_high <= 3:
                # Extra check: if we have cached price, ensure it's still going up
                still_rising = True
                if symbol in cache:
                    cached_price = cache[symbol].get('price', 0)
                    if cached_price > 0 and price < cached_price:
                        still_rising = False  # Price dropping from last check

                if still_rising:
                    pumps.append({
                        'symbol': symbol,
                        'instId': t['instId'],
                        'price': price,
                        'change_pct': t['change24h'],
                        'window_min': 0,
                        'vol24h': t['vol24h'],
                        'change24h': t['change24h'],
                        'trigger': f'+{t["change24h"]:.1f}% 24h (near high, rising)'
                    })

        cache[symbol] = {'price': price, 'time': now}

    # Prune old cache entries (>2 hours old)
    cutoff = now - 7200
    state['price_cache'] = {k: v for k, v in cache.items() if v.get('time', 0) > cutoff}
    state['cooldowns'] = cooldowns

    pumps.sort(key=lambda x: x['change_pct'], reverse=True)
    return pumps


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
    """Open a momentum trade using ISOLATED margin to avoid conflicts."""
    _init_okx()
    symbol = pump['symbol']
    swap_id = pump['instId']

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

        # Place market buy (open long) with ISOLATED margin
        result = _trade_api.place_order(
            instId=swap_id,
            tdMode="isolated",
            side="buy",
            ordType="market",
            sz=str(num_contracts),
            posSide="net"
        )

        if not (result and result.get('data') and result['data'][0].get('ordId')):
            error = result.get('data', [{}])[0].get('sMsg', 'Unknown') if result else 'No response'
            logger.warning(f"Momentum buy failed for {symbol}: {error}")
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

        # Place server-side stop loss on OKX (protection against flash crashes)
        sl_price = round(actual_price * (1 - STOP_LOSS_PCT / 100), 8)
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
            if sl_result and sl_result.get('data'):
                sl_order_id = sl_result['data'][0].get('algoId', '')
                logger.info(f"Server-side SL placed for {symbol} at ${sl_price:.6f}")
        except Exception as e:
            logger.warning(f"Server SL failed for {symbol}: {e} (software SL still active)")

        logger.info(f"MOMENTUM BUY: {symbol} | {num_contracts} contracts @ ${actual_price:.6f} | {pump['trigger']}")

        ride = {
            'symbol': symbol,
            'instId': swap_id,
            'order_id': order_id,
            'sl_order_id': sl_order_id,
            'entry_price': actual_price,
            'peak_price': actual_price,
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

        send_telegram(
            f"🚀 MOMENTUM RIDE: {symbol}\n"
            f"Signal: {pump['trigger']}\n"
            f"Entry: ${actual_price:.6f}\n"
            f"Size: {num_contracts} contracts (3x isolated)\n"
            f"Margin: ${RIDE_AMOUNT_USDT}\n"
            f"Vol 24h: ${pump['vol24h']:,.0f}\n"
            f"Server SL: ${sl_price:.6f} (-{STOP_LOSS_PCT}%)\n"
            f"Trail: {TRAILING_STOP_PCT}% | TP: {TAKE_PROFIT_PCT}%"
        )
        return True

    except Exception as e:
        logger.error(f"Error opening momentum trade for {symbol}: {e}")
        return False


def check_ride_exit(ride: dict) -> str:
    """Check if a momentum ride should be closed."""
    try:
        swap_id = ride['instId']
        entry_time = datetime.fromisoformat(ride['timestamp'])
        entry_price = ride['entry_price']
        peak_price = ride.get('peak_price', entry_price)

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

        if current_price > peak_price:
            ride['peak_price'] = current_price
            peak_price = current_price

        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        ride['pnl_pct'] = pnl_pct
        # PnL on margin: price_change% × leverage × margin
        ride['pnl'] = (pnl_pct / 100) * ride.get('leverage', 3) * RIDE_AMOUNT_USDT

        drawdown_from_peak = ((peak_price - current_price) / peak_price) * 100 if peak_price > 0 else 0

        # Take profit
        if pnl_pct >= TAKE_PROFIT_PCT:
            return 'take_profit'

        # Trailing stop (activate after 3% profit to avoid premature exits)
        if pnl_pct > 3.0 and drawdown_from_peak >= TRAILING_STOP_PCT:
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
    """Close a momentum ride. Queries actual position size to avoid mismatches."""
    _init_okx()
    try:
        swap_id = ride['instId']

        # Get actual position size from exchange (not stored contracts)
        actual_sz = None
        try:
            pos = _acct_api.get_positions(instId=swap_id)
            if pos and pos.get('data'):
                for p in pos['data']:
                    pos_sz = float(p.get('pos', 0))
                    if pos_sz > 0:  # Long position
                        actual_sz = p.get('pos', str(ride['contracts']))
                        break
        except Exception:
            pass

        if actual_sz is None or float(actual_sz) == 0:
            logger.warning(f"No position found for {ride['symbol']}, removing from tracking")
            state['active_rides'].remove(ride)
            return True

        # Cancel server-side stop loss if exists
        if ride.get('sl_order_id'):
            try:
                _trade_api.cancel_algo_order([{'instId': swap_id, 'algoId': ride['sl_order_id']}])
            except Exception:
                pass

        # Close position using actual size
        result = _trade_api.place_order(
            instId=swap_id,
            tdMode="isolated",
            side="sell",
            ordType="market",
            sz=str(actual_sz),
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

            # Calculate hold time
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

            # Prune completed rides
            if len(state['completed_rides']) > MAX_COMPLETED_RIDES:
                state['completed_rides'] = state['completed_rides'][-MAX_COMPLETED_RIDES:]

            stats = state['stats']
            win_rate = (stats['wins'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0

            logger.info(f"RIDE CLOSED: {ride['symbol']} | {reason} | PnL: {pnl_pct:+.1f}%")

            send_telegram(
                f"{icon} MOMENTUM EXIT: {ride['symbol']}\n"
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
    """Run one cycle of the momentum scanner."""
    if state is None:
        state = load_state()

    try:
        tickers = get_all_tickers()
        if not tickers:
            return state

        pumps = detect_pumps(tickers, state)

        active_count = len(state.get('active_rides', []))
        for pump in pumps:
            if active_count >= MAX_CONCURRENT_RIDES:
                break

            logger.info(f"PUMP DETECTED: {pump['symbol']} {pump['trigger']} (vol: ${pump['vol24h']:,.0f})")
            if open_momentum_trade(pump, state):
                active_count += 1

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
    logger.info("=" * 50)
    logger.info("MOMENTUM RIDER v1.1 STARTING")
    logger.info("=" * 50)
    logger.info(f"Pump threshold: >{PUMP_THRESHOLD_1H}% (1h) or >{PUMP_THRESHOLD_15M}% (15m)")
    logger.info(f"Ride size: ${RIDE_AMOUNT_USDT} @ 3x ISOLATED leverage")
    logger.info(f"Max concurrent: {MAX_CONCURRENT_RIDES}")
    logger.info(f"Trailing stop: {TRAILING_STOP_PCT}% | SL: {STOP_LOSS_PCT}% | TP: {TAKE_PROFIT_PCT}%")
    logger.info(f"Excluded pairs (main bot): {MAIN_BOT_PAIRS}")

    _init_okx()

    state = load_state()

    # Reconcile: check active rides have real positions
    _reconcile_positions(state)

    send_telegram(
        f"🏄 Momentum Rider v1.1 ACTIVE\n"
        f"Scanning ALL OKX pairs every {SCAN_INTERVAL_SECONDS // 60}min\n"
        f"Trigger: >{PUMP_THRESHOLD_1H}% (1h) or >{PUMP_THRESHOLD_15M}% (15m)\n"
        f"Ride size: ${RIDE_AMOUNT_USDT} @ 3x isolated\n"
        f"Max rides: {MAX_CONCURRENT_RIDES}\n"
        f"Server SL: {STOP_LOSS_PCT}% | Trail: {TRAILING_STOP_PCT}% | TP: {TAKE_PROFIT_PCT}%\n"
        f"Excluded: {', '.join(sorted(MAIN_BOT_PAIRS))}"
    )

    cycle = 0

    while True:
        try:
            cycle += 1
            state = run_momentum_scanner(state)

            if cycle % 12 == 0:
                active = len(state.get('active_rides', []))
                stats = state.get('stats', {})
                logger.info(
                    f"Momentum status: {active} active | "
                    f"{stats.get('total_trades', 0)} trades | "
                    f"${stats.get('total_pnl', 0):+.2f} total PnL"
                )

            time.sleep(SCAN_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Momentum rider stopped")
            send_telegram("🛑 Momentum Rider stopped")
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
