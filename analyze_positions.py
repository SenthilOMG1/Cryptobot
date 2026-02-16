#!/usr/bin/env python3
"""
Live Futures Position Analyzer
==============================
Connects to OKX, fetches open futures positions, runs the ensemble model
on current market data for each position, and provides risk assessment.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Load .env before any imports that use env vars
from dotenv import load_dotenv
load_dotenv("/root/Cryptobot/.env")

# Add project root to path
sys.path.insert(0, "/root/Cryptobot")

import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("okx").setLevel(logging.ERROR)

import time
import numpy as np
import pandas as pd
from datetime import datetime

from src.config import get_config
from src.security.vault import SecureVault
from src.trading.okx_client import SecureOKXClient
from src.data.collector import DataCollector
from src.data.features import FeatureEngine
from src.models.xgboost_model import XGBoostPredictor
from src.models.rl_agent import RLTradingAgent
from src.models.ensemble import EnsembleDecider, Action

# --- Formatting Helpers ---

def fmt_usd(val):
    if val >= 0:
        return f"${val:,.2f}"
    return f"-${abs(val):,.2f}"

def fmt_pct(val):
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"

def action_str(action_int):
    mapping = {1: "BUY (Go Long)", -1: "SELL (Go Short)", 0: "HOLD"}
    return mapping.get(action_int, "UNKNOWN")

def side_label(side):
    return "LONG" if side == "long" else "SHORT"

def signal_assessment(current_side, model_action):
    if model_action == 0:
        return "NEUTRAL - Model says HOLD"
    if current_side == "long" and model_action == 1:
        return "AGREES - Model confirms LONG"
    if current_side == "short" and model_action == -1:
        return "AGREES - Model confirms SHORT"
    if current_side == "long" and model_action == -1:
        return "DISAGREES - Model says SHORT (consider closing)"
    if current_side == "short" and model_action == 1:
        return "DISAGREES - Model says LONG (consider closing)"
    return "MIXED"

def safe_float(val, default=0.0):
    """Safely convert to float, handling empty strings."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# --- Main Analysis ---

def main():
    print("=" * 78)
    print("  CRYPTOBOT - LIVE FUTURES POSITION ANALYSIS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)
    print()

    # 1. Initialize
    print("[1/6] Initializing connections and models...")
    config = get_config()
    vault = SecureVault()
    client = SecureOKXClient(vault, demo_mode=config.exchange.demo_mode)
    collector = DataCollector(client)
    feature_engine = FeatureEngine()

    # Load ML models
    xgb_model = XGBoostPredictor(model_path="/root/Cryptobot/models/xgboost_model.json")
    rl_agent = RLTradingAgent(model_path="/root/Cryptobot/models/rl_agent.zip")
    ensemble = EnsembleDecider(
        xgb_model=xgb_model,
        rl_agent=rl_agent,
        min_confidence=config.trading.min_confidence_to_trade
    )
    print(f"  Models loaded. XGBoost features: {len(xgb_model.feature_names)}")
    print(f"  RL Agent features: {len(rl_agent.feature_columns)}")
    print(f"  Min confidence threshold: {config.trading.min_confidence_to_trade}")
    print(f"  Stop-loss: {config.trading.stop_loss_percent}%  |  Take-profit: {config.trading.take_profit_percent}%")
    print(f"  Trailing stop: {config.trading.trailing_stop_percent}%")
    print()

    # 2. Get account balance
    print("[2/6] Fetching account balance...")
    usdt_balance = 0
    total_equity = 0
    try:
        raw_resp = client._account_api.get_account_balance()
        acct_data = raw_resp.get("data", [{}])[0] if raw_resp.get("code") == "0" else {}

        total_equity = safe_float(acct_data.get("totalEq"))

        # Get USDT details
        usdt_detail = {}
        balances = {}
        for d in acct_data.get("details", []):
            ccy = d.get("ccy", "")
            eq = safe_float(d.get("eq"))
            if eq > 0:
                balances[ccy] = eq
            if ccy == "USDT":
                usdt_detail = d

        usdt_balance = safe_float(usdt_detail.get("availBal"))
        usdt_equity = safe_float(usdt_detail.get("eq"))
        usdt_frozen = safe_float(usdt_detail.get("frozenBal"))
        usdt_imr = safe_float(usdt_detail.get("imr"))
        usdt_mmr = safe_float(usdt_detail.get("mmr"))
        usdt_upl = safe_float(usdt_detail.get("upl"))
        usdt_mgn_ratio = safe_float(usdt_detail.get("mgnRatio"))
        usdt_notional_lever = safe_float(usdt_detail.get("notionalLever"))

        print(f"  Total Account Equity:  {fmt_usd(total_equity)}")
        print(f"  USDT Equity:           {fmt_usd(usdt_equity)}")
        print(f"  USDT Available:        {fmt_usd(usdt_balance)}")
        print(f"  USDT Frozen (margin):  {fmt_usd(usdt_frozen)}")
        print(f"  Initial Margin (IMR):  {fmt_usd(usdt_imr)}")
        print(f"  Maint. Margin (MMR):   {fmt_usd(usdt_mmr)}")
        print(f"  Unrealized PnL:        {fmt_usd(usdt_upl)}")
        print(f"  Margin Ratio:          {usdt_mgn_ratio:.2f}%")
        print(f"  Notional Leverage:     {usdt_notional_lever:.2f}x")

        # Show non-USDT balances
        non_usdt = {k: v for k, v in balances.items() if k != "USDT"}
        if non_usdt:
            print(f"  Other Assets:")
            for ccy, eq in non_usdt.items():
                print(f"    {ccy}: {eq:.8f}")

    except Exception as e:
        print(f"  ERROR fetching balance: {e}")
        import traceback
        traceback.print_exc()
    print()

    # 3. Fetch open futures positions from OKX
    print("[3/6] Fetching open futures positions...")
    try:
        futures_positions = client.get_futures_positions()
        open_positions = [p for p in futures_positions if float(p.get("pos", 0)) != 0]
    except Exception as e:
        print(f"  ERROR: {e}")
        open_positions = []

    if not open_positions:
        print("  No open futures positions found.")
        print()
        print("=" * 78)
        print("  SUMMARY: No futures positions to analyze. Account is flat.")
        print(f"  Total Equity: {fmt_usd(total_equity)}")
        print("=" * 78)
        return

    print(f"  Found {len(open_positions)} open futures position(s)")
    print()

    # 4. Analyze each futures position
    print("[4/6] Analyzing each position with current market data + ML models...")
    print()

    position_analyses = []
    total_unrealized_pnl = 0
    total_notional = 0

    for i, pos in enumerate(open_positions):
        inst_id = pos.get("instId", "")
        pair = inst_id.replace("-SWAP", "")
        pos_amt = safe_float(pos.get("pos"))
        avg_px = safe_float(pos.get("avgPx"))
        mark_px = safe_float(pos.get("markPx"))
        upl = safe_float(pos.get("upl"))
        uplRatio = safe_float(pos.get("uplRatio"))
        lever = int(safe_float(pos.get("lever"), 1))
        margin_mode = pos.get("mgnMode", "cross")
        liq_px_str = pos.get("liqPx", "")
        margin = safe_float(pos.get("margin"))
        notional_usd = safe_float(pos.get("notionalUsd"))
        side = "long" if pos_amt > 0 else "short"
        ctime = pos.get("cTime", "")

        # Get current ticker
        try:
            ticker = client.get_ticker(inst_id)
            current_price = safe_float(ticker.get("last"), mark_px)
            bid = safe_float(ticker.get("bidPx"))
            ask = safe_float(ticker.get("askPx"))
            vol24h = safe_float(ticker.get("vol24h"))
            chg24h = safe_float(ticker.get("open24h"))
        except:
            current_price = mark_px
            bid = ask = vol24h = 0
            chg24h = 0

        total_unrealized_pnl += upl
        total_notional += abs(notional_usd)

        # Calculate P&L
        if avg_px > 0:
            if side == "long":
                pnl_pct = ((current_price - avg_px) / avg_px) * 100
            else:
                pnl_pct = ((avg_px - current_price) / avg_px) * 100
            leveraged_pnl_pct = pnl_pct * lever
        else:
            pnl_pct = leveraged_pnl_pct = 0

        # Position age
        age_str = "unknown"
        if ctime:
            try:
                entry_dt = datetime.fromtimestamp(int(ctime) / 1000)
                age = datetime.now() - entry_dt
                hours = age.seconds // 3600
                mins = (age.seconds % 3600) // 60
                age_str = f"{age.days}d {hours}h {mins}m"
            except:
                pass

        # 24h price change
        price_chg_24h = ""
        if chg24h > 0:
            pct_24h = ((current_price - chg24h) / chg24h) * 100
            price_chg_24h = f" (24h: {fmt_pct(pct_24h)})"

        print(f"  {'=' * 72}")
        print(f"  POSITION {i+1}: {pair} | {side_label(side)} | {lever}x {margin_mode}")
        print(f"  {'=' * 72}")
        print(f"    Contracts:       {abs(pos_amt)}")
        print(f"    Entry Price:     {fmt_usd(avg_px)}")
        print(f"    Current Price:   {fmt_usd(current_price)}{price_chg_24h}")
        print(f"    Mark Price:      {fmt_usd(mark_px)}")
        if bid > 0:
            spread_pct = ((ask - bid) / bid * 100) if bid > 0 else 0
            print(f"    Bid/Ask:         {fmt_usd(bid)} / {fmt_usd(ask)} (spread: {spread_pct:.3f}%)")
        print(f"    Unrealized PnL:  {fmt_usd(upl)} ({fmt_pct(pnl_pct)} price, {fmt_pct(leveraged_pnl_pct)} with leverage)")
        if uplRatio != 0:
            print(f"    PnL Ratio (OKX): {fmt_pct(uplRatio * 100)}")
        print(f"    Notional Value:  {fmt_usd(abs(notional_usd))}")
        if margin > 0:
            print(f"    Margin Used:     {fmt_usd(margin)}")

        liq_price = safe_float(liq_px_str)
        if liq_price > 0:
            dist_to_liq = abs(current_price - liq_price) / current_price * 100
            print(f"    Liquidation Px:  {fmt_usd(liq_price)} ({dist_to_liq:.1f}% away)")

        print(f"    Position Age:    {age_str}")

        # Distance to stop-loss and take-profit
        sl_trigger = config.trading.stop_loss_percent
        tp_trigger = config.trading.take_profit_percent
        dist_to_sl = sl_trigger - abs(pnl_pct) if pnl_pct < 0 else sl_trigger + pnl_pct
        print(f"    Dist to SL:      {fmt_pct(sl_trigger + pnl_pct) if pnl_pct < 0 else 'N/A (profitable)'} (SL at -{sl_trigger}%)")
        print(f"    Dist to TP:      {fmt_pct(tp_trigger - pnl_pct)} more needed (TP at +{tp_trigger}%)")

        # --- Run ML ensemble ---
        print()
        print(f"    --- ML Model Analysis ---")
        try:
            candles_1h = collector.get_candles(pair, "1h", 300, use_cache=False)
            time.sleep(0.15)
            candles_4h = collector.get_candles(pair, "4h", 100, use_cache=False)
            time.sleep(0.15)
            candles_1d = collector.get_candles(pair, "1d", 60, use_cache=False)
            time.sleep(0.15)

            features_df = feature_engine.calculate_multi_tf_features(
                candles_1h, candles_4h, candles_1d
            )

            if features_df.empty:
                print(f"    WARNING: Could not calculate features for {pair}")
                position_analyses.append({
                    "pair": pair, "side": side, "lever": lever,
                    "pnl_usd": upl, "pnl_pct": pnl_pct,
                    "leveraged_pnl_pct": leveraged_pnl_pct,
                    "model_action": 0, "model_confidence": 0,
                    "assessment": "UNKNOWN - feature calculation failed",
                    "notional": abs(notional_usd),
                })
                continue

            latest_features = features_df.iloc[[-1]]

            portfolio_state = {
                "balance": usdt_balance,
                "position": abs(pos_amt),
                "entry_price": avg_px,
                "current_price": current_price,
                "side": 1 if side == "long" else -1,
            }

            decision = ensemble.get_decision(
                features=latest_features,
                portfolio_state=portfolio_state,
                pair=pair
            )

            xgb_action_s = action_str(decision.xgb_action)
            rl_action_s = action_str(decision.rl_action)
            ensemble_action_s = action_str(decision.action)

            print(f"    XGBoost Signal:  {xgb_action_s} (confidence: {decision.xgb_confidence:.3f})")
            print(f"    RL Agent Signal: {rl_action_s} (confidence: {decision.rl_confidence:.3f})")
            print(f"    -------------------------------------------------")
            print(f"    ENSEMBLE:        {ensemble_action_s} (confidence: {decision.confidence:.3f})")
            print(f"    Reasoning:       {decision.reasoning}")

            assessment = signal_assessment(side, decision.action)
            print(f"    vs Position:     {assessment}")

            # Additional context: what would XGBoost probabilities look like?
            try:
                proba = xgb_model.predict_proba(latest_features[xgb_model.feature_names])
                print(f"    XGB Probas:      SELL={proba[0][0]:.3f}  HOLD={proba[0][1]:.3f}  BUY={proba[0][2]:.3f}")
            except:
                pass

            position_analyses.append({
                "pair": pair, "side": side, "lever": lever,
                "pnl_usd": upl, "pnl_pct": pnl_pct,
                "leveraged_pnl_pct": leveraged_pnl_pct,
                "model_action": decision.action,
                "model_confidence": decision.confidence,
                "xgb_action": decision.xgb_action,
                "xgb_conf": decision.xgb_confidence,
                "rl_action": decision.rl_action,
                "rl_conf": decision.rl_confidence,
                "assessment": assessment,
                "reasoning": decision.reasoning,
                "notional": abs(notional_usd),
                "entry_price": avg_px,
                "current_price": current_price,
                "liq_px": liq_price,
            })

        except Exception as e:
            print(f"    ERROR running model for {pair}: {e}")
            import traceback
            traceback.print_exc()
            position_analyses.append({
                "pair": pair, "side": side, "lever": lever,
                "pnl_usd": upl, "pnl_pct": pnl_pct,
                "leveraged_pnl_pct": leveraged_pnl_pct,
                "model_action": 0, "model_confidence": 0,
                "assessment": f"ERROR: {e}",
                "notional": abs(notional_usd),
            })

        print()

    # 5. Risk Assessment
    print("[5/6] Risk Assessment")
    print("=" * 78)
    print()

    exposure_pct = (total_notional / total_equity * 100) if total_equity > 0 else 0
    print(f"  Total Notional Exposure:  {fmt_usd(total_notional)}")
    print(f"  Total Account Equity:     {fmt_usd(total_equity)}")
    print(f"  Exposure / Equity:        {exposure_pct:.1f}%")
    print(f"  USDT Available (free):    {fmt_usd(usdt_balance)}")
    print(f"  Total Unrealized PnL:     {fmt_usd(total_unrealized_pnl)}")
    pnl_vs_equity = (total_unrealized_pnl / total_equity * 100) if total_equity > 0 else 0
    print(f"  PnL / Equity:             {fmt_pct(pnl_vs_equity)}")
    print()

    # Winning/losing count
    winners = sum(1 for pa in position_analyses if pa.get("pnl_pct", 0) > 0)
    losers = sum(1 for pa in position_analyses if pa.get("pnl_pct", 0) < 0)
    flat = len(position_analyses) - winners - losers
    print(f"  Winning positions:  {winners}")
    print(f"  Losing positions:   {losers}")
    print(f"  Flat positions:     {flat}")
    print()

    # Risk level assessment
    if exposure_pct > 500:
        risk_level = "VERY HIGH"
        risk_note = "Extremely leveraged. A small adverse move could cause major losses."
    elif exposure_pct > 300:
        risk_level = "HIGH"
        risk_note = "Heavy exposure. Monitor closely."
    elif exposure_pct > 150:
        risk_level = "MODERATE"
        risk_note = "Reasonable exposure with leverage."
    elif exposure_pct > 50:
        risk_level = "LOW-MODERATE"
        risk_note = "Conservative positioning."
    else:
        risk_level = "LOW"
        risk_note = "Very light exposure."

    print(f"  Overall Risk Level: {risk_level}")
    print(f"  Note: {risk_note}")

    # Max single-position exposure
    if position_analyses:
        max_pos = max(position_analyses, key=lambda x: x.get("notional", 0))
        max_pct = (max_pos["notional"] / total_equity * 100) if total_equity > 0 else 0
        print(f"  Largest Position:   {max_pos['pair']} at {fmt_usd(max_pos['notional'])} ({max_pct:.1f}% of equity)")
    print()

    # 6. Recommendations
    print("[6/6] Position-by-Position Recommendations")
    print("=" * 78)
    print()

    positions_to_close = []
    positions_to_hold = []
    positions_to_watch = []

    for pa in position_analyses:
        pair = pa["pair"]
        side = pa["side"]
        lever = pa.get("lever", 1)
        pnl_pct = pa.get("pnl_pct", 0)
        leveraged = pa.get("leveraged_pnl_pct", pnl_pct)
        assessment = pa.get("assessment", "")
        model_action = pa.get("model_action", 0)
        model_conf = pa.get("model_confidence", 0)

        recommendation = ""
        reason = ""

        # Check stop-loss
        if pnl_pct <= -config.trading.stop_loss_percent:
            recommendation = "CLOSE (STOP-LOSS)"
            reason = f"Price move {fmt_pct(pnl_pct)} exceeds {config.trading.stop_loss_percent}% stop-loss"
            positions_to_close.append((pair, side, lever, recommendation, reason))
        # Check take-profit
        elif pnl_pct >= config.trading.take_profit_percent:
            recommendation = "CLOSE (TAKE PROFIT)"
            reason = f"Price move {fmt_pct(pnl_pct)} hit {config.trading.take_profit_percent}% target"
            positions_to_close.append((pair, side, lever, recommendation, reason))
        # Model disagrees with position direction
        elif "DISAGREES" in assessment and model_conf >= 0.55:
            recommendation = "CONSIDER CLOSING"
            reason = f"ML model disagrees (conf: {model_conf:.2f}): {pa.get('reasoning', '')}"
            positions_to_close.append((pair, side, lever, recommendation, reason))
        # Model agrees
        elif "AGREES" in assessment:
            recommendation = "HOLD"
            reason = f"ML model confirms direction (conf: {model_conf:.2f})"
            positions_to_hold.append((pair, side, lever, recommendation, reason))
        # Model says HOLD / neutral
        elif model_action == 0:
            if abs(pnl_pct) < 1:
                recommendation = "WATCH"
                reason = f"Model neutral, small PnL ({fmt_pct(pnl_pct)})"
                positions_to_watch.append((pair, side, lever, recommendation, reason))
            elif pnl_pct > 0:
                recommendation = "HOLD (PROFITABLE)"
                reason = f"Model neutral but profitable ({fmt_pct(pnl_pct)})"
                positions_to_hold.append((pair, side, lever, recommendation, reason))
            else:
                recommendation = "WATCH CLOSELY"
                reason = f"Model neutral, losing ({fmt_pct(pnl_pct)})"
                positions_to_watch.append((pair, side, lever, recommendation, reason))
        else:
            recommendation = "WATCH"
            reason = assessment
            positions_to_watch.append((pair, side, lever, recommendation, reason))

        print(f"  {pair} ({side_label(side)} {lever}x)")
        print(f"    PnL: {fmt_usd(pa['pnl_usd'])} | Price: {fmt_pct(pnl_pct)} | Leveraged: {fmt_pct(leveraged)}")
        print(f"    >> {recommendation}")
        print(f"       {reason}")
        print()

    # Final Summary
    print("=" * 78)
    print("  FINAL SUMMARY")
    print("=" * 78)
    print()
    print(f"  Account Equity:      {fmt_usd(total_equity)}")
    print(f"  USDT Free:           {fmt_usd(usdt_balance)}")
    print(f"  Open Positions:      {len(position_analyses)} futures")
    print(f"  Total Notional:      {fmt_usd(total_notional)}")
    print(f"  Unrealized PnL:      {fmt_usd(total_unrealized_pnl)} ({fmt_pct(pnl_vs_equity)} of equity)")
    print(f"  Risk Level:          {risk_level}")
    print(f"  Effective Leverage:  {exposure_pct/100:.1f}x")
    print()
    print(f"  Positions to HOLD:   {len(positions_to_hold)}")
    print(f"  Positions to CLOSE:  {len(positions_to_close)}")
    print(f"  Positions to WATCH:  {len(positions_to_watch)}")
    print()

    if positions_to_close:
        print("  ** ACTION ITEMS:")
        for pair, side, lever, rec, reason in positions_to_close:
            print(f"     [{rec}] {pair} ({side_label(side)} {lever}x)")
            print(f"       Reason: {reason}")
        print()

    if positions_to_hold:
        print("  HOLDING (model agrees or profitable):")
        for pair, side, lever, rec, reason in positions_to_hold:
            print(f"     {pair} ({side_label(side)} {lever}x) - {reason}")
        print()

    if positions_to_watch:
        print("  WATCHING (model neutral, small/negative PnL):")
        for pair, side, lever, rec, reason in positions_to_watch:
            print(f"     {pair} ({side_label(side)} {lever}x) - {reason}")
        print()

    # Overall assessment
    print("  OVERALL ASSESSMENT:")
    all_shorts = all(pa["side"] == "short" for pa in position_analyses)
    if all_shorts:
        print("  All positions are SHORT - the bot is bearish across the board.")
    all_longs = all(pa["side"] == "long" for pa in position_analyses)
    if all_longs:
        print("  All positions are LONG - the bot is bullish across the board.")

    model_disagree_count = sum(1 for pa in position_analyses if "DISAGREES" in pa.get("assessment", ""))
    model_neutral_count = sum(1 for pa in position_analyses if "NEUTRAL" in pa.get("assessment", ""))
    model_agree_count = sum(1 for pa in position_analyses if "AGREES" in pa.get("assessment", ""))

    print(f"  Model alignment: {model_agree_count} agree, {model_neutral_count} neutral, {model_disagree_count} disagree")

    if model_neutral_count == len(position_analyses):
        print("  NOTE: The ensemble is NEUTRAL on ALL positions (models disagree with each other).")
        print("  This means XGBoost and RL have different views on market direction.")
        print("  The bot will not open new positions but existing ones are managed by stop-loss/take-profit.")

    print()
    print("=" * 78)
    print("  Analysis complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
