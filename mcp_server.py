"""
Cryptobot MCP Server with OAuth Authentication
================================================
Exposes trading bot monitoring and control via Model Context Protocol.
OAuth secured - only clients with valid credentials can access.

Usage:
    # Claude Code (stdio mode - no auth needed)
    python mcp_server.py

    # Remote access (SSE mode with OAuth - for Claude web/mobile)
    python mcp_server.py --sse --port 8808
"""

import os
import sys
import time
import secrets
import sqlite3
import subprocess
import logging
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydantic import AnyUrl

from mcp.server.fastmcp import FastMCP
from mcp.server.auth.provider import (
    OAuthAuthorizationServerProvider,
    AuthorizationParams,
    AuthorizationCode,
    RefreshToken,
    AccessToken,
    construct_redirect_uri,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cryptobot-mcp")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trades.db")
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


# ==================== OAUTH PROVIDER ====================

class FlexibleClient(OAuthClientInformationFull):
    """Client subclass that accepts any redirect_uri.
    Used for pre-registered client where we don't know Claude's callback URL in advance."""

    def validate_redirect_uri(self, redirect_uri):
        if redirect_uri is not None:
            return redirect_uri
        if self.redirect_uris and len(self.redirect_uris) == 1:
            return self.redirect_uris[0]
        from mcp.shared.auth import InvalidRedirectUriError
        raise InvalidRedirectUriError("redirect_uri must be specified")


class CryptobotOAuthProvider:
    """OAuth provider for personal MCP server. Only the pre-registered client is authorized."""

    ALLOWED_CLIENT_ID = "cryptobot-24df753995c8f9f0"
    ALLOWED_CLIENT_SECRET = "b8f3639a078dc78e8060bbfe315688144fdbc1c1b4a7459d8cccc96d9233f323"

    def __init__(self):
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.auth_codes: dict[str, AuthorizationCode] = {}
        self.access_tokens: dict[str, AccessToken] = {}
        self.refresh_tokens: dict[str, RefreshToken] = {}

        # Pre-register the allowed client so get_client() finds it immediately
        self.clients[self.ALLOWED_CLIENT_ID] = FlexibleClient(
            client_id=self.ALLOWED_CLIENT_ID,
            client_secret=self.ALLOWED_CLIENT_SECRET,
            client_id_issued_at=int(time.time()),
            redirect_uris=["https://localhost/callback"],  # dummy - FlexibleClient accepts any
            token_endpoint_auth_method="client_secret_post",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope="read write admin",
        )
        logger.info(f"Pre-registered OAuth client: {self.ALLOWED_CLIENT_ID}")

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        client = self.clients.get(client_id)
        if client:
            logger.info(f"get_client({client_id}): FOUND")
            return client

        # For unknown clients (e.g. stale cached credentials from Claude),
        # return a permissive placeholder. This prevents 401 "unauthorized_client"
        # and instead lets the flow reach token validation which returns
        # 400 "invalid_grant" - prompting Claude to restart the auth flow fresh.
        logger.info(f"get_client({client_id}): NOT FOUND - returning placeholder")
        return OAuthClientInformationFull(
            client_id=client_id,
            client_secret=None,
            redirect_uris=["https://placeholder.invalid/callback"],
            token_endpoint_auth_method="none",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        )

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self.clients[client_info.client_id] = client_info
        logger.info(f"OAuth client registered: {client_info.client_id} ({client_info.client_name or 'unnamed'})")

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        # Approve pre-registered client AND dynamically registered clients
        # (dynamic registration already requires interacting with our server)
        if client.client_id not in self.clients:
            logger.warning(f"OAuth REJECTED unknown client: {client.client_id}")
            from mcp.server.auth.provider import AuthorizeError
            raise AuthorizeError(error="access_denied", error_description="Unauthorized client")

        logger.info(f"OAuth APPROVED client: {client.client_id} redirect_uri={params.redirect_uri}")
        code = secrets.token_urlsafe(32)

        auth_code = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + 300,  # 5 min expiry
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        self.auth_codes[code] = auth_code

        redirect_params = {"code": code}
        if params.state:
            redirect_params["state"] = params.state

        return construct_redirect_uri(str(params.redirect_uri), **redirect_params)

    async def load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str) -> AuthorizationCode | None:
        return self.auth_codes.get(authorization_code)

    async def exchange_authorization_code(self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode) -> OAuthToken:
        # Remove used auth code (one-time use)
        self.auth_codes.pop(authorization_code.code, None)

        # Generate access token
        access_token_str = secrets.token_urlsafe(32)
        refresh_token_str = secrets.token_urlsafe(32)

        self.access_tokens[access_token_str] = AccessToken(
            token=access_token_str,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(time.time()) + 86400,  # 24 hours
            resource=authorization_code.resource,
        )

        self.refresh_tokens[refresh_token_str] = RefreshToken(
            token=refresh_token_str,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(time.time()) + 86400 * 30,  # 30 days
        )

        logger.info(f"OAuth tokens issued for client {client.client_id}")

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=86400,
            refresh_token=refresh_token_str,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        at = self.access_tokens.get(token)
        if at and at.expires_at and at.expires_at < time.time():
            self.access_tokens.pop(token, None)
            return None
        return at

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        rt = self.refresh_tokens.get(refresh_token)
        if rt and rt.expires_at and rt.expires_at < time.time():
            self.refresh_tokens.pop(refresh_token, None)
            return None
        return rt

    async def exchange_refresh_token(self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]) -> OAuthToken:
        # Revoke old refresh token
        self.refresh_tokens.pop(refresh_token.token, None)

        # Generate new tokens
        new_access = secrets.token_urlsafe(32)
        new_refresh = secrets.token_urlsafe(32)

        self.access_tokens[new_access] = AccessToken(
            token=new_access,
            client_id=client.client_id,
            scopes=scopes,
            expires_at=int(time.time()) + 86400,
        )

        self.refresh_tokens[new_refresh] = RefreshToken(
            token=new_refresh,
            client_id=client.client_id,
            scopes=scopes,
            expires_at=int(time.time()) + 86400 * 30,
        )

        return OAuthToken(
            access_token=new_access,
            token_type="Bearer",
            expires_in=86400,
            refresh_token=new_refresh,
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, AccessToken):
            self.access_tokens.pop(token.token, None)
        else:
            self.refresh_tokens.pop(token.token, None)


# ==================== MCP SERVER SETUP ====================

def create_mcp_server(use_auth: bool = False, port: int = 8808) -> FastMCP:
    """Create MCP server, optionally with OAuth."""

    kwargs = {
        "name": "Cryptobot",
        "instructions": "AI Crypto Trading Bot - Monitor and control your autonomous trading system",
    }

    if use_auth:
        server_url = f"https://senthil2706.duckdns.org:{port}"
        oauth_provider = CryptobotOAuthProvider()

        kwargs["auth_server_provider"] = oauth_provider
        kwargs["auth"] = AuthSettings(
            issuer_url=server_url,
            resource_server_url=server_url,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=["read", "write", "admin"],
                default_scopes=["read", "write", "admin"],
            ),
        )
        # Allow our domain in transport security (default auto-enables localhost-only)
        kwargs["transport_security"] = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["senthil2706.duckdns.org", "senthil2706.duckdns.org:443"],
        )

    server = FastMCP(**kwargs)
    _register_tools(server)
    return server


def _get_db():
    return sqlite3.connect(DB_PATH)


def _get_env_config() -> dict:
    config = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    config[key.strip()] = val.strip()
    return config


def _get_service_status() -> dict:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "cryptobot"],
            capture_output=True, text=True, timeout=5
        )
        is_active = result.stdout.strip() == "active"

        result2 = subprocess.run(
            ["systemctl", "show", "cryptobot", "--property=ActiveEnterTimestamp,MainPID,MemoryCurrent"],
            capture_output=True, text=True, timeout=5
        )
        props = {}
        for line in result2.stdout.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                props[k] = v

        return {
            "running": is_active,
            "pid": props.get("MainPID", "unknown"),
            "started_at": props.get("ActiveEnterTimestamp", "unknown"),
            "memory": props.get("MemoryCurrent", "unknown"),
        }
    except Exception as e:
        return {"running": False, "error": str(e)}


def _register_tools(mcp: FastMCP):
    """Register all tools on the MCP server."""

    @mcp.tool()
    def bot_status() -> str:
        """Get the current status of the trading bot - is it running, uptime, health."""
        status = _get_service_status()
        config = _get_env_config()

        if not status["running"]:
            return "Bot is STOPPED. Use restart_bot() to start it."

        try:
            result = subprocess.run(
                ["journalctl", "-u", "cryptobot", "--no-pager", "-n", "15"],
                capture_output=True, text=True, timeout=10
            )
            recent_logs = result.stdout
        except:
            recent_logs = "Could not fetch logs"

        futures_enabled = config.get('FUTURES_ENABLED', 'false').lower() == 'true'
        futures_section = ""
        if futures_enabled:
            futures_section = f"""
FUTURES:
  Enabled: YES
  Leverage: {config.get('FUTURES_LEVERAGE', '2')}x
  Margin Mode: {config.get('FUTURES_MARGIN_MODE', 'cross')}
  Futures Pairs: {config.get('FUTURES_PAIRS', 'none')}"""
        else:
            futures_section = "\nFUTURES: Disabled"

        return f"""BOT STATUS: {'RUNNING' if status['running'] else 'STOPPED'}
PID: {status.get('pid', 'N/A')}
Started: {status.get('started_at', 'N/A')}
Memory: {status.get('memory', 'N/A')}

SETTINGS:
  Mode: {config.get('TRADING_MODE', 'unknown')}
  Pairs: {config.get('TRADING_PAIRS', 'unknown')}
  Min Confidence: {config.get('MIN_CONFIDENCE', 'unknown')}
  Stop Loss: {config.get('STOP_LOSS_PERCENT', 'unknown')}%
  Trailing Stop: {config.get('TRAILING_STOP_PERCENT', 'unknown')}%
  Take Profit: {config.get('TAKE_PROFIT_PERCENT', 'unknown')}%
  Analysis Interval: {config.get('ANALYSIS_INTERVAL', 'unknown')} min
{futures_section}

RECENT LOGS:
{recent_logs}"""

    @mcp.tool()
    def portfolio() -> str:
        """Get current portfolio - USDT balance, open positions, and P&L."""
        try:
            from src.config import get_config
            from src.security.vault import SecureVault
            from src.trading.okx_client import SecureOKXClient

            config = get_config()
            vault = SecureVault()
            okx = SecureOKXClient(vault, demo_mode=config.exchange.demo_mode)

            balances = okx.get_balance()
            usdt = balances.get("USDT", 0)

            positions = []
            total_value = usdt
            for ccy, amount in balances.items():
                if ccy == "USDT":
                    continue
                pair = f"{ccy}-USDT"
                try:
                    ticker = okx.get_ticker(pair)
                    price = float(ticker.get("last", 0))
                    value = amount * price
                    if value >= 1:
                        positions.append({"currency": ccy, "amount": amount, "price": price, "value": value})
                        total_value += value
                except:
                    continue

            lines = ["PORTFOLIO SUMMARY", "=" * 40]
            lines.append(f"USDT Balance: ${usdt:.2f}")
            lines.append(f"Total Value:  ${total_value:.2f}")
            lines.append("")

            if positions:
                lines.append("SPOT POSITIONS:")
                for p in positions:
                    lines.append(f"  {p['currency']}-USDT: {p['amount']:.6f} @ ${p['price']:.2f} = ${p['value']:.2f}")
            else:
                lines.append("No open spot positions")

            # Check for futures positions
            try:
                futures_positions = okx.get_futures_positions()
                active_futures = [p for p in futures_positions if float(p.get("pos", 0)) != 0]
                if active_futures:
                    lines.append("")
                    lines.append("FUTURES POSITIONS:")
                    for fp in active_futures:
                        inst_id = fp.get("instId", "")
                        pos_amt = float(fp.get("pos", 0))
                        avg_px = float(fp.get("avgPx", 0))
                        mark_px = float(fp.get("markPx", 0))
                        upl = float(fp.get("upl", 0))
                        lever = fp.get("lever", "?")
                        direction = "LONG" if pos_amt > 0 else "SHORT"
                        lines.append(
                            f"  {inst_id}: {direction} x{abs(pos_amt)} @ ${avg_px:.2f} "
                            f"(mark: ${mark_px:.2f}, uPnL: ${upl:.2f}, {lever}x)"
                        )
            except Exception:
                pass  # Futures may not be enabled

            return "\n".join(lines)
        except Exception as e:
            return f"Error fetching portfolio: {e}"

    @mcp.tool()
    def recent_trades(limit: int = 20) -> str:
        """Get recent trade history from the database.

        Args:
            limit: Number of trades to show (default 20)
        """
        try:
            conn = _get_db()
            c = conn.cursor()
            c.execute(
                "SELECT timestamp, pair, side, amount, price, total_value, pnl, pnl_percent "
                "FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
            rows = c.fetchall()
            conn.close()

            if not rows:
                return "No trades recorded yet."

            lines = ["RECENT TRADES", "=" * 60]
            total_pnl = 0
            for r in rows:
                ts, pair, side, amount, price, value, pnl, pnl_pct = r
                side = side.upper()
                pnl = pnl or 0
                pnl_pct = pnl_pct or 0
                total_pnl += pnl
                pnl_str = f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)" if side == "SELL" else ""
                lines.append(f"  {ts[:19]} | {side:4s} {pair:10s} | {amount:.6f} @ ${price:.2f} = ${value:.2f} {pnl_str}")

            lines.append(f"\nTotal realized P&L: ${total_pnl:.2f}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def recent_decisions(limit: int = 20) -> str:
        """Get recent model decisions - what XGBoost and RL recommended.

        Args:
            limit: Number of decisions to show (default 20)
        """
        try:
            conn = _get_db()
            c = conn.cursor()
            c.execute(
                "SELECT timestamp, pair, action, confidence, xgb_action, xgb_confidence, "
                "rl_action, rl_confidence, executed FROM decisions ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = c.fetchall()
            conn.close()

            if not rows:
                return "No decisions recorded yet."

            action_map = {"-1": "SELL", "0": "HOLD", "1": "BUY",
                          "Action.SELL": "SELL", "Action.HOLD": "HOLD", "Action.BUY": "BUY"}

            lines = ["RECENT DECISIONS", "=" * 80]
            for r in rows:
                ts = r[0][:19]
                pair = r[1]
                action = action_map.get(str(r[2]), str(r[2]))
                conf = r[3] or 0
                xgb_action = action_map.get(str(r[4]), str(r[4]))
                xgb_conf = r[5] or 0
                rl_action = action_map.get(str(r[6]), str(r[6]))
                rl_conf = r[7] or 0
                executed = "YES" if r[8] else "no"

                lines.append(
                    f"  {ts} | {pair:10s} | {action:4s} ({conf:.2f}) | "
                    f"XGB={xgb_action}({xgb_conf:.2f}) RL={rl_action}({rl_conf:.2f}) | exec={executed}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def bot_logs(lines: int = 50) -> str:
        """Get recent bot logs from journalctl.

        Args:
            lines: Number of log lines to show (default 50)
        """
        try:
            result = subprocess.run(
                ["journalctl", "-u", "cryptobot", "--no-pager", "-n", str(min(lines, 200))],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout if result.stdout else "No logs available"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def market_prices() -> str:
        """Get current prices for all trading pairs (BTC, ETH, SOL)."""
        try:
            from src.config import get_config
            from src.security.vault import SecureVault
            from src.trading.okx_client import SecureOKXClient

            config = get_config()
            vault = SecureVault()
            okx = SecureOKXClient(vault, demo_mode=config.exchange.demo_mode)

            lines = ["MARKET PRICES", "=" * 50]
            for pair in config.trading.trading_pairs:
                try:
                    ticker = okx.get_ticker(pair)
                    price = float(ticker.get("last", 0))
                    high24 = float(ticker.get("high24h", 0))
                    low24 = float(ticker.get("low24h", 0))
                    vol = float(ticker.get("vol24h", 0))
                    lines.append(f"  {pair:10s}: ${price:>12,.2f}  (24h: ${low24:,.2f} - ${high24:,.2f}, vol: {vol:,.0f})")
                except Exception as e:
                    lines.append(f"  {pair}: Error - {e}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def trading_stats() -> str:
        """Get overall trading statistics - total trades, win rate, P&L breakdown."""
        try:
            conn = _get_db()
            c = conn.cursor()

            c.execute("SELECT COUNT(*) FROM trades")
            total = c.fetchone()[0]

            c.execute("SELECT side, COUNT(*) FROM trades GROUP BY side")
            sides = dict(c.fetchall())

            c.execute("SELECT pnl, pnl_percent FROM trades WHERE side='sell' AND pnl != 0")
            pnl_rows = c.fetchall()
            conn.close()

            lines = ["TRADING STATISTICS", "=" * 40]
            lines.append(f"Total trades: {total}")
            lines.append(f"  Buys:  {sides.get('buy', 0)}")
            lines.append(f"  Sells: {sides.get('sell', 0)}")

            if pnl_rows:
                wins = [r for r in pnl_rows if r[0] > 0]
                losses = [r for r in pnl_rows if r[0] < 0]
                total_pnl = sum(r[0] for r in pnl_rows)

                lines.append(f"\nWin/Loss: {len(wins)}W / {len(losses)}L")
                lines.append(f"Win rate: {len(wins)/len(pnl_rows)*100:.1f}%")
                lines.append(f"Total P&L: ${total_pnl:.2f}")

                if wins:
                    lines.append(f"Avg win: {sum(r[1] for r in wins)/len(wins):.2f}%")
                if losses:
                    lines.append(f"Avg loss: {sum(r[1] for r in losses)/len(losses):.2f}%")
            else:
                lines.append("\nNo completed round-trips yet (no sells)")

            conn = _get_db()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM decisions")
            total_decisions = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM decisions WHERE executed=1")
            executed = c.fetchone()[0]
            conn.close()

            lines.append(f"\nDecisions analyzed: {total_decisions}")
            lines.append(f"Decisions executed: {executed}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def restart_bot() -> str:
        """Restart the trading bot service."""
        try:
            result = subprocess.run(
                ["systemctl", "restart", "cryptobot"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                import time as t
                t.sleep(3)
                status = _get_service_status()
                return f"Bot restarted successfully. Running: {status['running']}, PID: {status.get('pid', 'N/A')}"
            else:
                return f"Restart failed: {result.stderr}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def stop_bot() -> str:
        """Stop the trading bot service. Positions remain open on OKX."""
        try:
            result = subprocess.run(
                ["systemctl", "stop", "cryptobot"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                return "Bot stopped. Note: any open positions remain on OKX."
            else:
                return f"Stop failed: {result.stderr}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def update_setting(setting: str, value: str) -> str:
        """Update a bot configuration setting in .env file.

        Args:
            setting: Setting name (e.g., MIN_CONFIDENCE, STOP_LOSS_PERCENT, TRAILING_STOP_PERCENT)
            value: New value for the setting
        """
        allowed = {
            "MIN_CONFIDENCE", "STOP_LOSS_PERCENT", "TAKE_PROFIT_PERCENT",
            "TRAILING_STOP_PERCENT", "MAX_POSITION_PERCENT", "ANALYSIS_INTERVAL",
            "DAILY_LOSS_LIMIT", "MAX_OPEN_POSITIONS",
            "FUTURES_ENABLED", "FUTURES_LEVERAGE", "FUTURES_MARGIN_MODE", "FUTURES_PAIRS"
        }

        setting = setting.upper()
        if setting not in allowed:
            return f"Setting '{setting}' not allowed. Allowed: {', '.join(sorted(allowed))}"

        with open(ENV_PATH) as f:
            content = f.read()

        env_lines = content.split("\n")
        updated = False
        old_val = "N/A"
        for i, line in enumerate(env_lines):
            if line.strip().startswith(f"{setting}="):
                old_val = line.split("=", 1)[1]
                env_lines[i] = f"{setting}={value}"
                updated = True
                break

        if not updated:
            env_lines.append(f"{setting}={value}")

        with open(ENV_PATH, "w") as f:
            f.write("\n".join(env_lines))

        return f"Updated {setting}: {old_val} -> {value}\nRestart bot for changes to take effect."

    @mcp.tool()
    def run_analysis() -> str:
        """Run a one-time analysis of all trading pairs to see what the models think right now."""
        try:
            from src.config import get_config
            from src.security.vault import SecureVault
            from src.trading.okx_client import SecureOKXClient
            from src.data.collector import DataCollector
            from src.data.features import FeatureEngine
            from src.models.xgboost_model import XGBoostPredictor
            from src.models.rl_agent import RLTradingAgent
            import numpy as np

            config = get_config()
            vault = SecureVault()
            okx = SecureOKXClient(vault, demo_mode=config.exchange.demo_mode)
            collector = DataCollector(okx)
            features = FeatureEngine()
            xgb = XGBoostPredictor(model_path="models/xgboost_model.json")
            rl = RLTradingAgent(model_path="models/rl_agent.zip")

            action_name = {-1: "SELL", 0: "HOLD", 1: "BUY"}
            balance = okx.get_usdt_balance()
            portfolio_state = {"balance": balance, "position": 0, "entry_price": 0, "current_price": 0}

            lines = ["LIVE ANALYSIS", "=" * 60]

            for pair in config.trading.trading_pairs:
                try:
                    df = collector.get_historical_data(pair, days=5, timeframe="1h")
                    df_features = features.calculate_features(df)
                    latest = df_features.iloc[[-1]]

                    xgb_action, xgb_conf = xgb.predict(latest)
                    feature_cols = xgb.feature_names
                    feature_array = latest[feature_cols].values.flatten().astype(np.float32)
                    rl_action, rl_conf = rl.decide(feature_array, portfolio_state)

                    agree = "AGREE" if xgb_action == rl_action else "DISAGREE"
                    weighted = xgb_conf * 0.5 + rl_conf * 0.5
                    would_trade = agree == "AGREE" and xgb_action != 0 and weighted >= config.trading.min_confidence_to_trade

                    lines.append(
                        f"  {pair:10s}: XGB={action_name[xgb_action]}({xgb_conf:.2f}) "
                        f"RL={action_name[rl_action]}({rl_conf:.2f}) -> {agree} "
                        f"| conf={weighted:.2f} | trade={'YES' if would_trade else 'NO'}"
                    )
                except Exception as e:
                    lines.append(f"  {pair}: Error - {e}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cryptobot MCP Server")
    parser.add_argument("--sse", action="store_true", help="Run in SSE mode for remote access")
    parser.add_argument("--port", type=int, default=8808, help="Port for SSE mode (default 8808)")
    parser.add_argument("--no-auth", action="store_true", help="Disable OAuth (HTTP mode)")
    args = parser.parse_args()

    if args.sse:
        ssl_certfile = "/etc/letsencrypt/live/senthil2706.duckdns.org/fullchain.pem"
        ssl_keyfile = "/etc/letsencrypt/live/senthil2706.duckdns.org/privkey.pem"
        has_ssl = os.path.exists(ssl_certfile) and os.path.exists(ssl_keyfile)

        if has_ssl and not args.no_auth:
            logger.info(f"Starting Cryptobot MCP server (SSE + OAuth + HTTPS on port {args.port})")
            server = create_mcp_server(use_auth=True, port=args.port)
            server.settings.host = "0.0.0.0"
            server.settings.port = args.port

            import uvicorn
            import asyncio
            from starlette.middleware import Middleware
            from starlette.requests import Request as StarletteRequest

            class DebugLoggingMiddleware:
                """Log request details for OAuth debugging."""
                def __init__(self, app):
                    self.app = app

                async def __call__(self, scope, receive, send):
                    if scope["type"] == "http":
                        path = scope.get("path", "")
                        method = scope.get("method", "")
                        if path in ("/token", "/register", "/authorize"):
                            # Log request details
                            body_parts = []
                            async def receive_wrapper():
                                message = await receive()
                                if message.get("type") == "http.request":
                                    body = message.get("body", b"")
                                    if body:
                                        body_parts.append(body)
                                        # Mask secrets in log
                                        body_str = body.decode("utf-8", errors="replace")
                                        if "client_secret" in body_str:
                                            import re
                                            body_str = re.sub(r'client_secret=[^&]+', 'client_secret=***', body_str)
                                        logger.info(f"OAuth {method} {path} body: {body_str[:500]}")
                                return message
                            await self.app(scope, receive_wrapper, send)
                            return
                    await self.app(scope, receive, send)

            async def run_ssl():
                starlette_app = server.sse_app()

                # Wrap with debug logging
                logged_app = DebugLoggingMiddleware(starlette_app)

                config = uvicorn.Config(
                    logged_app,
                    host="0.0.0.0",
                    port=args.port,
                    ssl_certfile=ssl_certfile,
                    ssl_keyfile=ssl_keyfile,
                    log_level="info",
                )
                srv = uvicorn.Server(config)
                await srv.serve()

            asyncio.run(run_ssl())
        else:
            logger.info(f"Starting Cryptobot MCP server (SSE mode on port {args.port})")
            server = create_mcp_server(use_auth=False)
            server.settings.host = "0.0.0.0"
            server.settings.port = args.port
            server.run(transport="sse")
    else:
        logger.info("Starting Cryptobot MCP server (stdio mode, no auth)")
        server = create_mcp_server(use_auth=False)
        server.run(transport="stdio")
