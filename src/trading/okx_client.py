"""
Secure OKX Client
=================
Secure wrapper for OKX exchange API.
All credentials loaded from encrypted vault.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SecureOKXClient:
    """
    Secure OKX API client.

    Features:
    - Loads credentials from encrypted vault
    - Implements rate limiting
    - Comprehensive error handling
    - Supports both live and demo trading
    """

    def __init__(self, vault, demo_mode: bool = False):
        """
        Initialize OKX client.

        Args:
            vault: SecureVault instance with encrypted API keys
            demo_mode: If True, use demo trading (paper trading)
        """
        self.vault = vault
        self.demo_mode = demo_mode
        self._client = None
        self._market_api = None
        self._trade_api = None
        self._account_api = None
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

        self._initialize_client()

    def _initialize_client(self):
        """Initialize OKX SDK clients."""
        try:
            import okx.MarketData as MarketData
            import okx.Trade as Trade
            import okx.Account as Account

            # Load decrypted API keys
            keys = self.vault.load_api_keys()

            # Flag: "0" for live, "1" for demo
            flag = "1" if self.demo_mode else "0"

            # Initialize API clients
            self._market_api = MarketData.MarketAPI(flag=flag)

            self._trade_api = Trade.TradeAPI(
                api_key=keys["api_key"],
                api_secret_key=keys["secret_key"],
                passphrase=keys["passphrase"],
                use_server_time=False,
                flag=flag
            )

            self._account_api = Account.AccountAPI(
                api_key=keys["api_key"],
                api_secret_key=keys["secret_key"],
                passphrase=keys["passphrase"],
                use_server_time=False,
                flag=flag
            )

            mode = "DEMO" if self.demo_mode else "LIVE"
            logger.info(f"OKX client initialized in {mode} mode")

        except ImportError:
            logger.error("python-okx not installed! Run: pip install python-okx")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OKX client: {e}")
            raise

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _handle_response(self, response: dict, operation: str) -> dict:
        """Handle API response and errors."""
        if response is None:
            raise Exception(f"{operation} returned None response")

        code = response.get("code", "")
        msg = response.get("msg", "")

        if code != "0":
            error_msg = f"{operation} failed: [{code}] {msg}"
            logger.error(error_msg)
            raise Exception(error_msg)

        return response.get("data", response)

    # ==================== Market Data ====================

    def get_ticker(self, inst_id: str) -> Dict[str, Any]:
        """
        Get ticker data for a trading pair.

        Args:
            inst_id: Trading pair (e.g., "BTC-USDT")

        Returns:
            Dict with last, bid, ask, high24h, low24h, vol24h, etc.
        """
        self._rate_limit()
        response = self._market_api.get_ticker(instId=inst_id)
        data = self._handle_response(response, "get_ticker")
        return data[0] if data else {}

    def get_candles(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 100,
        after: Optional[str] = None
    ) -> List[List]:
        """
        Get candlestick data.

        Args:
            inst_id: Trading pair
            bar: Timeframe (1m, 5m, 15m, 30m, 1H, 4H, 1D, etc.)
            limit: Number of candles (max 300)
            after: Pagination - get candles before this timestamp

        Returns:
            List of candles [timestamp, open, high, low, close, vol, ...]
        """
        self._rate_limit()
        params = {"instId": inst_id, "bar": bar, "limit": str(min(limit, 300))}
        if after:
            params["after"] = str(after)

        response = self._market_api.get_candlesticks(**params)
        return self._handle_response(response, "get_candles")

    def get_orderbook(self, inst_id: str, sz: int = 20) -> Dict[str, List]:
        """
        Get order book.

        Returns:
            Dict with 'bids' and 'asks' lists
        """
        self._rate_limit()
        response = self._market_api.get_orderbook(instId=inst_id, sz=str(sz))
        data = self._handle_response(response, "get_orderbook")
        return data[0] if data else {"bids": [], "asks": []}

    # ==================== Account ====================

    def get_balance(self, currency: str = "") -> Dict[str, float]:
        """
        Get account balance.

        Args:
            currency: Specific currency (empty for all)

        Returns:
            Dict with currency balances {currency: available_balance}
        """
        self._rate_limit()
        params = {}
        if currency:
            params["ccy"] = currency

        response = self._account_api.get_account_balance(**params)
        data = self._handle_response(response, "get_balance")

        balances = {}
        if data and len(data) > 0:
            details = data[0].get("details", [])
            for item in details:
                ccy = item.get("ccy", "")
                avail = float(item.get("availBal", 0))
                if avail > 0:
                    balances[ccy] = avail

        return balances

    def get_usdt_balance(self) -> float:
        """Get available USDT balance."""
        balances = self.get_balance("USDT")
        return balances.get("USDT", 0.0)

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.

        Returns:
            List of position dicts with inst_id, size, avg_price, pnl, etc.
        """
        self._rate_limit()
        response = self._account_api.get_positions()
        return self._handle_response(response, "get_positions")

    # ==================== Trading ====================

    def get_instrument_info(self, inst_id: str) -> Dict[str, Any]:
        """
        Get trading pair info (min size, tick size, etc.)

        Returns:
            Dict with minSz, lotSz, tickSz, etc.
        """
        self._rate_limit()
        response = self._market_api.get_instruments(instType="SPOT", instId=inst_id)
        data = self._handle_response(response, "get_instrument_info")
        return data[0] if data else {}

    def get_min_order_size(self, inst_id: str) -> float:
        """Get minimum order size for a trading pair."""
        info = self.get_instrument_info(inst_id)
        return float(info.get("minSz", 0))

    def place_market_buy(
        self,
        inst_id: str,
        usdt_amount: float,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place market buy order.

        Args:
            inst_id: Trading pair (e.g., "BTC-USDT")
            usdt_amount: Amount in USDT to spend
            client_order_id: Optional custom order ID

        Returns:
            Order result dict
        """
        self._rate_limit()

        params = {
            "instId": inst_id,
            "tdMode": "cash",  # Spot trading
            "side": "buy",
            "ordType": "market",
            "sz": str(usdt_amount),
            "tgtCcy": "quote_ccy"  # Size is in USDT
        }

        if client_order_id:
            params["clOrdId"] = client_order_id

        logger.info(f"Placing market BUY: {inst_id} for ${usdt_amount:.2f}")
        response = self._trade_api.place_order(**params)
        result = self._handle_response(response, "place_market_buy")

        if result:
            order_id = result[0].get("ordId", "")
            logger.info(f"Order placed successfully: {order_id}")

        return result[0] if result else {}

    def place_market_sell(
        self,
        inst_id: str,
        amount: float,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place market sell order.

        Args:
            inst_id: Trading pair (e.g., "BTC-USDT")
            amount: Amount of crypto to sell
            client_order_id: Optional custom order ID

        Returns:
            Order result dict
        """
        self._rate_limit()

        params = {
            "instId": inst_id,
            "tdMode": "cash",
            "side": "sell",
            "ordType": "market",
            "sz": str(amount),
            "tgtCcy": "base_ccy"  # Size is in base currency (e.g., BTC)
        }

        if client_order_id:
            params["clOrdId"] = client_order_id

        logger.info(f"Placing market SELL: {inst_id}, amount: {amount}")
        response = self._trade_api.place_order(**params)
        result = self._handle_response(response, "place_market_sell")

        if result:
            order_id = result[0].get("ordId", "")
            logger.info(f"Order placed successfully: {order_id}")

        return result[0] if result else {}

    def get_order(self, inst_id: str, order_id: str) -> Dict[str, Any]:
        """Get order details."""
        self._rate_limit()
        response = self._trade_api.get_order(instId=inst_id, ordId=order_id)
        data = self._handle_response(response, "get_order")
        return data[0] if data else {}

    def cancel_order(self, inst_id: str, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        self._rate_limit()
        response = self._trade_api.cancel_order(instId=inst_id, ordId=order_id)
        return self._handle_response(response, "cancel_order")

    def get_order_history(
        self,
        inst_id: str = "",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent order history."""
        self._rate_limit()
        params = {"instType": "SPOT", "limit": str(limit)}
        if inst_id:
            params["instId"] = inst_id

        response = self._trade_api.get_orders_history(**params)
        return self._handle_response(response, "get_order_history")

    # ==================== Utility ====================

    def test_connection(self) -> bool:
        """Test API connection and credentials."""
        try:
            # Test public API
            ticker = self.get_ticker("BTC-USDT")
            if not ticker:
                return False

            # Test private API
            balance = self.get_balance()

            logger.info("OKX connection test successful")
            return True

        except Exception as e:
            logger.error(f"OKX connection test failed: {e}")
            return False

    def get_server_time(self) -> datetime:
        """Get OKX server time."""
        self._rate_limit()
        response = self._market_api.get_system_time()
        data = self._handle_response(response, "get_server_time")
        if data:
            ts = int(data[0].get("ts", 0))
            return datetime.fromtimestamp(ts / 1000)
        return datetime.now()
