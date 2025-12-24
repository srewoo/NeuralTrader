"""
Angel One Smart API Data Provider
Real-time market data from Angel One broker
Requires: pip install smartapi-python
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from data_providers.base_provider import BaseDataProvider, StockData

logger = logging.getLogger(__name__)

# Check if smartapi is available
SMARTAPI_AVAILABLE = False
try:
    from SmartApi import SmartConnect
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    import pyotp
    SMARTAPI_AVAILABLE = True
except ImportError:
    logger.warning("SmartApi not installed. Install with: pip install smartapi-python pyotp")
    SmartConnect = None
    SmartWebSocketV2 = None
    pyotp = None


class AngelOneProvider(BaseDataProvider):
    """
    Angel One Smart API Data Provider for real-time Indian market data.

    Setup:
    1. Create account at https://www.angelone.in/
    2. Enable Smart API from profile
    3. Get API Key, Client ID, and TOTP Secret
    4. Set environment variables or pass to constructor

    Features:
    - Real-time quotes
    - Historical data (1min to 1day candles)
    - WebSocket streaming
    - Order placement (if enabled)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        password: Optional[str] = None,
        totp_secret: Optional[str] = None
    ):
        """
        Initialize Angel One provider.

        Args:
            api_key: Angel One API key
            client_id: Angel One client ID
            password: Trading password
            totp_secret: TOTP secret for 2FA
        """
        self.api_key = api_key or os.getenv("ANGELONE_API_KEY")
        self.client_id = client_id or os.getenv("ANGELONE_CLIENT_ID")
        self.password = password or os.getenv("ANGELONE_PASSWORD")
        self.totp_secret = totp_secret or os.getenv("ANGELONE_TOTP_SECRET")

        self.smart_api = None
        self.auth_token = None
        self.feed_token = None
        self.is_authenticated = False

        # Symbol token mapping (NSE)
        self._symbol_tokens = {}

        if SMARTAPI_AVAILABLE and self.api_key:
            self._initialize()

    def _initialize(self):
        """Initialize and authenticate with Angel One"""
        if not SMARTAPI_AVAILABLE:
            logger.error("SmartApi library not available")
            return

        if not all([self.api_key, self.client_id, self.password, self.totp_secret]):
            logger.warning("Angel One credentials not fully configured")
            return

        try:
            self.smart_api = SmartConnect(api_key=self.api_key)

            # Generate TOTP
            totp = pyotp.TOTP(self.totp_secret).now()

            # Login
            data = self.smart_api.generateSession(
                self.client_id,
                self.password,
                totp
            )

            if data['status']:
                self.auth_token = data['data']['jwtToken']
                self.feed_token = self.smart_api.getfeedToken()
                self.is_authenticated = True
                logger.info("Angel One authentication successful")

                # Load symbol token mapping
                self._load_symbol_tokens()
            else:
                logger.error(f"Angel One authentication failed: {data.get('message')}")

        except Exception as e:
            logger.error(f"Angel One initialization error: {e}")

    def _load_symbol_tokens(self):
        """Load symbol to token mapping from Angel One"""
        try:
            # Get instrument list
            # In production, you'd download the full instrument list from Angel One
            # For now, we'll use a basic mapping for common stocks
            self._symbol_tokens = {
                # NSE Stocks (Exchange: NSE, Token from Angel One instruments)
                "RELIANCE": {"token": "2885", "exchange": "NSE"},
                "TCS": {"token": "11536", "exchange": "NSE"},
                "INFY": {"token": "1594", "exchange": "NSE"},
                "HDFCBANK": {"token": "1333", "exchange": "NSE"},
                "ICICIBANK": {"token": "4963", "exchange": "NSE"},
                "WIPRO": {"token": "3787", "exchange": "NSE"},
                "ITC": {"token": "1660", "exchange": "NSE"},
                "SBIN": {"token": "3045", "exchange": "NSE"},
                "BHARTIARTL": {"token": "10604", "exchange": "NSE"},
                "KOTAKBANK": {"token": "1922", "exchange": "NSE"},
                "LT": {"token": "11483", "exchange": "NSE"},
                "HINDUNILVR": {"token": "1394", "exchange": "NSE"},
                "AXISBANK": {"token": "5900", "exchange": "NSE"},
                "BAJFINANCE": {"token": "317", "exchange": "NSE"},
                "MARUTI": {"token": "10999", "exchange": "NSE"},
                "ASIANPAINT": {"token": "236", "exchange": "NSE"},
                "SUNPHARMA": {"token": "3351", "exchange": "NSE"},
                "TITAN": {"token": "3506", "exchange": "NSE"},
                "TATAMOTORS": {"token": "3456", "exchange": "NSE"},
                "TATASTEEL": {"token": "3499", "exchange": "NSE"},
                "NTPC": {"token": "11630", "exchange": "NSE"},
                "POWERGRID": {"token": "14977", "exchange": "NSE"},
                "ONGC": {"token": "2475", "exchange": "NSE"},
                "HCLTECH": {"token": "7229", "exchange": "NSE"},
                "TECHM": {"token": "13538", "exchange": "NSE"},
                "ADANIENT": {"token": "25", "exchange": "NSE"},
                "ADANIPORTS": {"token": "15083", "exchange": "NSE"},
                # Index
                "NIFTY": {"token": "99926000", "exchange": "NSE"},
                "BANKNIFTY": {"token": "99926009", "exchange": "NSE"},
                "SENSEX": {"token": "99919000", "exchange": "BSE"},
            }
            logger.info(f"Loaded {len(self._symbol_tokens)} symbol tokens")
        except Exception as e:
            logger.error(f"Failed to load symbol tokens: {e}")

    @property
    def name(self) -> str:
        return "Angel One Smart API"

    @property
    def is_available(self) -> bool:
        return SMARTAPI_AVAILABLE and self.is_authenticated

    async def get_quote(self, symbol: str) -> Optional[StockData]:
        """Get real-time quote from Angel One"""
        if not self.is_available:
            logger.debug("Angel One not available, falling back")
            return None

        symbol_clean = symbol.replace('.NS', '').replace('.BO', '').upper()

        if symbol_clean not in self._symbol_tokens:
            logger.warning(f"Symbol {symbol_clean} not in token mapping")
            return None

        try:
            token_info = self._symbol_tokens[symbol_clean]
            exchange = token_info['exchange']
            token = token_info['token']

            # Get LTP (Last Traded Price) data
            ltp_data = self.smart_api.ltpData(exchange, symbol_clean, token)

            if ltp_data and ltp_data.get('status'):
                data = ltp_data['data']

                return StockData(
                    symbol=symbol_clean,
                    name=symbol_clean,  # Angel One doesn't return name in LTP
                    current_price=float(data.get('ltp', 0)),
                    previous_close=float(data.get('close', data.get('ltp', 0))),
                    change=float(data.get('ltp', 0)) - float(data.get('close', data.get('ltp', 0))),
                    change_percent=0.0,  # Calculate if needed
                    volume=int(data.get('tradingSymbol', 0)) if isinstance(data.get('tradingSymbol'), (int, float)) else 0,
                    high=float(data.get('high', 0)),
                    low=float(data.get('low', 0)),
                    open_price=float(data.get('open', 0)),
                    timestamp=datetime.now(),
                    source="angelone"
                )

        except Exception as e:
            logger.error(f"Angel One quote error for {symbol}: {e}")

        return None

    async def get_historical_data(
        self,
        symbol: str,
        interval: str = "ONE_DAY",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get historical candle data from Angel One.

        Args:
            symbol: Stock symbol
            interval: Candle interval (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE,
                     THIRTY_MINUTE, ONE_HOUR, ONE_DAY)
            from_date: Start date
            to_date: End date

        Returns:
            List of candle data
        """
        if not self.is_available:
            return None

        symbol_clean = symbol.replace('.NS', '').replace('.BO', '').upper()

        if symbol_clean not in self._symbol_tokens:
            return None

        try:
            token_info = self._symbol_tokens[symbol_clean]
            exchange = token_info['exchange']
            token = token_info['token']

            # Default date range
            if not to_date:
                to_date = datetime.now()
            if not from_date:
                from_date = to_date - timedelta(days=365)

            # Angel One interval mapping
            interval_map = {
                "1m": "ONE_MINUTE",
                "5m": "FIVE_MINUTE",
                "15m": "FIFTEEN_MINUTE",
                "30m": "THIRTY_MINUTE",
                "1h": "ONE_HOUR",
                "1d": "ONE_DAY",
                "ONE_DAY": "ONE_DAY",
                "ONE_MINUTE": "ONE_MINUTE",
            }

            angelone_interval = interval_map.get(interval, "ONE_DAY")

            historic_params = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": angelone_interval,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }

            candle_data = self.smart_api.getCandleData(historic_params)

            if candle_data and candle_data.get('status'):
                candles = candle_data.get('data', [])
                return [
                    {
                        "timestamp": candle[0],
                        "open": candle[1],
                        "high": candle[2],
                        "low": candle[3],
                        "close": candle[4],
                        "volume": candle[5]
                    }
                    for candle in candles
                ]

        except Exception as e:
            logger.error(f"Angel One historical data error for {symbol}: {e}")

        return None

    def get_market_depth(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market depth (order book) for a symbol"""
        if not self.is_available:
            return None

        symbol_clean = symbol.replace('.NS', '').replace('.BO', '').upper()

        if symbol_clean not in self._symbol_tokens:
            return None

        try:
            token_info = self._symbol_tokens[symbol_clean]
            exchange = token_info['exchange']
            token = token_info['token']

            # Get full quote with market depth
            quote_data = self.smart_api.getMarketData({
                "mode": "FULL",
                "exchangeTokens": {exchange: [token]}
            })

            if quote_data and quote_data.get('status'):
                return quote_data.get('data', {})

        except Exception as e:
            logger.error(f"Angel One market depth error for {symbol}: {e}")

        return None

    def logout(self):
        """Logout from Angel One session"""
        if self.smart_api and self.is_authenticated:
            try:
                self.smart_api.terminateSession(self.client_id)
                self.is_authenticated = False
                logger.info("Angel One session terminated")
            except Exception as e:
                logger.error(f"Angel One logout error: {e}")


# Global instance
_angelone_provider_instance = None


def get_angelone_provider() -> AngelOneProvider:
    """Get or create global AngelOne provider instance"""
    global _angelone_provider_instance
    if _angelone_provider_instance is None:
        _angelone_provider_instance = AngelOneProvider()
    return _angelone_provider_instance
