"""
Data Provider Factory with Fallback Logic
Manages multiple data providers with automatic fallback
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class DataProviderFactory:
    """
    Factory for managing multiple data providers with fallback logic
    Tries providers in order until one succeeds
    """

    def __init__(self, provider_keys: Dict[str, Any] = None):
        """
        Initialize data provider factory

        Args:
            provider_keys: Dict with API keys for different providers
                {
                    "alpaca": {"key": "...", "secret": "..."},
                    "iex": "...",
                    "finnhub": "...",
                    "fmp": "..."
                }
        """
        self.provider_keys = provider_keys or {}
        self.providers = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available providers based on API keys"""

        # Initialize Alpaca if keys available
        if "alpaca" in self.provider_keys and self.provider_keys["alpaca"]:
            try:
                from .alpaca_live import get_alpaca_provider

                alpaca_config = self.provider_keys["alpaca"]
                self.providers["alpaca"] = get_alpaca_provider(
                    api_key=alpaca_config.get("key"),
                    api_secret=alpaca_config.get("secret"),
                    paper=alpaca_config.get("paper", True)
                )
                logger.info("Alpaca provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca: {e}")

        # Initialize IEX Cloud if key available
        if "iex" in self.provider_keys and self.provider_keys["iex"]:
            try:
                from .iex_cloud import get_iex_provider

                self.providers["iex"] = get_iex_provider(
                    api_key=self.provider_keys["iex"],
                    sandbox=False
                )
                logger.info("IEX Cloud provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize IEX Cloud: {e}")

        # Initialize Finnhub if key available
        if "finnhub" in self.provider_keys and self.provider_keys["finnhub"]:
            try:
                from .finnhub import FinnhubProvider

                self.providers["finnhub"] = FinnhubProvider(
                    api_key=self.provider_keys["finnhub"]
                )
                logger.info("Finnhub provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Finnhub: {e}")

        # Initialize FMP if key available
        if "fmp" in self.provider_keys and self.provider_keys["fmp"]:
            try:
                from .fmp import FMPProvider

                self.providers["fmp"] = FMPProvider(
                    api_key=self.provider_keys["fmp"]
                )
                logger.info("FMP provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize FMP: {e}")

        # Always fallback to yfinance (no key required)
        self.providers["yfinance"] = "yfinance"
        logger.info("yfinance provider available as fallback")

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote with automatic fallback

        Tries providers in this order:
        1. Alpaca (if available)
        2. IEX Cloud (if available)
        3. yfinance (always available)

        Args:
            symbol: Stock symbol

        Returns:
            Quote data dict or None
        """
        provider_order = ["alpaca", "iex", "yfinance"]

        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue

            try:
                provider = self.providers[provider_name]

                if provider_name == "alpaca":
                    result = await provider.get_latest_quote(symbol)
                    if result:
                        logger.info(f"Quote from Alpaca: {symbol}")
                        return result

                elif provider_name == "iex":
                    result = await provider.get_quote(symbol)
                    if result:
                        logger.info(f"Quote from IEX Cloud: {symbol}")
                        return result

                elif provider_name == "yfinance":
                    # Fallback to yfinance
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    if info and "regularMarketPrice" in info:
                        logger.info(f"Quote from yfinance: {symbol}")
                        return {
                            "symbol": symbol,
                            "price": info.get("regularMarketPrice"),
                            "change": info.get("regularMarketChange"),
                            "change_percent": info.get("regularMarketChangePercent"),
                            "volume": info.get("volume"),
                            "market_cap": info.get("marketCap"),
                            "source": "yfinance"
                        }

            except Exception as e:
                logger.warning(f"{provider_name} failed for {symbol}: {e}")
                continue

        logger.error(f"All providers failed for {symbol}")
        return None

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data with automatic fallback

        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)

        Returns:
            DataFrame with OHLCV data
        """
        provider_order = ["alpaca", "iex", "yfinance"]

        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue

            try:
                provider = self.providers[provider_name]

                if provider_name == "alpaca":
                    # Convert period to start/end dates
                    from datetime import timedelta
                    end = datetime.now()

                    period_map = {
                        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
                        "6mo": 180, "1y": 365, "2y": 730, "5y": 1825
                    }
                    days = period_map.get(period, 30)
                    start = end - timedelta(days=days)

                    # Map interval to Alpaca timeframe
                    timeframe_map = {
                        "1m": "1Min", "5m": "5Min", "15m": "15Min",
                        "1h": "1Hour", "1d": "1Day"
                    }
                    timeframe = timeframe_map.get(interval, "1Day")

                    result = await provider.get_historical_bars(
                        symbol, start, end, timeframe
                    )

                    if result is not None and not result.empty:
                        logger.info(f"Historical data from Alpaca: {symbol}")
                        return result

                elif provider_name == "iex":
                    # IEX uses different period naming
                    iex_period_map = {
                        "1d": "1d", "5d": "5d", "1mo": "1m", "3mo": "3m",
                        "6mo": "6m", "1y": "1y", "2y": "2y", "5y": "5y"
                    }
                    iex_period = iex_period_map.get(period, "1m")

                    result = await provider.get_historical_prices(symbol, iex_period)

                    if result is not None and not result.empty:
                        logger.info(f"Historical data from IEX Cloud: {symbol}")
                        return result

                elif provider_name == "yfinance":
                    import yfinance as yf

                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period, interval=interval)

                    if not hist.empty:
                        logger.info(f"Historical data from yfinance: {symbol}")
                        return hist

            except Exception as e:
                logger.warning(f"{provider_name} historical data failed for {symbol}: {e}")
                continue

        logger.error(f"All providers failed for historical data: {symbol}")
        return None

    async def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company information with fallback"""
        provider_order = ["iex", "yfinance"]

        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue

            try:
                if provider_name == "iex":
                    provider = self.providers["iex"]
                    result = await provider.get_company_info(symbol)

                    if result:
                        logger.info(f"Company info from IEX Cloud: {symbol}")
                        return result

                elif provider_name == "yfinance":
                    import yfinance as yf

                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    if info:
                        logger.info(f"Company info from yfinance: {symbol}")
                        return {
                            "symbol": symbol,
                            "name": info.get("longName"),
                            "sector": info.get("sector"),
                            "industry": info.get("industry"),
                            "description": info.get("longBusinessSummary"),
                            "website": info.get("website"),
                            "employees": info.get("fullTimeEmployees"),
                            "source": "yfinance"
                        }

            except Exception as e:
                logger.warning(f"{provider_name} company info failed for {symbol}: {e}")
                continue

        return None

    async def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get fundamental data with fallback"""
        provider_order = ["iex", "finnhub", "fmp", "yfinance"]

        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue

            try:
                if provider_name == "iex":
                    provider = self.providers["iex"]
                    stats = await provider.get_key_stats(symbol)

                    if stats:
                        logger.info(f"Fundamentals from IEX Cloud: {symbol}")
                        return stats

                elif provider_name == "yfinance":
                    import yfinance as yf

                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    if info:
                        logger.info(f"Fundamentals from yfinance: {symbol}")
                        return {
                            "symbol": symbol,
                            "market_cap": info.get("marketCap"),
                            "pe_ratio": info.get("trailingPE"),
                            "forward_pe": info.get("forwardPE"),
                            "peg_ratio": info.get("pegRatio"),
                            "price_to_book": info.get("priceToBook"),
                            "dividend_yield": info.get("dividendYield"),
                            "profit_margin": info.get("profitMargins"),
                            "roe": info.get("returnOnEquity"),
                            "debt_to_equity": info.get("debtToEquity"),
                            "source": "yfinance"
                        }

            except Exception as e:
                logger.warning(f"{provider_name} fundamentals failed for {symbol}: {e}")
                continue

        return None


# Singleton instance
_factory_instance: Optional[DataProviderFactory] = None


def get_data_provider_factory(provider_keys: Dict[str, Any] = None) -> DataProviderFactory:
    """Get or create data provider factory instance"""
    global _factory_instance

    if _factory_instance is None or provider_keys:
        _factory_instance = DataProviderFactory(provider_keys)

    return _factory_instance
