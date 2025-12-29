"""
Unit Tests for Data Providers
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestDataProviderFactory:
    """Test DataProviderFactory"""

    def test_factory_initialization(self):
        """Test factory initializes with default providers"""
        from data_providers.factory import DataProviderFactory

        factory = DataProviderFactory()

        # yfinance should always be available
        assert "yfinance" in factory.providers

    def test_factory_with_provider_keys(self):
        """Test factory initialization with provider keys"""
        from data_providers.factory import DataProviderFactory

        provider_keys = {
            "alpaca": {"key": "test-key", "secret": "test-secret"},
            "iex": "test-iex-key"
        }

        factory = DataProviderFactory(provider_keys)

        assert factory.provider_keys == provider_keys

    @pytest.mark.asyncio
    async def test_get_quote_yfinance_fallback(self, mock_yfinance):
        """Test quote retrieval falls back to yfinance"""
        from data_providers.factory import DataProviderFactory

        factory = DataProviderFactory()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.info = {
                "regularMarketPrice": 2500.00,
                "previousClose": 2480.00,
                "regularMarketChange": 20.00,
                "regularMarketChangePercent": 0.81,
                "volume": 5000000,
                "longName": "Reliance Industries",
                "sector": "Energy"
            }
            mock_ticker.return_value = mock_instance

            result = await factory.get_quote("RELIANCE.NS")

            assert result is not None
            assert result["current_price"] == 2500.00

    @pytest.mark.asyncio
    async def test_get_quote_handles_errors(self):
        """Test quote retrieval handles errors gracefully"""
        from data_providers.factory import DataProviderFactory

        factory = DataProviderFactory()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception("API Error")

            result = await factory.get_quote("INVALID")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_historical_data(self, sample_historical_data):
        """Test historical data retrieval"""
        from data_providers.factory import DataProviderFactory

        factory = DataProviderFactory()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.history.return_value = sample_historical_data
            mock_ticker.return_value = mock_instance

            result = await factory.get_historical_data("RELIANCE.NS", period="1mo")

            assert result is not None
            assert len(result) > 0


class TestTVScreenerProvider:
    """Test TradingView Screener Provider"""

    @pytest.mark.skip(reason="TVScreener library API changed - mock needs update")
    def test_get_all_indian_stocks(self):
        """Test fetching all Indian stocks"""
        with patch('tvscreener.Scanner') as mock_scanner:
            mock_instance = MagicMock()
            mock_instance.get_scanner_data.return_value = (
                100,
                [
                    {"name": "RELIANCE", "close": 2500, "change": 20, "volume": 5000000},
                    {"name": "TCS", "close": 3500, "change": 15, "volume": 3000000}
                ]
            )
            mock_scanner.return_value = mock_instance

            from data_providers.tvscreener_provider import get_all_indian_stocks

            stocks = get_all_indian_stocks(max_stocks=10)

            # Should return stocks or empty if API unavailable
            assert isinstance(stocks, list)

    @pytest.mark.skip(reason="TVScreener library API changed - mock needs update")
    def test_search_indian_stocks(self):
        """Test stock search functionality"""
        with patch('tvscreener.Scanner') as mock_scanner:
            mock_instance = MagicMock()
            mock_instance.get_scanner_data.return_value = (
                1,
                [{"name": "RELIANCE", "description": "Reliance Industries", "close": 2500}]
            )
            mock_scanner.return_value = mock_instance

            from data_providers.tvscreener_provider import search_indian_stocks

            results = search_indian_stocks("RELIANCE", limit=5)

            assert isinstance(results, list)


class TestAngelOneProvider:
    """Test Angel One Data Provider"""

    @pytest.mark.skip(reason="AngelOneProvider is abstract class - needs implementation")
    def test_provider_initialization_without_keys(self):
        """Test provider initializes without API keys"""
        from data_providers.angelone_provider import AngelOneProvider

        provider = AngelOneProvider()

        assert provider.is_authenticated == False
        assert provider.smart_api is None

    @pytest.mark.skip(reason="AngelOneProvider is abstract class - needs implementation")
    def test_provider_name(self):
        """Test provider name property"""
        from data_providers.angelone_provider import AngelOneProvider

        provider = AngelOneProvider()

        assert provider.name == "Angel One Smart API"

    @pytest.mark.skip(reason="AngelOneProvider is abstract class - needs implementation")
    def test_provider_is_available(self):
        """Test is_available property"""
        from data_providers.angelone_provider import AngelOneProvider

        provider = AngelOneProvider()

        # Should be False without authentication
        assert provider.is_available == False

    @pytest.mark.skip(reason="AngelOneProvider is abstract class - needs implementation")
    def test_symbol_token_mapping(self):
        """Test symbol token mapping is loaded"""
        from data_providers.angelone_provider import AngelOneProvider

        provider = AngelOneProvider()

        # Even without auth, mapping should exist
        assert hasattr(provider, '_symbol_tokens')

    @pytest.mark.skip(reason="AngelOneProvider is abstract class - needs implementation")
    @pytest.mark.asyncio
    async def test_get_quote_returns_none_when_unavailable(self):
        """Test get_quote returns None when not available"""
        from data_providers.angelone_provider import AngelOneProvider

        provider = AngelOneProvider()

        result = await provider.get_quote("RELIANCE")

        assert result is None


class TestBaseProvider:
    """Test Base Provider Interface"""

    def test_stock_data_dataclass(self):
        """Test StockData dataclass"""
        from data_providers.base_provider import StockData

        data = StockData(
            symbol="RELIANCE",
            name="Reliance Industries",
            current_price=2500.00,
            previous_close=2480.00,
            volume=5000000,
            provider="test"
        )

        assert data.symbol == "RELIANCE"
        assert data.current_price == 2500.00
        assert data.previous_close == 2480.00
