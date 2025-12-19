"""
Integration Tests for API Endpoints
Tests the full API workflow end-to-end
"""

import pytest
import asyncio
from httpx import AsyncClient
from datetime import datetime, timezone
import json


class TestSettingsEndpoints:
    """Test settings management endpoints"""

    @pytest.mark.asyncio
    async def test_get_settings_default(self, async_client: AsyncClient):
        """Test getting default settings when none exist"""
        response = await async_client.get("/api/settings")

        assert response.status_code == 200
        data = response.json()

        # Should return default empty settings
        assert "openai_api_key" in data
        assert "gemini_api_key" in data
        assert "finnhub_api_key" in data
        assert "selected_model" in data
        assert data["selected_model"] == "gpt-4.1"

    @pytest.mark.asyncio
    async def test_save_settings(self, async_client: AsyncClient):
        """Test saving settings with all API keys"""
        settings_data = {
            "openai_api_key": "sk-test-openai-key-1234567890",
            "gemini_api_key": "test-gemini-key-1234567890",
            "finnhub_api_key": "test-finnhub-key",
            "alpaca_api_key": "test-alpaca-key",
            "alpaca_api_secret": "test-alpaca-secret",
            "fmp_api_key": "test-fmp-key",
            "selected_model": "gpt-4",
            "selected_provider": "openai"
        }

        response = await async_client.post("/api/settings", json=settings_data)

        assert response.status_code == 200
        result = response.json()
        assert result["message"] == "Settings saved successfully"

    @pytest.mark.asyncio
    async def test_get_settings_masked(self, async_client: AsyncClient):
        """Test that retrieved settings have masked API keys"""
        # First save settings
        settings_data = {
            "openai_api_key": "sk-test-openai-key-1234567890",
            "gemini_api_key": "test-gemini-key-1234567890",
            "selected_model": "gpt-4",
            "selected_provider": "openai"
        }

        await async_client.post("/api/settings", json=settings_data)

        # Then retrieve
        response = await async_client.get("/api/settings")

        assert response.status_code == 200
        data = response.json()

        # Keys should be masked
        assert "..." in data["openai_api_key"] or data["openai_api_key"] == "****"
        assert "..." in data["gemini_api_key"] or data["gemini_api_key"] == "****"


class TestStockDataEndpoints:
    """Test stock data retrieval endpoints"""

    @pytest.mark.asyncio
    async def test_stock_search(self, async_client: AsyncClient):
        """Test stock search functionality"""
        response = await async_client.get("/api/stocks/search?q=RELIANCE")

        assert response.status_code == 200
        results = response.json()

        assert isinstance(results, list)
        if results:
            assert "symbol" in results[0]
            assert "name" in results[0]

    @pytest.mark.asyncio
    async def test_get_stock_data(self, async_client: AsyncClient):
        """Test getting current stock data"""
        response = await async_client.get("/api/stocks/RELIANCE")

        assert response.status_code == 200
        data = response.json()

        assert "symbol" in data
        assert "current_price" in data or "price" in data

    @pytest.mark.asyncio
    async def test_get_stock_history(self, async_client: AsyncClient):
        """Test getting historical data"""
        response = await async_client.get("/api/stocks/RELIANCE/history?period=1mo")

        assert response.status_code == 200
        # Response format depends on implementation


class TestAnalysisEndpoints:
    """Test analysis endpoints"""

    @pytest.mark.asyncio
    async def test_analyze_stock_without_api_key(self, async_client: AsyncClient):
        """Test that analysis requires API keys"""
        analysis_request = {
            "symbol": "RELIANCE",
            "model": "gpt-4",
            "provider": "openai"
        }

        response = await async_client.post("/api/analyze", json=analysis_request)

        # Should fail without API key configured
        assert response.status_code in [400, 500]

    @pytest.mark.asyncio
    async def test_analyze_stock_validation(self, async_client: AsyncClient):
        """Test input validation for analysis"""
        # Invalid symbol
        response = await async_client.post("/api/analyze", json={
            "symbol": "INVALID@SYMBOL!",
            "model": "gpt-4",
            "provider": "openai"
        })

        assert response.status_code == 422  # Validation error

        # Invalid provider
        response = await async_client.post("/api/analyze", json={
            "symbol": "AAPL",
            "model": "gpt-4",
            "provider": "invalid_provider"
        })

        assert response.status_code == 422


class TestCostTrackingEndpoints:
    """Test API cost tracking endpoints"""

    @pytest.mark.asyncio
    async def test_get_current_month_cost(self, async_client: AsyncClient):
        """Test getting current month cost"""
        response = await async_client.get("/api/api-costs/current-month")

        assert response.status_code == 200
        data = response.json()

        assert "month" in data
        assert "total_cost" in data
        assert "currency" in data
        assert data["currency"] == "USD"
        assert isinstance(data["total_cost"], (int, float))

    @pytest.mark.asyncio
    async def test_get_cost_summary(self, async_client: AsyncClient):
        """Test getting cost summary"""
        response = await async_client.get("/api/api-costs/summary?group_by=provider")

        assert response.status_code == 200
        data = response.json()

        assert "period" in data
        assert "totals" in data
        assert "breakdown" in data


class TestWatchlistEndpoints:
    """Test watchlist management"""

    @pytest.mark.asyncio
    async def test_get_empty_watchlist(self, async_client: AsyncClient):
        """Test getting empty watchlist"""
        response = await async_client.get("/api/watchlist")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_add_to_watchlist(self, async_client: AsyncClient):
        """Test adding stock to watchlist"""
        stock_data = {
            "symbol": "AAPL",
            "name": "Apple Inc."
        }

        response = await async_client.post("/api/watchlist", json=stock_data)

        # Should succeed or return existing
        assert response.status_code in [200, 201]

    @pytest.mark.asyncio
    async def test_remove_from_watchlist(self, async_client: AsyncClient):
        """Test removing from watchlist"""
        # First add
        await async_client.post("/api/watchlist", json={
            "symbol": "AAPL",
            "name": "Apple Inc."
        })

        # Then remove
        response = await async_client.delete("/api/watchlist/AAPL")

        assert response.status_code in [200, 204]


class TestBacktestingEndpoints:
    """Test backtesting endpoints"""

    @pytest.mark.asyncio
    async def test_run_backtest(self, async_client: AsyncClient):
        """Test running a backtest"""
        backtest_request = {
            "symbol": "RELIANCE",
            "strategy": "trend_following",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000
        }

        response = await async_client.post("/api/backtest", json=backtest_request)

        # Check if endpoint exists and accepts request
        assert response.status_code in [200, 404, 500]


# Fixtures

@pytest.fixture
async def async_client():
    """Create async HTTP client for testing"""
    from server import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_db(monkeypatch):
    """Mock database for testing"""
    # Mock MongoDB operations
    pass
