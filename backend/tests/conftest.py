"""
Pytest Configuration and Fixtures
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_mongo_client():
    """Mock MongoDB client"""
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.get_database.return_value = mock_db

    # Mock collections
    mock_db.alerts = MagicMock()
    mock_db.paper_orders = MagicMock()
    mock_db.predictions = MagicMock()
    mock_db.news = MagicMock()
    mock_db.settings = MagicMock()
    mock_db.analyses = MagicMock()

    return mock_client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client"""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True
    mock_redis.delete.return_value = True
    return mock_redis


@pytest.fixture
def sample_stock_data():
    """Sample stock data for testing"""
    return {
        "symbol": "RELIANCE",
        "name": "Reliance Industries Ltd.",
        "current_price": 2500.50,
        "previous_close": 2480.00,
        "change": 20.50,
        "change_percent": 0.83,
        "volume": 5000000,
        "high": 2520.00,
        "low": 2475.00,
        "open": 2485.00,
        "market_cap": 16900000000000,
        "pe_ratio": 25.5,
        "week_52_high": 2850.00,
        "week_52_low": 2200.00,
        "sector": "Energy",
        "industry": "Oil & Gas Refining & Marketing"
    }


@pytest.fixture
def sample_historical_data():
    """Sample historical OHLCV data"""
    import pandas as pd
    import numpy as np

    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(42)

    prices = np.random.uniform(2400, 2600, 100)

    return pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.02, 0.02, 100)),
        'High': prices * (1 + np.random.uniform(0, 0.03, 100)),
        'Low': prices * (1 - np.random.uniform(0, 0.03, 100)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)


@pytest.fixture
def sample_news_articles():
    """Sample news articles for testing"""
    return [
        {
            "title": "Reliance Industries Reports Strong Q3 Results",
            "description": "The company exceeded market expectations with robust revenue growth.",
            "link": "https://example.com/news/1",
            "source": "economic_times",
            "published": datetime.now().isoformat(),
            "published_timestamp": datetime.now().timestamp()
        },
        {
            "title": "Market Update: Nifty Hits New High",
            "description": "Indian stock market indices reached all-time highs today.",
            "link": "https://example.com/news/2",
            "source": "business_standard",
            "published": datetime.now().isoformat(),
            "published_timestamp": datetime.now().timestamp()
        },
        {
            "title": "RBI Holds Interest Rates Steady",
            "description": "The Reserve Bank of India maintains status quo on interest rates.",
            "link": "https://example.com/news/3",
            "source": "ndtv_profit",
            "published": datetime.now().isoformat(),
            "published_timestamp": datetime.now().timestamp()
        }
    ]


@pytest.fixture
def sample_alert():
    """Sample alert for testing"""
    return {
        "user_id": "test_user",
        "symbol": "RELIANCE",
        "alert_type": "price",
        "condition": "above",
        "target_price": 2600.00,
        "message": "Price above 2600",
        "active": True,
        "created_at": datetime.now()
    }


@pytest.fixture
def sample_paper_order():
    """Sample paper trading order"""
    return {
        "user_id": "test_user",
        "symbol": "TCS",
        "side": "BUY",
        "quantity": 10,
        "order_type": "MARKET",
        "price": 3500.00,
        "status": "FILLED",
        "filled_at": datetime.now()
    }


@pytest.fixture
def sample_analysis_result():
    """Sample AI analysis result"""
    return {
        "symbol": "RELIANCE",
        "recommendation": "BUY",
        "confidence": 0.75,
        "target_price": 2750.00,
        "stop_loss": 2350.00,
        "analysis": {
            "technical": {
                "trend": "bullish",
                "rsi": 55,
                "macd": "positive",
                "moving_averages": "above_50dma"
            },
            "fundamental": {
                "pe_ratio": "fair",
                "revenue_growth": "strong",
                "profit_margin": "healthy"
            },
            "sentiment": {
                "news_sentiment": "positive",
                "social_sentiment": "neutral"
            }
        },
        "models_used": ["gpt-4", "gemini-pro", "claude-3"],
        "created_at": datetime.now()
    }


@pytest.fixture
async def async_client():
    """Create async HTTP client for API testing"""
    from httpx import AsyncClient

    # Import app with mock dependencies
    with patch.dict(os.environ, {
        "MONGODB_URL": "mongodb://localhost:27017",
        "REDIS_URL": "redis://localhost:6379",
        "OPENAI_API_KEY": "test-key"
    }):
        from server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client


@pytest.fixture
def mock_yfinance():
    """Mock yfinance for testing"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.info = {
            "regularMarketPrice": 2500.00,
            "previousClose": 2480.00,
            "regularMarketChange": 20.00,
            "regularMarketChangePercent": 0.81,
            "volume": 5000000,
            "dayHigh": 2520.00,
            "dayLow": 2475.00,
            "open": 2485.00,
            "marketCap": 16900000000000,
            "longName": "Reliance Industries Limited",
            "sector": "Energy",
            "industry": "Oil & Gas Refining & Marketing"
        }
        mock_ticker.return_value = mock_instance
        yield mock_ticker


@pytest.fixture
def mock_openai():
    """Mock OpenAI for testing"""
    with patch('openai.ChatCompletion.create') as mock_create:
        mock_create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"recommendation": "BUY", "confidence": 0.75}'
                    )
                )
            ]
        )
        yield mock_create


@pytest.fixture
def mock_feedparser():
    """Mock feedparser for news testing"""
    with patch('feedparser.parse') as mock_parse:
        mock_parse.return_value = MagicMock(
            entries=[
                MagicMock(
                    title="Test News Article",
                    summary="This is a test article about the stock market.",
                    link="https://example.com/news/test",
                    published_parsed=(2024, 1, 15, 10, 30, 0, 0, 0, 0)
                ),
                MagicMock(
                    title="Another Test Article",
                    summary="More market news for testing purposes.",
                    link="https://example.com/news/test2",
                    published_parsed=(2024, 1, 15, 11, 0, 0, 0, 0, 0)
                )
            ]
        )
        yield mock_parse
