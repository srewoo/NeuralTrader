"""
Pytest Configuration and Shared Fixtures
Provides common test fixtures for the NeuralTrader test suite
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Generate realistic price data
    base_price = 100.0
    prices = [base_price]
    for _ in range(99):
        change = np.random.normal(0, 2)
        prices.append(prices[-1] + change)

    data = pd.DataFrame({
        'Open': prices,
        'High': [p + np.random.uniform(0, 3) for p in prices],
        'Low': [p - np.random.uniform(0, 3) for p in prices],
        'Close': [p + np.random.uniform(-1, 1) for p in prices],
        'Volume': [int(np.random.uniform(1000000, 5000000)) for _ in prices]
    }, index=dates)

    # Ensure High > Open, Close and Low < Open, Close
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1) + 0.1
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1) - 0.1

    return data


@pytest.fixture
def sample_stock_data():
    """Sample stock data dictionary"""
    return {
        "symbol": "RELIANCE.NS",
        "name": "Reliance Industries Limited",
        "current_price": 2500.0,
        "previous_close": 2480.0,
        "change": 20.0,
        "change_percent": 0.81,
        "volume": 5000000,
        "market_cap": 16000000000000,
        "pe_ratio": 25.5,
        "week_52_high": 2800.0,
        "week_52_low": 2200.0,
        "sector": "Energy",
        "industry": "Oil & Gas Refining & Marketing"
    }


@pytest.fixture
def sample_technical_indicators():
    """Sample technical indicators dictionary"""
    return {
        "rsi": 55.5,
        "macd": 15.2,
        "macd_signal": 12.8,
        "macd_histogram": 2.4,
        "sma_20": 2450.0,
        "sma_50": 2400.0,
        "sma_200": 2350.0,
        "bb_upper": 2550.0,
        "bb_middle": 2450.0,
        "bb_lower": 2350.0,
        "atr": 45.5,
        "obv": 150000000.0,
        "stochastic_k": 65.0,
        "stochastic_d": 60.0,
        "adx": 25.0,
        "cci": 50.0
    }


@pytest.fixture
def sample_technical_signals():
    """Sample technical signals dictionary"""
    return {
        "rsi": "neutral",
        "macd": "bullish",
        "trend": "uptrend",
        "momentum": "neutral",
        "volatility": "normal"
    }


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result from reasoning agent"""
    return {
        "recommendation": "BUY",
        "confidence": 75,
        "entry_price": 2500.0,
        "target_price": 2700.0,
        "stop_loss": 2400.0,
        "risk_reward_ratio": 2.0,
        "time_horizon": "medium_term",
        "reasoning": "1) Price action shows strong support at SMA50. 2) RSI is neutral at 55.5 with room to move higher. 3) MACD is bullish with positive histogram. 4) Risk is limited with stop loss at 2400. 5) Overall technical setup favors buying.",
        "key_risks": ["Market volatility", "Sector headwinds", "Global factors"],
        "key_opportunities": ["Strong momentum", "Sector rotation"],
        "similar_patterns": "Bullish trend continuation",
        "confidence_breakdown": {
            "technical_score": 70,
            "momentum_score": 75,
            "risk_score": 80
        }
    }


@pytest.fixture
def sample_agent_state():
    """Sample state dictionary for agent testing"""
    return {
        "symbol": "RELIANCE.NS",
        "model": "gpt-4.1",
        "provider": "openai",
        "api_key": "test-api-key",
        "agent_steps": []
    }


# ============================================================================
# Candlestick Pattern Fixtures
# ============================================================================

@pytest.fixture
def doji_candle_data():
    """Data that forms a doji pattern"""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    data = pd.DataFrame({
        'Open': [100, 102, 103, 105, 105.5],  # Last candle is doji
        'High': [102, 104, 106, 108, 108.0],
        'Low': [98, 100, 101, 103, 103.0],
        'Close': [101, 103, 105, 107, 105.6],  # Close very close to open
    }, index=dates)
    return data


@pytest.fixture
def hammer_candle_data():
    """Data that forms a hammer pattern"""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    data = pd.DataFrame({
        'Open': [100, 99, 98, 97, 96.0],
        'High': [101, 100, 99, 98, 97.0],
        'Low': [98, 97, 96, 95, 90.0],  # Long lower shadow
        'Close': [99, 98, 97, 96, 96.5],  # Small body at top
    }, index=dates)
    return data


@pytest.fixture
def bullish_engulfing_data():
    """Data that forms a bullish engulfing pattern"""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    data = pd.DataFrame({
        'Open': [100, 102, 101, 100, 97.0],  # Last candle opens below prev close
        'High': [103, 104, 103, 101, 104.0],
        'Low': [99, 100, 99, 98, 96.0],
        'Close': [101, 100, 99, 98, 102.0],  # Last candle closes above prev open
    }, index=dates)
    return data


# ============================================================================
# News & Sentiment Fixtures
# ============================================================================

@pytest.fixture
def sample_positive_article():
    """Sample positive news article"""
    return {
        "title": "Stock Surges on Strong Earnings Beat",
        "description": "The company reported record profits and strong growth, leading to a bullish rally.",
        "source": "Financial Times",
        "published": "2024-01-15T10:00:00Z"
    }


@pytest.fixture
def sample_negative_article():
    """Sample negative news article"""
    return {
        "title": "Stock Crashes After Disappointing Results",
        "description": "Investors worried as company reports losses and declining revenue, triggering a bearish selloff.",
        "source": "Reuters",
        "published": "2024-01-15T11:00:00Z"
    }


@pytest.fixture
def sample_neutral_article():
    """Sample neutral news article"""
    return {
        "title": "Company Maintains Stable Performance",
        "description": "The stock remains unchanged as the company holds steady in flat market conditions.",
        "source": "Bloomberg",
        "published": "2024-01-15T12:00:00Z"
    }


# ============================================================================
# Backtesting Fixtures
# ============================================================================

@pytest.fixture
def sample_equity_curve(sample_ohlcv_data):
    """Sample equity curve for metrics testing"""
    initial_capital = 100000
    dates = sample_ohlcv_data.index

    # Simulate equity curve with some variation
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    equity = [initial_capital]
    for r in returns[1:]:
        equity.append(equity[-1] * (1 + r))

    return pd.Series(equity, index=dates)


@pytest.fixture
def sample_returns(sample_equity_curve):
    """Sample returns series for metrics testing"""
    return sample_equity_curve.pct_change().dropna()


@pytest.fixture
def sample_trades():
    """Sample trades DataFrame for metrics testing"""
    return pd.DataFrame({
        'entry_date': pd.date_range('2024-01-01', periods=10, freq='10D'),
        'exit_date': pd.date_range('2024-01-05', periods=10, freq='10D'),
        'entry_price': [100, 102, 98, 105, 103, 101, 99, 104, 102, 100],
        'exit_price': [105, 100, 102, 108, 100, 106, 102, 101, 107, 103],
        'pnl': [500, -200, 400, 300, -300, 500, 300, -300, 500, 300],
        'pnl_pct': [5.0, -2.0, 4.0, 3.0, -3.0, 5.0, 3.0, -3.0, 5.0, 3.0],
        'duration_days': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    })


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_yfinance():
    """Mock yfinance Ticker"""
    with patch('yfinance.Ticker') as mock:
        ticker = MagicMock()
        ticker.info = {
            'longName': 'Test Company',
            'previousClose': 100.0,
            'marketCap': 1000000000,
            'trailingPE': 20.0,
            'fiftyTwoWeekHigh': 120.0,
            'fiftyTwoWeekLow': 80.0,
            'sector': 'Technology',
            'industry': 'Software'
        }

        # Create sample history data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        hist_data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 200),
            'High': np.random.uniform(100, 110, 200),
            'Low': np.random.uniform(90, 100, 200),
            'Close': np.random.uniform(95, 105, 200),
            'Volume': np.random.randint(1000000, 5000000, 200)
        }, index=dates)
        ticker.history.return_value = hist_data

        mock.return_value = ticker
        yield mock


@pytest.fixture
def mock_openai():
    """Mock OpenAI client"""
    with patch('openai.AsyncOpenAI') as mock:
        client = AsyncMock()

        # Mock completion response
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = '{"recommendation": "BUY", "confidence": 75, "reasoning": "Test reasoning"}'

        client.chat.completions.create = AsyncMock(return_value=completion)
        mock.return_value = client
        yield mock


@pytest.fixture
def mock_chromadb():
    """Mock FAISS vector store (legacy name kept for test compatibility)"""
    mock_store = MagicMock()

    # Mock query results
    mock_store.query.return_value = {
        'ids': [['doc1', 'doc2', 'doc3']],
        'documents': [['Document 1 content', 'Document 2 content', 'Document 3 content']],
        'metadatas': [[{'category': 'patterns'}, {'category': 'strategies'}, {'category': 'patterns'}]],
        'distances': [[0.1, 0.2, 0.3]]
    }
    mock_store.count.return_value = 3
    mock_store.get_collection_info.return_value = {'name': 'test', 'count': 3, 'metadata': {'backend': 'FAISS'}}

    with patch('rag.vector_store.get_vector_store', return_value=mock_store):
        with patch('rag.vector_store.VectorStore', return_value=mock_store):
            yield mock_store


# ============================================================================
# Async Test Helpers
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
