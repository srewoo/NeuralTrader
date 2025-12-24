"""
Unit Tests for Celery Background Tasks
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestMarketTasks:
    """Test Market Data Tasks"""

    @patch('tasks.market_tasks.get_all_indian_stocks')
    @patch('tasks.market_tasks.redis.from_url')
    def test_update_market_data(self, mock_redis, mock_get_stocks):
        """Test market data update task"""
        from tasks.market_tasks import update_market_data

        mock_get_stocks.return_value = [
            {"symbol": "RELIANCE", "close": 2500, "change": 20, "volume": 5000000},
            {"symbol": "TCS", "close": 3500, "change": 15, "volume": 3000000}
        ]

        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Create mock task
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"

        result = update_market_data.run()

        assert result["status"] == "success"
        assert result["updated"] == 2

    @patch('tasks.market_tasks.get_all_indian_stocks')
    @patch('tasks.market_tasks.clear_stock_cache')
    def test_refresh_stock_cache(self, mock_clear, mock_get_stocks):
        """Test stock cache refresh task"""
        from tasks.market_tasks import refresh_stock_cache

        mock_get_stocks.return_value = [
            {"symbol": "RELIANCE"},
            {"symbol": "TCS"}
        ]

        result = refresh_stock_cache.run()

        assert result["status"] == "success"
        mock_clear.assert_called_once()

    @patch('yfinance.Ticker')
    @patch('tasks.market_tasks.redis.from_url')
    def test_fetch_live_price(self, mock_redis, mock_ticker):
        """Test single stock price fetch"""
        from tasks.market_tasks import fetch_live_price

        mock_instance = MagicMock()
        mock_instance.info = {
            "regularMarketPrice": 2500,
            "previousClose": 2480,
            "regularMarketChange": 20,
            "volume": 5000000
        }
        mock_ticker.return_value = mock_instance

        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        result = fetch_live_price.run("RELIANCE")

        assert result["symbol"] == "RELIANCE"
        assert result["current_price"] == 2500


class TestAITasks:
    """Test AI Analysis Tasks"""

    @patch('tasks.ai_tasks.get_all_indian_stocks')
    @patch('tasks.ai_tasks.get_ensemble_analyzer')
    @patch('tasks.ai_tasks.get_mongo_client')
    def test_update_predictions(self, mock_mongo, mock_analyzer, mock_stocks):
        """Test AI predictions update task"""
        from tasks.ai_tasks import update_predictions

        mock_stocks.return_value = [
            {"symbol": "RELIANCE"},
            {"symbol": "TCS"}
        ]

        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze_stock = AsyncMock(return_value={
            "recommendation": "BUY",
            "confidence": 0.75
        })
        mock_analyzer.return_value = mock_analyzer_instance

        mock_db = MagicMock()
        mock_mongo.return_value.get_database.return_value = mock_db

        result = update_predictions.run()

        assert result["status"] == "success"

    @patch('tasks.ai_tasks.get_ensemble_analyzer')
    @patch('tasks.ai_tasks.get_mongo_client')
    def test_daily_market_analysis(self, mock_mongo, mock_analyzer):
        """Test daily market analysis task"""
        from tasks.ai_tasks import daily_market_analysis

        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.get_market_overview = AsyncMock(return_value={
            "sentiment": "bullish",
            "key_events": ["RBI policy", "Q3 results"]
        })
        mock_analyzer.return_value = mock_analyzer_instance

        mock_db = MagicMock()
        mock_mongo.return_value.get_database.return_value = mock_db

        result = daily_market_analysis.run()

        assert result["status"] == "success"

    @patch('yfinance.Ticker')
    @patch('tasks.ai_tasks.get_mongo_client')
    def test_track_prediction_accuracy(self, mock_mongo, mock_ticker):
        """Test prediction accuracy tracking"""
        from tasks.ai_tasks import track_prediction_accuracy

        mock_db = MagicMock()
        mock_db.predictions.find.return_value = [
            {
                "symbol": "RELIANCE",
                "prediction": "BUY",
                "created_at": datetime.now()
            }
        ]
        mock_mongo.return_value.get_database.return_value = mock_db

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = MagicMock(
            empty=False,
            iloc=[
                {"Open": 2500, "Close": 2550}
            ]
        )
        mock_ticker.return_value = mock_ticker_instance

        result = track_prediction_accuracy.run()

        # Should return status (may be success or no_predictions)
        assert "status" in result


class TestAlertTasks:
    """Test Alert Checking Tasks"""

    @patch('yfinance.Ticker')
    @patch('tasks.alert_tasks.get_mongo_client')
    def test_check_price_alerts(self, mock_mongo, mock_ticker):
        """Test price alert checking"""
        from tasks.alert_tasks import check_price_alerts

        mock_db = MagicMock()
        mock_db.alerts.find.return_value = [
            {
                "_id": "alert1",
                "symbol": "RELIANCE",
                "alert_type": "price",
                "condition": "above",
                "target_price": 2500,
                "active": True
            }
        ]
        mock_mongo.return_value.get_database.return_value = mock_db

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            "regularMarketPrice": 2600  # Above target
        }
        mock_ticker.return_value = mock_ticker_instance

        result = check_price_alerts.run()

        assert "status" in result

    @patch('tasks.alert_tasks.get_mongo_client')
    def test_trigger_alert(self, mock_mongo):
        """Test alert triggering"""
        from tasks.alert_tasks import trigger_alert
        from bson.objectid import ObjectId

        alert_id = ObjectId()
        mock_db = MagicMock()
        mock_db.alerts.find_one.return_value = {
            "_id": alert_id,
            "symbol": "RELIANCE",
            "target_price": 2600
        }
        mock_mongo.return_value.get_database.return_value = mock_db

        result = trigger_alert.run(str(alert_id), 2650.0, "test_user")

        assert result["status"] == "triggered"


class TestNewsTasks:
    """Test News Fetching Tasks"""

    @patch('tasks.news_tasks.get_news_aggregator')
    @patch('tasks.news_tasks.get_mongo_client')
    @patch('tasks.news_tasks.redis.from_url')
    def test_fetch_latest_news(self, mock_redis, mock_mongo, mock_aggregator):
        """Test latest news fetching"""
        from tasks.news_tasks import fetch_latest_news

        mock_aggregator_instance = MagicMock()
        mock_aggregator_instance.fetch_latest_news.return_value = [
            {
                "title": "Test News",
                "link": "https://example.com/news",
                "source": "test"
            }
        ]
        mock_aggregator.return_value = mock_aggregator_instance

        mock_db = MagicMock()
        mock_db.news.update_one.return_value = MagicMock(upserted_id="new_id")
        mock_mongo.return_value.get_database.return_value = mock_db

        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        result = fetch_latest_news.run()

        assert result["status"] == "success"

    @patch('tasks.news_tasks.get_news_aggregator')
    @patch('tasks.news_tasks.get_mongo_client')
    @patch('tasks.news_tasks.redis.from_url')
    def test_fetch_stock_news(self, mock_redis, mock_mongo, mock_aggregator):
        """Test stock-specific news fetching"""
        from tasks.news_tasks import fetch_stock_news

        mock_aggregator_instance = MagicMock()
        mock_aggregator_instance.fetch_stock_news.return_value = [
            {
                "title": "Reliance Q3 Results",
                "link": "https://example.com/reliance",
                "source": "test"
            }
        ]
        mock_aggregator.return_value = mock_aggregator_instance

        mock_db = MagicMock()
        mock_mongo.return_value.get_database.return_value = mock_db

        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        result = fetch_stock_news.run("RELIANCE")

        assert result["status"] == "success"
        assert result["symbol"] == "RELIANCE"


class TestTaskRetries:
    """Test Task Retry Behavior"""

    @patch('tasks.market_tasks.get_all_indian_stocks')
    def test_market_task_retries_on_error(self, mock_get_stocks):
        """Test that market tasks retry on failure"""
        from tasks.market_tasks import update_market_data

        mock_get_stocks.side_effect = Exception("API Error")

        # Task should raise for retry
        with pytest.raises(Exception):
            update_market_data.run()

    @patch('tasks.news_tasks.get_news_aggregator')
    def test_news_task_retries_on_error(self, mock_aggregator):
        """Test that news tasks retry on failure"""
        from tasks.news_tasks import fetch_latest_news

        mock_aggregator.side_effect = Exception("Fetch Error")

        # Task should raise for retry
        with pytest.raises(Exception):
            fetch_latest_news.run()
