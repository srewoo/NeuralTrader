"""
Unit Tests for News Sources
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestNewsAggregator:
    """Test NewsAggregator class"""

    def test_aggregator_initialization(self):
        """Test aggregator initializes correctly"""
        from news.sources import NewsAggregator

        aggregator = NewsAggregator()

        assert aggregator.session is not None
        assert hasattr(aggregator, 'RSS_FEEDS')

    def test_rss_feeds_defined(self):
        """Test RSS feeds are properly defined"""
        from news.sources import NewsAggregator

        assert len(NewsAggregator.RSS_FEEDS) > 0
        assert "economic_times_markets" in NewsAggregator.RSS_FEEDS

    @patch('feedparser.parse')
    def test_fetch_from_rss(self, mock_parse):
        """Test fetching from RSS feed"""
        from news.sources import NewsAggregator

        # Setup mock
        mock_parse.return_value = MagicMock(
            entries=[
                MagicMock(
                    title="Test Article",
                    summary="Test summary",
                    link="https://example.com/article",
                    published_parsed=(2024, 12, 15, 10, 30, 0, 0, 0, 0)
                )
            ]
        )

        aggregator = NewsAggregator()
        articles = aggregator._fetch_from_rss(
            "https://test.com/feed",
            "test_source"
        )

        assert len(articles) == 1
        assert articles[0]["title"] == "Test Article"
        assert articles[0]["source"] == "test_source"

    @patch('feedparser.parse')
    def test_fetch_filters_old_articles(self, mock_parse):
        """Test that old articles are filtered out"""
        from news.sources import NewsAggregator

        # Article older than 30 days
        old_date = datetime.now() - timedelta(days=35)

        mock_parse.return_value = MagicMock(
            entries=[
                MagicMock(
                    title="Old Article",
                    summary="Old summary",
                    link="https://example.com/old",
                    published_parsed=old_date.timetuple()
                )
            ]
        )

        aggregator = NewsAggregator()
        articles = aggregator._fetch_from_rss(
            "https://test.com/feed",
            "test_source"
        )

        assert len(articles) == 0

    @patch('feedparser.parse')
    def test_fetch_latest_news(self, mock_parse):
        """Test fetching latest news from all sources"""
        from news.sources import NewsAggregator

        mock_parse.return_value = MagicMock(
            entries=[
                MagicMock(
                    title="Latest News",
                    summary="Latest summary",
                    link="https://example.com/latest",
                    published_parsed=datetime.now().timetuple()
                )
            ]
        )

        aggregator = NewsAggregator()
        articles = aggregator.fetch_latest_news(limit=5)

        assert isinstance(articles, list)

    @patch('feedparser.parse')
    def test_search_news(self, mock_parse):
        """Test news search functionality"""
        from news.sources import NewsAggregator

        mock_parse.return_value = MagicMock(
            entries=[
                MagicMock(
                    title="Reliance Reports Strong Earnings",
                    summary="Reliance Industries reported...",
                    link="https://example.com/reliance",
                    published_parsed=datetime.now().timetuple()
                ),
                MagicMock(
                    title="Market Update",
                    summary="General market news",
                    link="https://example.com/market",
                    published_parsed=datetime.now().timetuple()
                )
            ]
        )

        aggregator = NewsAggregator()
        results = aggregator.search_news("reliance", days_back=7)

        # Should filter to only Reliance-related articles
        assert all(
            "reliance" in r["title"].lower() or "reliance" in r.get("description", "").lower()
            for r in results
        )

    @patch('feedparser.parse')
    def test_fetch_stock_news(self, mock_parse):
        """Test fetching news for specific stock"""
        from news.sources import NewsAggregator

        mock_parse.return_value = MagicMock(
            entries=[
                MagicMock(
                    title="TCS Wins Major Contract",
                    summary="Tata Consultancy Services...",
                    link="https://example.com/tcs",
                    published_parsed=datetime.now().timetuple()
                )
            ]
        )

        aggregator = NewsAggregator()
        articles = aggregator.fetch_stock_news("TCS", limit=5)

        assert isinstance(articles, list)

    @patch('feedparser.parse')
    def test_get_trending_topics(self, mock_parse):
        """Test trending topics extraction"""
        from news.sources import NewsAggregator

        mock_parse.return_value = MagicMock(
            entries=[
                MagicMock(
                    title="Nifty Hits Record High",
                    summary="Index reaches new peak",
                    link="https://example.com/1",
                    published_parsed=datetime.now().timetuple()
                ),
                MagicMock(
                    title="Nifty Continues Rally",
                    summary="Markets bullish",
                    link="https://example.com/2",
                    published_parsed=datetime.now().timetuple()
                ),
                MagicMock(
                    title="Banking Stocks Lead Nifty Gains",
                    summary="Bank nifty strong",
                    link="https://example.com/3",
                    published_parsed=datetime.now().timetuple()
                )
            ]
        )

        aggregator = NewsAggregator()
        topics = aggregator.get_trending_topics(limit=5)

        assert isinstance(topics, list)
        # 'nifty' should appear in trending topics
        if topics:
            assert "topic" in topics[0]
            assert "count" in topics[0]

    def test_symbol_to_name_mapping(self):
        """Test symbol to company name mapping exists"""
        from news.sources import NewsAggregator

        aggregator = NewsAggregator()

        # Symbol mapping should be used in fetch_stock_news
        # Check by looking at method implementation
        import inspect
        source = inspect.getsource(aggregator.fetch_stock_news)

        assert "symbol_to_name" in source or "RELIANCE" in source


class TestNewsAggregatorSingleton:
    """Test NewsAggregator singleton pattern"""

    def test_get_news_aggregator_returns_instance(self):
        """Test singleton returns instance"""
        from news.sources import get_news_aggregator

        aggregator = get_news_aggregator()

        assert aggregator is not None

    def test_get_news_aggregator_returns_same_instance(self):
        """Test singleton returns same instance"""
        from news.sources import get_news_aggregator

        aggregator1 = get_news_aggregator()
        aggregator2 = get_news_aggregator()

        assert aggregator1 is aggregator2


class TestNewsArticleParsing:
    """Test news article parsing"""

    @patch('feedparser.parse')
    def test_article_has_required_fields(self, mock_parse):
        """Test articles have all required fields"""
        from news.sources import NewsAggregator

        mock_parse.return_value = MagicMock(
            entries=[
                MagicMock(
                    title="Test Article",
                    summary="Test summary",
                    link="https://example.com/test",
                    published_parsed=datetime.now().timetuple()
                )
            ]
        )

        aggregator = NewsAggregator()
        articles = aggregator._fetch_from_rss("https://test.com/feed", "test")

        if articles:
            article = articles[0]
            assert "title" in article
            assert "description" in article
            assert "link" in article
            assert "source" in article
            assert "published" in article

    @patch('feedparser.parse')
    def test_handles_missing_published_date(self, mock_parse):
        """Test handling of missing published date"""
        from news.sources import NewsAggregator

        mock_entry = MagicMock()
        mock_entry.get.side_effect = lambda key, default=None: {
            'title': 'Test',
            'summary': 'Summary',
            'link': 'https://example.com',
            'published_parsed': None
        }.get(key, default)

        mock_parse.return_value = MagicMock(entries=[mock_entry])

        aggregator = NewsAggregator()
        # Should not raise error
        articles = aggregator._fetch_from_rss("https://test.com/feed", "test")

        assert isinstance(articles, list)

    @patch('feedparser.parse')
    def test_handles_feed_parse_error(self, mock_parse):
        """Test handling of feed parsing errors"""
        from news.sources import NewsAggregator

        mock_parse.side_effect = Exception("Parse error")

        aggregator = NewsAggregator()
        articles = aggregator._fetch_from_rss("https://test.com/feed", "test")

        # Should return empty list, not raise
        assert articles == []
