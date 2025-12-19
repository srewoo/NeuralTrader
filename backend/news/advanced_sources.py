"""
Advanced News Sources
Reuters, Bloomberg, Financial Times and other premium sources
"""

import logging
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncio

logger = logging.getLogger(__name__)


class ReutersNewsSource:
    """Reuters news source via RSS feeds"""

    RSS_FEEDS = {
        "business": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
        "markets": "https://www.reuters.com/markets/",
        "companies": "https://www.reuters.com/companies/",
    }

    async def get_articles(self, symbol: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch Reuters articles"""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days)

        try:
            for category, url in self.RSS_FEEDS.items():
                feed = await asyncio.to_thread(feedparser.parse, url)

                for entry in feed.entries[:20]:  # Limit to 20 per feed
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])

                        if pub_date < cutoff_date:
                            continue

                        # Filter by symbol if provided
                        if symbol and symbol.upper() not in entry.title.upper() and symbol.upper() not in entry.get('summary', '').upper():
                            continue

                        articles.append({
                            "title": entry.title,
                            "description": entry.get('summary', ''),
                            "url": entry.link,
                            "source": f"Reuters - {category.title()}",
                            "published_at": pub_date.isoformat(),
                            "symbol": symbol
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse Reuters entry: {e}")
                        continue

            logger.info(f"Fetched {len(articles)} articles from Reuters")
            return articles

        except Exception as e:
            logger.error(f"Failed to fetch Reuters news: {e}")
            return []


class FinancialTimesSource:
    """Financial Times news via RSS"""

    RSS_FEEDS = {
        "markets": "https://www.ft.com/markets?format=rss",
        "companies": "https://www.ft.com/companies?format=rss",
        "global_economy": "https://www.ft.com/global-economy?format=rss",
    }

    async def get_articles(self, symbol: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch Financial Times articles"""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days)

        try:
            for category, url in self.RSS_FEEDS.items():
                feed = await asyncio.to_thread(feedparser.parse, url)

                for entry in feed.entries[:15]:
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])

                        if pub_date < cutoff_date:
                            continue

                        if symbol and symbol.upper() not in entry.title.upper() and symbol.upper() not in entry.get('summary', '').upper():
                            continue

                        articles.append({
                            "title": entry.title,
                            "description": entry.get('summary', ''),
                            "url": entry.link,
                            "source": f"Financial Times - {category.replace('_', ' ').title()}",
                            "published_at": pub_date.isoformat(),
                            "symbol": symbol
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse FT entry: {e}")
                        continue

            logger.info(f"Fetched {len(articles)} articles from Financial Times")
            return articles

        except Exception as e:
            logger.error(f"Failed to fetch FT news: {e}")
            return []


class MarketWatchSource:
    """MarketWatch news via RSS"""

    RSS_FEEDS = {
        "top_stories": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "real_time": "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
        "market_pulse": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    }

    async def get_articles(self, symbol: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch MarketWatch articles"""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days)

        try:
            for category, url in self.RSS_FEEDS.items():
                feed = await asyncio.to_thread(feedparser.parse, url)

                for entry in feed.entries[:20]:
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])

                        if pub_date < cutoff_date:
                            continue

                        if symbol and symbol.upper() not in entry.title.upper() and symbol.upper() not in entry.get('summary', '').upper():
                            continue

                        articles.append({
                            "title": entry.title,
                            "description": entry.get('summary', ''),
                            "url": entry.link,
                            "source": f"MarketWatch - {category.replace('_', ' ').title()}",
                            "published_at": pub_date.isoformat(),
                            "symbol": symbol
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse MarketWatch entry: {e}")
                        continue

            logger.info(f"Fetched {len(articles)} articles from MarketWatch")
            return articles

        except Exception as e:
            logger.error(f"Failed to fetch MarketWatch news: {e}")
            return []


class CNBCNewsSource:
    """CNBC news via RSS"""

    RSS_FEEDS = {
        "top_news": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "investing": "https://www.cnbc.com/id/15839135/device/rss/rss.html",
        "markets": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    }

    async def get_articles(self, symbol: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch CNBC articles"""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days)

        try:
            for category, url in self.RSS_FEEDS.items():
                feed = await asyncio.to_thread(feedparser.parse, url)

                for entry in feed.entries[:20]:
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])

                        if pub_date < cutoff_date:
                            continue

                        if symbol and symbol.upper() not in entry.title.upper() and symbol.upper() not in entry.get('summary', '').upper():
                            continue

                        articles.append({
                            "title": entry.title,
                            "description": entry.get('summary', ''),
                            "url": entry.link,
                            "source": f"CNBC - {category.replace('_', ' ').title()}",
                            "published_at": pub_date.isoformat(),
                            "symbol": symbol
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse CNBC entry: {e}")
                        continue

            logger.info(f"Fetched {len(articles)} articles from CNBC")
            return articles

        except Exception as e:
            logger.error(f"Failed to fetch CNBC news: {e}")
            return []


class AdvancedNewsAggregator:
    """Aggregate news from multiple premium sources"""

    def __init__(self):
        self.reuters = ReutersNewsSource()
        self.ft = FinancialTimesSource()
        self.marketwatch = MarketWatchSource()
        self.cnbc = CNBCNewsSource()

    async def get_all_news(self, symbol: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch news from all sources

        Args:
            symbol: Stock symbol to filter by
            days: Number of days to look back

        Returns:
            List of news articles from all sources
        """
        # Fetch from all sources concurrently
        results = await asyncio.gather(
            self.reuters.get_articles(symbol, days),
            self.ft.get_articles(symbol, days),
            self.marketwatch.get_articles(symbol, days),
            self.cnbc.get_articles(symbol, days),
            return_exceptions=True
        )

        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"News source error: {result}")

        # Sort by date (newest first)
        all_articles.sort(key=lambda x: x.get('published_at', ''), reverse=True)

        logger.info(f"Aggregated {len(all_articles)} total articles")
        return all_articles


# Singleton instances
_reuters_instance = None
_ft_instance = None
_marketwatch_instance = None
_cnbc_instance = None
_aggregator_instance = None


def get_reuters_source() -> ReutersNewsSource:
    """Get Reuters news source instance"""
    global _reuters_instance
    if _reuters_instance is None:
        _reuters_instance = ReutersNewsSource()
    return _reuters_instance


def get_ft_source() -> FinancialTimesSource:
    """Get Financial Times source instance"""
    global _ft_instance
    if _ft_instance is None:
        _ft_instance = FinancialTimesSource()
    return _ft_instance


def get_marketwatch_source() -> MarketWatchSource:
    """Get MarketWatch source instance"""
    global _marketwatch_instance
    if _marketwatch_instance is None:
        _marketwatch_instance = MarketWatchSource()
    return _marketwatch_instance


def get_cnbc_source() -> CNBCNewsSource:
    """Get CNBC source instance"""
    global _cnbc_instance
    if _cnbc_instance is None:
        _cnbc_instance = CNBCNewsSource()
    return _cnbc_instance


def get_advanced_news_aggregator() -> AdvancedNewsAggregator:
    """Get advanced news aggregator instance"""
    global _aggregator_instance
    if _aggregator_instance is None:
        _aggregator_instance = AdvancedNewsAggregator()
    return _aggregator_instance
