"""
News Sources Aggregator
Fetches news from multiple sources (REAL API calls)
"""

import feedparser
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import sys
import os
import ssl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = logging.getLogger(__name__)

# Configure SSL context globally for feedparser (which uses urllib internally)
# This fixes SSL certificate verification issues with some RSS feeds
ssl._create_default_https_context = ssl._create_unverified_context


class NewsAggregator:
    """
    Aggregates news from multiple sources
    """
    
    # RSS feeds for Indian financial news (FREE, no API key required)
    # Updated with more reliable and current feeds
    RSS_FEEDS = {
        # Economic Times - Markets section (most reliable)
        "economic_times_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "economic_times_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "economic_times_news": "https://economictimes.indiatimes.com/news/rssfeeds/1715249553.cms",
        # Business Standard
        "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
        # NDTV Profit
        "ndtv_profit": "https://feeds.feedburner.com/ndtvprofit-latest",
        # Yahoo Finance India
        "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
        # Reuters Business
        "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
        # Google News - India Business
        "google_news_india": "https://news.google.com/rss/search?q=indian+stock+market&hl=en-IN&gl=IN&ceid=IN:en",
    }
    
    def __init__(self):
        """Initialize news aggregator"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_latest_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch latest financial news (REAL API CALL)
        
        Args:
            symbol: Optional stock symbol to filter
            limit: Maximum number of articles
            
        Returns:
            List of news articles
        """
        all_articles = []
        
        # Fetch from all RSS feeds
        for source_name, feed_url in self.RSS_FEEDS.items():
            try:
                articles = self._fetch_from_rss(feed_url, source_name)
                all_articles.extend(articles)
            except Exception as e:
                logger.warning(f"Failed to fetch from {source_name}: {e}")
        
        # Filter by symbol if provided
        if symbol:
            symbol_clean = symbol.replace('.NS', '').replace('.BO', '').upper()
            all_articles = [
                article for article in all_articles
                if symbol_clean.lower() in article['title'].lower() or
                   symbol_clean.lower() in article.get('description', '').lower()
            ]
        
        # Sort by date (newest first)
        all_articles.sort(key=lambda x: x['published'], reverse=True)
        
        return all_articles[:limit]
    
    def _fetch_from_rss(self, feed_url: str, source_name: str) -> List[Dict[str, Any]]:
        """
        Fetch articles from RSS feed (REAL API CALL)
        
        Args:
            feed_url: RSS feed URL
            source_name: Source name
            
        Returns:
            List of articles
        """
        try:
            # Parse RSS feed (REAL API CALL)
            feed = feedparser.parse(feed_url)
            
            articles = []
            for entry in feed.entries[:20]:  # Get up to 20 from each source
                try:
                    # Parse published date
                    published = entry.get('published_parsed')
                    if published:
                        published_date = datetime(*published[:6])
                    else:
                        published_date = datetime.now()
                    
                    # Skip old articles (older than 30 days - increased from 7)
                    if datetime.now() - published_date > timedelta(days=30):
                        continue
                    
                    article = {
                        "title": entry.get('title', ''),
                        "description": entry.get('summary', entry.get('description', '')),
                        "link": entry.get('link', ''),
                        "source": source_name,
                        "published": published_date.isoformat(),
                        "published_timestamp": published_date.timestamp()
                    }
                    
                    articles.append(article)

                except Exception as e:
                    logger.debug(f"Failed to parse entry: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from {source_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []
    
    def search_news(
        self,
        query: str,
        days_back: int = 7,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search news by query
        
        Args:
            query: Search query
            days_back: Number of days to look back
            limit: Maximum results
            
        Returns:
            List of matching articles
        """
        all_articles = self.fetch_latest_news(limit=100)
        
        # Filter by query
        query_lower = query.lower()
        matching = [
            article for article in all_articles
            if query_lower in article['title'].lower() or
               query_lower in article.get('description', '').lower()
        ]
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        matching = [
            article for article in matching
            if datetime.fromisoformat(article['published']) >= cutoff_date
        ]
        
        return matching[:limit]
    
    def fetch_stock_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch news specifically for a stock using Google News RSS

        Args:
            symbol: Stock symbol
            limit: Maximum articles

        Returns:
            List of articles for the stock
        """
        symbol_clean = symbol.replace('.NS', '').replace('.BO', '').upper()

        # Map common symbols to company names for better search
        symbol_to_name = {
            'RELIANCE': 'Reliance Industries',
            'TCS': 'Tata Consultancy Services TCS',
            'INFY': 'Infosys',
            'HDFCBANK': 'HDFC Bank',
            'ICICIBANK': 'ICICI Bank',
            'WIPRO': 'Wipro',
            'BHARTIARTL': 'Bharti Airtel',
            'ITC': 'ITC Limited',
            'SBIN': 'State Bank of India SBI',
            'LT': 'Larsen Toubro',
            'KOTAKBANK': 'Kotak Mahindra Bank',
            'HINDUNILVR': 'Hindustan Unilever HUL',
            'ASIANPAINT': 'Asian Paints',
            'MARUTI': 'Maruti Suzuki',
            'BAJFINANCE': 'Bajaj Finance',
            'AXISBANK': 'Axis Bank',
            'SUNPHARMA': 'Sun Pharma',
            'TITAN': 'Titan Company',
            'ULTRACEMCO': 'UltraTech Cement',
            'ONGC': 'ONGC Oil Natural Gas',
            'NTPC': 'NTPC Power',
            'POWERGRID': 'Power Grid Corporation',
            'TATAMOTORS': 'Tata Motors',
            'TATASTEEL': 'Tata Steel',
            'ADANIENT': 'Adani Enterprises',
            'ADANIPORTS': 'Adani Ports',
            'COALINDIA': 'Coal India',
            'HCLTECH': 'HCL Technologies',
            'TECHM': 'Tech Mahindra',
        }

        search_term = symbol_to_name.get(symbol_clean, symbol_clean)

        # Google News RSS for specific stock
        google_news_url = f"https://news.google.com/rss/search?q={search_term}+stock+NSE&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            articles = self._fetch_from_rss(google_news_url, "google_news")

            # If no results, try broader search
            if not articles:
                google_news_url = f"https://news.google.com/rss/search?q={symbol_clean}&hl=en-IN&gl=IN&ceid=IN:en"
                articles = self._fetch_from_rss(google_news_url, "google_news")

            return articles[:limit]
        except Exception as e:
            logger.error(f"Failed to fetch stock news for {symbol}: {e}")
            return []

    def get_trending_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending topics from recent news
        
        Args:
            limit: Number of topics
            
        Returns:
            List of trending topics with counts
        """
        articles = self.fetch_latest_news(limit=100)
        
        # Extract keywords from titles
        word_counts = {}
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        for article in articles:
            words = article['title'].lower().split()
            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum())
                if len(word) > 3 and word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by count
        trending = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"topic": word, "count": count}
            for word, count in trending[:limit]
        ]


# Global instance
_news_aggregator_instance = None


def get_news_aggregator() -> NewsAggregator:
    """Get or create global NewsAggregator instance"""
    global _news_aggregator_instance
    if _news_aggregator_instance is None:
        _news_aggregator_instance = NewsAggregator()
    return _news_aggregator_instance

