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

# Add RAG import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from rag.ingestion import get_ingestion_pipeline
except ImportError:
    get_ingestion_pipeline = None

logger = logging.getLogger(__name__)


class NewsAggregator:
    """
    Aggregates news from multiple sources
    """
    
    # RSS feeds for Indian financial news (FREE, no API key required)
    RSS_FEEDS = {
        "moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
        "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
        "livemint": "https://www.livemint.com/rss/markets",
    }
    
    def __init__(self):
        """Initialize news aggregator"""
        self.session = requests.Session()
        self.ingestion = get_ingestion_pipeline() if get_ingestion_pipeline else None
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
                    
                    # Skip old articles (older than 7 days)
                    if datetime.now() - published_date > timedelta(days=7):
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
                    
                    # RAG Ingestion (Auto-feed news to brain)
                    if self.ingestion:
                        try:
                            self.ingestion.ingest_document(
                                content=f"{article['title']}\n{article['description']}",
                                metadata={
                                    "category": "news", 
                                    "source": source_name,
                                    "url": article['link'],
                                    "date": article['published']
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Failed to ingest article into RAG: {e}")
                    
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

